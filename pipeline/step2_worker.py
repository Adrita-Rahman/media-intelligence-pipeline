#!/usr/bin/env python3
"""
The Step 2 media worker: a child process that does everything which can hang.

Why this exists
---------------
The first watchdog was a check between units of work. It could only fire when
control came back to it, so it could not stop a call that never returned. The
run of 22 Sep 2026 proved the point three times:

    V9  export_1754731689768  ceiling  358s   abandoned at  537s
    V13 IMG_2956              ceiling  340s   abandoned at 2036s
    V14 IMG_2963              ceiling  359s   abandoned at 5137s   (14x over)
    V7  IMG_1147              ceiling 3615s   abandoned at 7495s

V7 and V13 stalled inside one easyocr call; V14 stalled inside a Drive read.
The stall is not OCR-specific, so the fix cannot be OCR-specific either: an
in-process guard of any shape - a flag, a signal handler, an async exception -
still needs the interpreter to reach a bytecode boundary, and a thread parked
in torch or in a blocking socket read does not.

So the work that can hang runs HERE, in a separate process, and the parent
enforces the ceiling by killing that process and everything it spawned. The
ceiling stops being a hope and becomes a fact.

Two layers, deliberately:

    cooperative   this process carries the same ceiling and raises a normal
                  AssetFailed the moment it notices - a clean abort that names
                  the stage it died in, and leaves the worker reusable
    hard          the parent kills the process tree at ceiling + grace, which
                  works whether or not this process is capable of noticing

The worker is long-lived and takes one job at a time, so easyocr and
faster-whisper load their models once for the whole run rather than once per
asset. A killed worker is simply replaced before the next asset.

Protocol
--------
Length-prefixed pickle, parent -> child on stdin, child -> parent on fd 1.

fd 1 is duplicated away before anything else is imported, and the real stdout
is pointed at stderr. ffmpeg, torch and easyocr all write to fd 1 sooner or
later; none of that may land in the middle of a pickle frame. Everything
human-readable therefore goes to stderr, which the parent inherits.

Not run by hand. step2_analyse.py spawns it.
"""
from __future__ import annotations

import os

import sys
import time

# ---------------------------------------------------------------------------
# Claim fd 1 for the protocol BEFORE importing anything that might write to it.
# ---------------------------------------------------------------------------
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
_PROTO = os.fdopen(_PROTO_FD, "wb", buffering=0)
sys.stdout = sys.stderr

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_worker_proto as proto  # noqa: E402
import step2_analyse as s2  # noqa: E402

import datetime as _dt  # noqa: E402


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def send(obj):
    proto.send_to(_PROTO, obj)


def recv(stream):
    return proto.recv(stream)


# ---------------------------------------------------------------------------
# the cooperative half of the watchdog
# ---------------------------------------------------------------------------

class StageClock:
    """A budget per stage, announced to the parent as each stage opens.

    The parent cannot size a ceiling it does not understand: OCR's budget
    depends on how many frames there turned out to be, which only this side
    knows. So each stage says what it is and what it is allowed, and the
    parent times THAT, resetting its knife at every boundary.

    check() is still best-effort by nature. It is kept because when it fires
    it raises StageOverBudget, which the OCR stage can catch and degrade from
    - a kill cannot be degraded from, it can only be recovered from. The
    parent's ceiling is this one plus a grace period, so the stage always
    gets the chance to stand down before it is put down.
    """

    def __init__(self, item):
        self.item = item
        self.stage = None
        self.limit = None
        self.t0 = time.time()
        # One entry per stage, with real wall-clock bounds. Without this every
        # STEP2 row for an asset carried the same started_at - the moment the
        # asset began - so "1019s" could be read as the cost of any stage you
        # liked, and no budget could be audited against what it actually cost.
        self.timings = []

    @property
    def elapsed(self):
        return time.time() - self.t0

    def _close(self):
        if self.stage is not None and self.timings:
            last = self.timings[-1]
            if last["completed_at"] is None:
                last["completed_at"] = _utcnow()
                last["elapsed_seconds"] = round(self.elapsed, 3)

    def open(self, stage, limit):
        self._close()
        self.stage, self.limit, self.t0 = stage, float(limit), time.time()
        self.timings.append({"stage": stage, "started_at": _utcnow(),
                             "completed_at": None,
                             "budget_seconds": round(self.limit, 1),
                             "elapsed_seconds": None})
        send(("stage", stage, self.limit))

    def finish(self):
        """Close the stage still open, and hand back the whole record."""
        self._close()
        return self.timings

    def check(self, stage=None):
        if self.limit is not None and self.elapsed > self.limit:
            raise s2.StageOverBudget(self.stage, self.elapsed, self.limit)

    def heartbeat(self, stage, done, total, every=10):
        if done % every == 0 or done == total:
            send(("progress", stage, done, total, self.elapsed))


# ---------------------------------------------------------------------------
# one job
# ---------------------------------------------------------------------------

def run_job(job):
    """Download, then measure. Raises AssetFailed like the in-process path."""
    import step1_ingest as s1

    cfg = job["cfg"]
    deadline = StageClock(job["item"])

    # OCR_ONLY needs no Drive and no media: it re-reads frames that are
    # already on disk from the attempt whose OCR was killed.
    if job["kind"] == "OCR_ONLY":
        out = ocr_only(job, cfg, deadline)
        out["stage_timings"] = deadline.finish()
        return out

    asset = job["asset"]
    tmp_dir = Path(job["tmp_dir"])
    media = tmp_dir / job["media_name"]

    deadline.open("download", s2.fixed_budget(cfg))
    if s2.is_local(asset):
        # The original is on a mounted share (the NAS): copy it, no Drive.
        deadline.check("download")
        attempts = s2.fetch_original(s1, None, asset, media)
    else:
        if not asset.get("google_file_id"):
            raise s2.AssetFailed(
                "no google_file_id; this asset was created locally and its bytes "
                "are not retrievable from Drive")
        service = _drive_service(s1)
        attempts = download_with_deadline(s1, service, asset["google_file_id"],
                                          media, deadline)

    if job["kind"] == "VIDEO":
        out = s2.process_video(
            asset, media, tmp_dir, cfg, deadline=deadline,
            on_checkpoint=lambda c: send(("checkpoint", c)))
    elif job["kind"] == "REPAIR":
        out = reextract(job, asset, cfg, tmp_dir, media, deadline)
    else:
        out = s2.process_image(asset, media, tmp_dir, cfg)

    out["download_attempts"] = attempts
    out["stage_timings"] = deadline.finish()
    return out


def ocr_only(job, cfg, deadline):
    """Read on-screen text from frames already on disk.

    This is the second half of the kill path. When OCR is killed outright the
    frames survive - they are files in the asset's temp directory, and the
    parent does not sweep that directory until the asset is finished - so a
    fresh worker can read the representative ones without re-downloading the
    video or re-cutting a single frame.
    """
    # The STEP2_OCR 3.0.0 engine and rules, exactly as the full run uses them
    # (step2_analyse.ocr3_*); there is no second, older reader here.
    frames = job["frames"]
    deadline.open("ocr_sampled", s2.ocr_budget(len(frames) or 1, cfg))
    return s2.ocr3_read_paths(frames, cfg, deadline, "ocr_sampled")


def reextract(job, asset, cfg, tmp_dir, media, deadline):
    """Re-cut named frames of a video whose keyframe rows already exist.

    Used when an asset measured correctly but an upload dropped, so the only
    thing missing is the object behind one row. The ffmpeg command is
    s2.extract_frames' - the same seek, the same scale, the same qscale - so
    the bytes that land are the bytes the first run would have landed.
    """
    size = tuple(job["size"])
    times = [float(f["timestamp"]) for f in job["frames"]]
    deadline.open("extract_frames", s2.fixed_budget(cfg))
    files, kept, notes = s2.extract_frames(
        media, tmp_dir / "repair", times, size,
        s2.need(cfg, "frame_jpeg_quality"),
        float(s2.need(cfg, "frame_interval_seconds")), deadline=deadline)

    # extract_frames tolerates a missing FINAL sample, because the last
    # planned time of a video can legitimately sit past its last coded frame.
    # Here every time asked for already produced a frame once, so a miss is a
    # failure, not a tolerated gap.
    if len(files) != len(times):
        raise s2.AssetFailed(
            "re-extraction produced {} frame(s) for {} requested timestamp(s)"
            "{}".format(len(files), len(times),
                        "; " + "; ".join(notes) if notes else ""))

    out = []
    for spec, t, path in zip(job["frames"], kept, files):
        s2.decode_check(path, size[0], size[1])
        out.append({"ordinal": spec["ordinal"], "keyframe_id": spec["keyframe_id"],
                    "timestamp": round(t, 3), "path": str(path),
                    "width": size[0], "height": size[1]})
    return {"repaired": out}


_SERVICE = {}


def _drive_service(s1):
    """Built once per worker. A killed worker rebuilds it, which is also how a
    connection poisoned by the stall that got it killed gets thrown away."""
    if "s" not in _SERVICE:
        _SERVICE["s"], _SERVICE["email"] = s1.build_drive_service()
    return _SERVICE["s"]


def download_with_deadline(s1, service, file_id, dest, deadline,
                           attempts=4, socket_timeout=180):
    """download_with_retry, with the ceiling checked between attempts.

    The socket default bounds a single read, not a transfer: V14's fourth
    attempt satisfied every 180s read and still ran for well over an hour,
    because a resumable download is thousands of reads and none of them was
    individually late. Checking here caps the retry ladder; the parent's kill
    caps a single attempt that never ends.
    """
    import socket
    last = None
    previous = socket.getdefaulttimeout()
    for attempt in range(1, attempts + 1):
        deadline.check("download")
        try:
            if dest.exists():
                dest.unlink()
            socket.setdefaulttimeout(socket_timeout)
            try:
                s1.drive_download(service, file_id, dest)
            finally:
                socket.setdefaulttimeout(previous)
            return attempt
        except Exception as exc:
            last = exc
            if attempt < attempts:
                wait = 2 ** attempt
                send(("log", "download attempt {} failed ({}); retrying in {}s"
                      .format(attempt, type(exc).__name__, wait)))
                time.sleep(wait)
    raise s2.AssetFailed("download failed after {} attempts: {}:{}".format(
        attempts, type(last).__name__, str(last)[:200]))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main():
    stdin = sys.stdin.buffer
    send(("hello", os.getpid()))
    while True:
        job = recv(stdin)
        if job is None:
            return 0
        try:
            out = run_job(job)
            send(("result", out))
        except s2.StageOverBudget as exc:
            # OCR catches this and degrades. Anywhere else there is nothing
            # partial worth keeping, so it ends the asset - but cleanly, by
            # this process's own choice, naming the stage, and without the
            # parent having to kill anything.
            send(("failed",
                  "watchdog: {} exceeded its {:.0f}s ceiling at stage {!r} "
                  "({:.0f}s elapsed). Abandoned so the run continues."
                  .format(job.get("item"), exc.limit, exc.stage, exc.elapsed),
                  True))
        except s2.AssetFailed as exc:
            send(("failed", str(exc), True))
        except BaseException as exc:          # noqa: BLE001 - reported, not swallowed
            import traceback
            traceback.print_exc()
            send(("failed", "unexpected:{}:{}".format(
                type(exc).__name__, exc), False))


if __name__ == "__main__":
    raise SystemExit(main())
