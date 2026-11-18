#!/usr/bin/env python3
"""
run_ml_test (ml_test controller, 29 Oct 2026): one command takes a batch of NAS
files through Steps 1-4 into ml_test, each step starting as soon as the one
before it finishes. It calls the existing step scripts and edge functions
unchanged; it adds no analysis of its own.

    plan   list the media files below a folder (--folder X, repeatable; --all = the
           whole share) that ml_test does not have yet and write batch manifests
           (item codes B<n>-V0001 / B<n>-P0001). --settle-minutes N leaves out files
           changed in the last N minutes (maybe still copying). The manifest holds
           file names and paths (patient names), so it is written outside git, to
           /Volumes/WorkDrive/batches. Prints counts only.
    run    take a manifest through every step, in order (2.0.0, 9 Nov 2026: exactly
           as the whole-library run of 2-9 Nov):
             STEP1   step1_ingest.py --manifest --commit (--remaining-only on a
                     re-run); then the MD5 of each cached original goes into
                     source_files.metadata.md5, so the same file can later be
                     matched on Google Drive (which reports MD5) without
                     downloading it again. A few failed files (duplicates, broken)
                     do not stop the rest.
             STEP2   step2_analyse.py --commit --only <20 videos> (scenes, keyframes,
                     speech by MLX Whisper large-v3-turbo, OCR with the clinic word
                     list), a second pass for what failed; step2_stills.py --commit;
                     step2_stills_ocr.py --engine apple_vision; then classify_videos.py
                     (RAW/EDITED), link_people_from_text() and ref_refresh()
             STEP3   step3_v3_supervise.py: prompt v3lib through ml-test-media-analyse
                     (DeepSeek) for every file still without a v3lib reading; spend
                     caps and the balance guard stop it
             STEP4   step4_v3_embed.py: gte-small for every description still to embed
           Steps 3 and 4 take the whole library's leftovers, not only this batch.
           Each step works out what is still to do from the database, so a run
           that stopped (power cut, unplugged drive, spend cap) is resumed by
           running the same command again: nothing is done twice.
           --dry-run prints what each step would do and changes nothing.

    python scripts_content_v1/run_ml_test.py plan --folder "KLINIK DR INDER/2024" --limit 200
    python scripts_content_v1/run_ml_test.py plan --all --settle-minutes 60 --batch-size 500
    python scripts_content_v1/run_ml_test.py run --manifest /Volumes/WorkDrive/batches/batch2_manifest.json --dry-run
    python scripts_content_v1/run_ml_test.py run --manifest /Volumes/WorkDrive/batches/batch2_manifest.json

The nightly 9 pm run (nightly.py) does plan --all and run for every new file.

Rules: the NAS is only read. The DeepSeek spend caps in ml_test.pipeline_config
(ai_gate) stop Step 3, and the controller stops with them. The runner token is
read from ml_test.pipeline_config and only ever sent as a request header; it is
never printed or logged. Logs and the report go next to the manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.environ.setdefault("PIPELINE_TARGET", "ml_test") != "ml_test":
    raise SystemExit("run_ml_test only runs with PIPELINE_TARGET=ml_test")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import target as target_mod  # noqa: E402

VERSION = "run_ml_test/2.0.0"
PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
NAS_ROOT = "/Volumes/MediaShare"
BATCH_DIR = Path("/Volumes/WorkDrive/batches")
# 9 Nov 2026 (2.0.0): every step runs exactly as in the whole-library run of 2-9 Nov:
#   Step 2  config ml_test_step2_2026_11_03_a (clinic OCR word list), speech by Whisper large-v3-turbo
#           on the GPU (MLX), videos 20 at a time with a second pass for a chunk that failed, photo OCR
#           by Apple Vision, then RAW/EDITED labels, people from text and the reference-library refresh
#   Step 3  step3_v3_supervise.py: prompt v3lib, pauses through DeepSeek/Supabase hiccups
#   Step 4  step4_v3_embed.py: gte-small, exact ids, 8 calls side by side
# Steps 3 and 4 take every file that still needs them, so a stopped earlier run is finished too.
STEP2_CONFIG_VERSION = "ml_test_step2_2026_11_03_a"
STEP2_ENV = {"STEP2_WHISPER_ENGINE": "mlx", "STEP2_WHISPER_MLX_REPO": "mlx-community/whisper-large-v3-turbo",
             "STEP2_SHARED_WRITERS": "1", "PIPELINE_TARGET": "ml_test",
             # 18 Nov 2026 (user: yes): the full pre-flight probe reads the whole library and
             # ran before every write, about 5 times per video; at ~102k files each probe took
             # 60-80 s, so a video took ~5 min instead of ~1. The light gate (step2_writer.py,
             # 6 Nov) still checks the Step 2 row counts before every write and runs the full
             # probe once every 100 writes. Analysis is unchanged.
             "STEP2_LIGHT_GATE": "1"}
VIDEO_CHUNK = 20         # videos per step2_analyse call (as step2_all/videos.sh)
OCR_CHUNK = 150          # photos per step2_stills_ocr call (as step2_all/ocr_loop.py)
JOB_END = ("done", "failed", "quarantined")
SKIP_DIRS = ("#", ".", "@")
MIN_BYTES = 1024         # smaller "photos" are broken files or icons
STEP2_VIDEO_STEPS = ("STEP2_SCENES", "STEP2_KEYFRAMES", "STEP2_DERIVATIVES")


class Stop(Exception):
    """A step cannot go on; the run stops and says why."""


def now():
    return dt.datetime.now(dt.timezone.utc)


def say(msg=""):
    print(msg, flush=True)


def connect():
    from db import Reconnecting
    return Reconnecting()


def rows(handle, sql, params=()):
    conn = handle.ensure()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        out = cur.fetchall() if cur.description else []
    conn.rollback()
    return out


def write(handle, sql, params=()):
    conn = handle.ensure()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        out = cur.fetchall() if cur.description else []
    conn.commit()
    return out


def run_script(log_dir, name, script_args, must_pass=True):
    """Run one step script with the Step 2 engine settings; its output goes to a log file in
    log_dir. Returns the exit code; a non-zero one raises Stop when must_pass."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "{}_{}.log".format(now().strftime("%Y%m%dT%H%M%SZ"), name)
    cmd = ["caffeinate", "-i", str(PYTHON), "-u", str(HERE / script_args[0])] + script_args[1:]
    say("  running {} (log {})".format(name, log.name))
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), stdout=fh,
                             stderr=subprocess.STDOUT, env={**os.environ, **STEP2_ENV})
    say("  {} finished in {:.0f} min, exit {}".format(name, (time.time() - t0) / 60, rc))
    if rc and must_pass:
        tail = [l for l in log.read_text(errors="replace").splitlines()
                if l.strip() and "objc[" not in l][-6:]
        raise Stop("{} stopped with exit {}. Last lines of {}:\n    {}".format(
            name, rc, log, "\n    ".join(tail)))
    return rc


def step2_extras(handle, log_dir):
    """After Step 2, as in the whole-library run: RAW/EDITED label of each new video
    (classify_videos.py), doctors and influencers named in on-screen text or speech
    (link_people_from_text), then the reference-library refresh (ref_refresh: branch,
    people and object links), which Step 3's prompt v3lib reads."""
    say("  Step 2 extras: raw/edited labels, people from text, reference library")
    run_script(log_dir, "classify_videos", ["classify_videos.py"], must_pass=False)
    t0 = time.time()
    try:
        write(handle, "set statement_timeout = '60min'")
        write(handle, "select ml_test.link_people_from_text()")
        out = write(handle, "select * from ml_test.ref_refresh()")
        write(handle, "set statement_timeout = default")
        say("  people from text + reference refresh in {:.0f} s: {}".format(
            time.time() - t0, ", ".join("{} {}".format(r[0], r[1]) for r in out)))
        return None
    except Exception as exc:   # Step 3 still runs; the new files just miss the people hints
        handle.close()
        say("  WARNING people from text / reference refresh failed: {}".format(str(exc)[:200]))
        return "people from text / reference refresh failed: {}".format(str(exc)[:200])


def content_desk(handle, log_dir):
    """After Step 4: refresh the Content Desk search index (ml_test.desk_index) so the new files
    can be found from the page, then make their thumbnails (desk_thumbs.py). Returns a warning
    string, or None."""
    say("  Content Desk: search index + thumbnails")
    t0 = time.time()
    try:
        write(handle, "set statement_timeout = '30min'")
        write(handle, "select ml_test.desk_refresh()")
        write(handle, "set statement_timeout = default")
        say("  search index refreshed in {:.0f} s".format(time.time() - t0))
    except Exception as exc:   # the page keeps yesterday's index; tomorrow's run tries again
        handle.close()
        say("  WARNING search index refresh failed: {}".format(str(exc)[:200]))
        return "Content Desk index refresh failed: {}".format(str(exc)[:200])
    if run_script(log_dir, "desk_thumbs", ["desk_thumbs.py"], must_pass=False):
        return "some Content Desk thumbnails could not be made (see the desk_thumbs log)"
    return None


def step3_left(handle):
    """(main, patient-record) files with a Step 2 picture and no v3lib reading."""
    import step3_v3_run as s3
    return len(s3.pick_files(handle, "main", 0)), len(s3.pick_files(handle, "patients", 0))


def step4_left(handle):
    import step4_v3_embed as s4
    return s4.left(handle)


def run_step3(handle, log_dir, notes):
    """step3_v3_supervise.py, exactly as the whole-library run. Returns the files still left."""
    if subprocess.run(["pgrep", "-f", "step3_v3_(supervise|run).py"], capture_output=True).returncode == 0:
        notes.append("Step 3 skipped: another Step 3 run is going")
        return step3_left(handle)
    before = step3_left(handle)
    say("  {} main and {} patient-record file(s) to analyse".format(*before))
    if sum(before):
        run_script(log_dir, "step3", ["step3_v3_supervise.py"], must_pass=False)
    left = step3_left(handle)
    if sum(left):
        notes.append("Step 3 left {} main and {} patient-record file(s) without a reading "
                     "(see the step3 log)".format(*left))
    return left


def run_step4(handle, log_dir, notes):
    """step4_v3_embed.py, exactly as the whole-library run. Returns the descriptions still left."""
    if subprocess.run(["pgrep", "-f", "step4_v3_embed.py"], capture_output=True).returncode == 0:
        notes.append("Step 4 skipped: another Step 4 run is going")
        return step4_left(handle)
    before = step4_left(handle)
    say("  {} scene and {} file description(s) to embed".format(*before))
    if sum(before):
        run_script(log_dir, "step4", ["step4_v3_embed.py"], must_pass=False)
    left = step4_left(handle)
    if sum(left):
        notes.append("Step 4 left {} scene and {} file description(s) not embedded".format(*left))
    return left


def load_extensions():
    cfg = json.loads((PROJECT_ROOT / "config" / "pipeline_config.json").read_text())
    return {ext: kind for kind, exts in cfg["media_kind_by_extension"].items()
            for ext in exts}


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def manifest_files(batch_dir):
    return sorted(batch_dir.glob("batch*_manifest.json"))


def cmd_plan(args):
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit("{} is not mounted".format(root))
    exts = load_extensions()
    handle = connect()
    have = {r[0] for r in rows(handle, """
        select sf.relative_path from ml_test.source_files sf
        join ml_test.source_folders f on f.id = sf.source_folder_id
        where f.source_kind = 'LOCAL' and f.root_path = %s""", (str(root),))}
    batches = [int(m.group(1)) for (item,) in rows(handle, """
        select metadata->>'step1_item' from ml_test.source_files
        where metadata ? 'step1_item'""")
               for m in [re.match(r"B(\d+)-", item or "")] if m]
    handle.close()
    planned = set()
    for mf in manifest_files(args.out_dir):
        m = json.loads(mf.read_text())
        batches.append(int(m.get("batch") or 0))
        planned |= {e["relative_path"] for e in m.get("files", [])}
    batch = max(batches or [0]) + 1

    found, skipped_have, skipped_planned, skipped_tiny, skipped_new = [], 0, 0, 0, 0
    settle = time.time() - 60 * (args.settle_minutes or 0)
    folders = args.folder or []
    if args.all:   # every top-level folder of the share (the nightly scan)
        folders = sorted(d.name for d in root.iterdir()
                         if d.is_dir() and not d.name.startswith(SKIP_DIRS))
    if not folders:
        raise SystemExit("give --folder X (repeatable) or --all")
    for folder in folders:
        start = (root / folder).resolve()
        if root.resolve() not in (start, *start.parents) or not start.is_dir():
            raise SystemExit("--folder {!r} is not a folder below {}".format(folder, root))
        say("listing {} (read only) ...".format(folder))
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(SKIP_DIRS))
            for name in sorted(filenames):
                kind = exts.get(Path(name).suffix.lower())
                if name.startswith(".") or not kind:
                    continue
                path = Path(dirpath) / name
                rel = path.relative_to(root).as_posix()
                if rel in have:
                    skipped_have += 1
                    continue
                if rel in planned:
                    skipped_planned += 1
                    continue
                st = path.stat()
                size = st.st_size
                if size < MIN_BYTES:
                    skipped_tiny += 1
                elif st.st_mtime > settle:
                    skipped_new += 1    # may still be copying onto the NAS: the next scan takes it
                else:
                    found.append((rel, name, "photo" if kind == "image" else "video", size))
                if args.limit and len(found) >= args.limit:
                    break
            if args.limit and len(found) >= args.limit:
                break   # listing the NAS is slow: stop once the batch is full
        if args.limit and len(found) >= args.limit:
            break
    skipped = ("{} already in ml_test, {} already in a planned manifest, {} under 1 KB (not real "
               "media), {} changed in the last {} min (still copying?)").format(
        skipped_have, skipped_planned, skipped_tiny, skipped_new, args.settle_minutes or 0)
    if not found:
        say("nothing new: " + skipped)
        return []
    if getattr(args, "dry_run", False):
        nv = sum(f[2] == "video" for f in found)
        say("new: {} videos + {} photos = {} files, {:.1f} GB (dry run: no manifest written)".format(
            nv, len(found) - nv, len(found), sum(f[3] for f in found) / 1e9))
        say("skipped: " + skipped)
        return []
    size = args.batch_size or len(found)
    written = []
    for start in range(0, len(found), size):
        written.append(write_manifest(args, root, batch, found[start:start + size], folders))
        batch += 1
    say("skipped: " + skipped)
    return written


def write_manifest(args, root, batch, found, folders):
    files, n = [], {"video": 0, "photo": 0}
    for rel, name, kind, size in found:
        n[kind] += 1
        files.append({
            "item": "B{}-{}{:04d}".format(batch, "V" if kind == "video" else "P", n[kind]),
            "batch": batch, "media_kind": kind, "source_folder": root.name,
            "relative_path": rel, "file_name": name, "expected_bytes": size,
            "origin": "nas", "expected_capture_date": None,
            "provenance": "NAS / " + rel.split("/")[0], "tests": ""})
    manifest = {
        "manifest_version": "batch{}-1.0".format(batch),
        "created": now().date().isoformat(), "batch": batch,
        "target": {"supabase_project_ref": target_mod.profile()["project_ref"],
                   "schema": "ml_test"},
        "source_root_path": str(root), "source_folder_name": root.name,
        "notes": ["Made by {} from {}.".format(
                      VERSION, "the whole share" if args.all else ", ".join(folders)),
                  "File names and paths can contain patient names: this file "
                  "stays on the Mac (not in git, not in chat)."],
        "expected_counts": {"videos": n["video"], "photos": n["photo"],
                            "total": len(files)},
        "files": files}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "batch{}_manifest.json".format(batch)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    os.chmod(out, 0o600)
    gb = sum(f["expected_bytes"] for f in files) / 1e9
    say("batch {}: {} videos + {} photos = {} files, {:.1f} GB".format(
        batch, n["video"], n["photo"], len(files), gb))
    say("manifest: {}".format(out))
    return out


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

class Run:
    def __init__(self, args):
        self.args = args
        self.dry = args.dry_run
        self.manifest_path = args.manifest.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.entries = self.manifest["files"]
        self.items = [e["item"] for e in self.entries]
        self.root = self.manifest["source_root_path"]
        self.log_dir = self.manifest_path.parent / (self.manifest_path.stem + "_logs")
        self.started = now()
        self.notes = []
        self.handle = connect()
        cfg = dict(rows(self.handle, "select key, value from ml_test.pipeline_config"))
        self.cfg = cfg
        self._token = cfg.get("runner_token")
        self.runner_url = cfg.get("runner_url")
        if not self._token or not self.runner_url:
            raise SystemExit("ml_test.pipeline_config has no runner_token / runner_url")

    # ---- helpers --------------------------------------------------------
    def assets(self):
        """item -> (asset_id, sha256, media_kind, source_file_id, file_name)."""
        return {r[0]: r[1:] for r in rows(self.handle, """
            select sf.metadata->>'step1_item', a.id, a.content_hash, a.media_kind,
                   sf.id, sf.file_name
            from ml_test.source_files sf
            join ml_test.asset_sources s on s.source_file_id = sf.id
            join ml_test_compat.assets a on a.id = s.asset_id
            where sf.metadata->>'step1_item' = any(%s)""", (self.items,))}

    def latest_ok(self, asset_ids, steps):
        """asset_id -> set of steps whose latest attempt is COMPLETED/PARTIAL."""
        out = {}
        for aid, step in rows(self.handle, """
            select asset_id, step from (
              select distinct on (asset_id, step) asset_id, step, status
              from ml_test.analysis_runs
              where asset_id = any(%s) and step = any(%s)
              order by asset_id, step, attempt_number desc nulls last, created_at desc) x
            where status in ('COMPLETED', 'PARTIAL')""", (list(asset_ids), list(steps))):
            out.setdefault(aid, set()).add(step)
        return out

    def script(self, name, script_args, must_pass=True):
        return run_script(self.log_dir, name, script_args, must_pass)

    def post(self, url, body, timeout):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "x-runner-token": self._token})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read()[:300].decode(errors="replace")}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return 0, {"error": str(e)[:300]}

    # ---- STEP 1 ---------------------------------------------------------
    def step1(self):
        say("STEP 1  ingest")
        have = {r[0] for r in rows(self.handle, """
            select sf.relative_path from ml_test.source_files sf
            join ml_test.source_folders f on f.id = sf.source_folder_id
            where f.source_kind = 'LOCAL' and f.root_path = %s
              and sf.relative_path = any(%s)""",
            (self.root, [e["relative_path"] for e in self.entries]))}
        todo = len(self.entries) - len(have)
        say("  {} of {} files already ingested, {} to do".format(
            len(have), len(self.entries), todo))
        if todo and not self.dry:
            if not Path(self.root).is_dir():
                raise Stop("{} is not mounted".format(self.root))
            existing = rows(self.handle, "select count(*) from ml_test.media_assets")[0][0]
            a = ["step1_ingest.py", "--commit", "--manifest", str(self.manifest_path),
                 "--expect-existing-assets", str(existing)]
            rc = self.script("step1", a + (["--remaining-only"] if have else []), must_pass=False)
            if rc:
                # a few FAILED files (exact copies of a file already in, broken files, a file
                # moved on the NAS) leave the rest ingested: those go on to Step 2. Nothing
                # ingested at all stops the run.
                got = len(self.assets())
                if not got:
                    self.md5()
                    raise Stop("step1 ingested none of the {} file(s) (exit {}); see {}".format(
                        todo, rc, self.log_dir))
                self.notes.append("Step 1 exit {}: {} of {} file(s) ingested; the rest are "
                                  "duplicates or failed (see the step1 log)".format(
                                      rc, got, len(self.entries)))
        self.md5()

    def md5(self):
        """MD5 of each cached original into source_files.metadata.md5."""
        todo = rows(self.handle, """
            select id, content_sha256, file_name from ml_test.source_files
            where metadata->>'step1_item' = any(%s) and content_sha256 is not null
              and not (metadata ? 'md5')""", (self.items,))
        say("  MD5: {} file(s) to fingerprint".format(len(todo)))
        if self.dry:
            return
        missing = 0
        for sid, sha, name in todo:
            path = target_mod.cached_original_path(sha, name)
            if not path.is_file():
                missing += 1
                continue
            h = hashlib.md5()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(chunk)
            write(self.handle, """update ml_test.source_files
                set metadata = coalesce(metadata, '{}'::jsonb) || jsonb_build_object('md5', %s::text),
                    updated_at = now() where id = %s""", (h.hexdigest(), sid))
        if missing:
            self.notes.append("{} file(s) had no cached original, so no MD5".format(missing))

    # ---- STEP 2 ---------------------------------------------------------
    def step2(self):
        say("STEP 2  extraction")
        assets = self.assets()
        missing = [i for i in self.items if i not in assets]
        if missing:
            self.notes.append("{} manifest file(s) are not ingested (Step 1 failed or "
                              "duplicate): {}".format(len(missing), ", ".join(missing[:20])))
        videos = {i: v for i, v in assets.items() if v[2] == "VIDEO"}
        stills = {i: v for i, v in assets.items() if v[2] == "IMAGE"}
        ok = self.latest_ok([v[0] for v in videos.values()], STEP2_VIDEO_STEPS)
        todo_v = sorted(i for i, v in videos.items()
                        if ok.get(v[0], set()) != set(STEP2_VIDEO_STEPS))
        have_scene = {r[0] for r in rows(self.handle,
            "select distinct asset_sha256 from ml_test.media_scenes where asset_sha256 = any(%s)",
            ([v[1] for v in stills.values()],))}
        todo_s = sorted(i for i, v in stills.items() if v[1] not in have_scene)
        say("  videos: {} of {} to do; photos: {} of {} to do".format(
            len(todo_v), len(videos), len(todo_s), len(stills)))
        cv = self.args.config_version
        # step2_stills only reuses a snapshot (step2_analyse takes it), so a
        # photos-only batch needs it taken here; an identical one is reused and
        # a changed config still stops with step2_analyse's message.
        if (todo_v or todo_s) and not self.dry:
            import step2_analyse as s2
            conn = self.handle.ensure()
            fp, how = s2.take_snapshot(conn, cv, "Step 2 (run_ml_test)")
            conn.commit()
            say("  config snapshot {} {} ({})".format(cv, how, fp[:16]))
        if todo_v and not self.dry:
            # 20 at a time; a chunk that fails is retried once in a second pass, and one bad
            # video never holds up the rest (as step2_all/videos.sh)
            for p in (1, 2):
                if p == 2:
                    ok = self.latest_ok([v[0] for v in videos.values()], STEP2_VIDEO_STEPS)
                    todo_v = sorted(i for i, v in videos.items()
                                    if ok.get(v[0], set()) != set(STEP2_VIDEO_STEPS))
                    if not todo_v:
                        break
                    say("  second pass: {} video(s) not finished".format(len(todo_v)))
                for i in range(0, len(todo_v), VIDEO_CHUNK):
                    self.script("step2_videos_p{}_{}".format(p, i // VIDEO_CHUNK + 1),
                                ["step2_analyse.py", "--commit", "--only",
                                 ",".join(todo_v[i:i + VIDEO_CHUNK]), "--config-version", cv],
                                must_pass=False)
        if todo_s and not self.dry:
            self.script("step2_stills", ["step2_stills.py", "--commit", "--skip-done",
                                         "--config-version", cv], must_pass=False)
        ocr = self.latest_ok([v[0] for v in stills.values()], ("STEP2_OCR",))
        todo_o = sorted(i for i, v in stills.items() if "STEP2_OCR" not in ocr.get(v[0], ()))
        if self.dry:
            todo_o = sorted(set(todo_o) | set(todo_s))
        say("  photo text (OCR): {} to do".format(len(todo_o)))
        if todo_o and not self.dry:
            for i in range(0, len(todo_o), OCR_CHUNK):
                self.script("step2_stills_ocr_{}".format(i // OCR_CHUNK + 1),
                            ["step2_stills_ocr.py", "--commit", "--engine", "apple_vision",
                             "--only"] + todo_o[i:i + OCR_CHUNK], must_pass=False)
        # Files still without their Step 2 rows after the run (the step script reports why in
        # its log); Step 3 simply has nothing to send for them.
        if not self.dry and (todo_v or todo_s):
            ok = self.latest_ok([v[0] for v in videos.values()], STEP2_VIDEO_STEPS)
            bad = sorted(i for i, v in videos.items()
                         if ok.get(v[0], set()) != set(STEP2_VIDEO_STEPS))
            have_scene = {r[0] for r in rows(self.handle,
                "select distinct asset_sha256 from ml_test.media_scenes where asset_sha256 = any(%s)",
                ([v[1] for v in stills.values()],))}
            bad += sorted(i for i, v in stills.items() if v[1] not in have_scene)
            if bad:
                self.notes.append("Step 2 did not finish {} file(s): {}".format(
                    len(bad), ", ".join(bad[:20])))
        if not self.dry:
            warn = step2_extras(self.handle, self.log_dir)
            if warn:
                self.notes.append(warn)

    # ---- STEP 3 ---------------------------------------------------------
    PREPARE_SQL = (
        # one picture per scene: the stored keyframe nearest the middle (003 section 4)
        """update ml_test.media_scenes s set thumb_path = k.storage_path, updated_at = now()
           from (select distinct on (kf.scene_id) kf.scene_id, d.storage_path
                   from ml_test.asset_keyframes kf
                   join ml_test.asset_derivatives d on d.id = kf.derivative_id
                   join ml_test.media_scenes ms on ms.id = kf.scene_id
                  where ms.asset_sha256 = any(%(shas)s)
                  order by kf.scene_id,
                           abs(kf.timestamp_seconds - (coalesce(ms.start_s, 0) + coalesce(ms.end_s, 0)) / 2),
                           kf.technical_quality_score desc nulls last, kf.id) k
          where s.id = k.scene_id and s.thumb_path is distinct from k.storage_path""",
        """update ml_test.media_scenes s set ocr_text = o.txt, updated_at = now()
           from (select scene_id, string_agg(t, ' | ' order by first_t) as txt
                   from (select o.scene_id, o.normalized_text as t, min(o.timestamp_seconds) as first_t
                           from ml_test.ocr_observations o
                           join ml_test.media_scenes ms on ms.id = o.scene_id
                          where o.accepted_for_search and o.normalized_text <> ''
                            and ms.asset_sha256 = any(%(shas)s)
                          group by o.scene_id, o.normalized_text) x
                  group by scene_id) o
          where s.id = o.scene_id and s.ocr_text is distinct from o.txt""",
        """update ml_test.media_assets a set transcript = t.txt, updated_at = now()
           from (select asset_id, string_agg(transcript_text, ' ' order by start_seconds) as txt
                   from ml_test.asset_transcript_chunks
                  where transcription_status = 'COMPLETE' and coalesce(transcript_text, '') <> ''
                    and asset_id = any(%(ids)s)
                  group by asset_id) t
          where a.id = t.asset_id and a.transcript is distinct from t.txt""",
    )

    def job_counts(self, shas):
        return dict(rows(self.handle, """select status, count(*)::int from ml_test.ai_jobs
            where asset_sha256 = any(%s) group by status""", (shas,)))

    def step3(self):
        say("STEP 3  DeepSeek analysis (prompt v3lib, every file still without a reading)")
        if self.dry:
            say("  {} main and {} patient-record file(s) would be analysed".format(*step3_left(self.handle)))
            return
        run_step3(self.handle, self.log_dir, self.notes)

    def step4(self):
        say("STEP 4  embeddings (gte-small, every description still to embed)")
        if self.dry:
            say("  {} scene and {} file description(s) would be embedded".format(*step4_left(self.handle)))
            return
        run_step4(self.handle, self.log_dir, self.notes)

    # ---- report ---------------------------------------------------------
    def report(self, stopped):
        assets = self.assets()
        shas = [v[1] for v in assets.values()]
        cost = rows(self.handle, """select coalesce(sum(cost_usd), 0)::float, count(*)
            from ml_test.analysis_runs where provider = 'deepseek' and created_at >= %s""",
                    (self.started,))[0]
        scenes, searchable, review = rows(self.handle, """
            select count(distinct s.id),
                   count(distinct s.id) filter (where sa.embedding is not null),
                   (select json_object_agg(status, n) from (select status, count(*) n
                      from ml_test.scene_review where asset_sha256 = any(%s) group by 1) x)
            from ml_test.media_scenes s
            left join ml_test.scene_review r on r.scene_id = s.id
            left join ml_test.scene_analysis sa on sa.id = r.final_analysis_id
            where s.asset_sha256 = any(%s)""", (shas, shas))[0]
        jobs = self.job_counts(shas)
        lines = [
            "# {} {}".format(self.manifest_path.stem, "dry run" if self.dry else "run"),
            "",
            "- Controller: {}; started {} UTC, ended {} UTC".format(
                VERSION, self.started.strftime("%Y-%m-%d %H:%M"), now().strftime("%Y-%m-%d %H:%M")),
            "- Result: **{}**".format("stopped: " + stopped if stopped else "finished"),
            "- Files: {} in the manifest, {} ingested".format(len(self.items), len(assets)),
            "- Scenes: {}, searchable {}; review {}".format(scenes, searchable, json.dumps(review or {})),
            "- DeepSeek jobs for this batch: {}".format(json.dumps(jobs)),
            "- DeepSeek cost during this run: ${:.4f} ({} calls)".format(cost[0], cost[1]),
        ] + ["- Note: " + n for n in self.notes]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        out = self.log_dir / "{}_report.md".format(now().strftime("%Y%m%dT%H%M%SZ"))
        out.write_text("\n".join(lines) + "\n")
        say("")
        for l in lines[2:]:
            say(l)
        say("report: {}".format(out))


STEPS = ("step1", "step2", "step3", "step4")


def lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = int(path.read_text().strip() or 0)
        try:
            os.kill(pid, 0)
            raise SystemExit("another run (pid {}) is still going: {}".format(pid, path))
        except ProcessLookupError:
            path.unlink()
            return lock(path)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def cmd_run(args):
    run = Run(args)
    say("{}: {} ({} files){}".format(VERSION, run.manifest_path.name, len(run.items),
                                     ", DRY RUN, nothing is changed" if run.dry else ""))
    lock_path = run.manifest_path.parent / ".run_ml_test.lock"
    if not run.dry:
        lock(lock_path)
    stopped = None
    try:
        for step in STEPS[STEPS.index(args.from_step):STEPS.index(args.to_step) + 1]:
            getattr(run, step)()
    except Stop as e:
        stopped = str(e)
        say("STOPPED: " + stopped)
    except KeyboardInterrupt:
        stopped = "interrupted"
        say("STOPPED: interrupted")
    finally:
        try:
            run.report(stopped)
        finally:
            run.handle.close()
            if not run.dry and lock_path.exists():
                lock_path.unlink()
    return 1 if stopped else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write a manifest of new files below a folder")
    p.add_argument("--folder", action="append", default=[],
                   help="folder below --root (repeatable)")
    p.add_argument("--all", action="store_true", help="every top-level folder of --root")
    p.add_argument("--settle-minutes", type=int, default=0,
                   help="leave out files changed in the last N minutes (may still be copying)")
    p.add_argument("--dry-run", action="store_true", help="count the new files only, write no manifest")
    p.add_argument("--root", default=NAS_ROOT)
    p.add_argument("--limit", type=int, default=None, help="at most this many files")
    p.add_argument("--out-dir", type=Path, default=BATCH_DIR)
    p.add_argument("--batch-size", type=int, default=None,
                   help="split into manifests of at most this many files (one listing)")
    r = sub.add_parser("run", help="take a manifest through Steps 1-4")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--from-step", choices=STEPS, default="step1")
    r.add_argument("--to-step", choices=STEPS, default="step4",
                   help="stop after this step (step1 = ingestion only)")
    r.add_argument("--config-version", default=STEP2_CONFIG_VERSION)
    args = ap.parse_args()
    if args.cmd == "plan":
        return cmd_plan(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
