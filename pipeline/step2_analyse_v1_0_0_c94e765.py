#!/usr/bin/env python3
"""
Phase A / Step 2 - frames, scenes, speech and on-screen text.

Deterministic tools only. ffmpeg extracts frames and finds hard cuts, a colour
histogram measures drift, faster-whisper transcribes speech, easyocr reads
on-screen text. NO AI VISION MODEL IS USED OR IMPORTED ANYWHERE. Nothing here
describes a picture; it measures one.

Per video, in order:
  1. download by google_file_id (Drive stays read-only)
  2. extract frames at t = k * frame_interval_seconds, k = 0, 1, 2, ...
     dropping a sample that would land at or beyond the duration
  3. check the frame count against floor(duration / interval) within
     frame_count_tolerance - outside it the asset FAILS rather than writes
  4. detect hard cuts, build scenes that COVER THE FULL DURATION with no gaps
  5. transcribe the audio
  6. read on-screen text from every extracted frame
  7. upload each frame, read it back, compare SHA-256, and only then write the
     derivative row
  8. delete every local file

Rules honoured:
  1 Drive is read-only.
  2 No AI model that interprets images. faster-whisper and easyocr are local
    deterministic engines named in pipeline_config; neither describes content.
  4 Media is never kept: downloaded and extracted files are deleted.
  6 Every tunable comes from test_v1.pipeline_config, never a constant here.
  8 Anything incomplete is FAILED, never COMPLETED - with one deliberate
    exception, described below.

The watchdog
------------
Every stage runs in a child process the supervisor can kill, and carries its
own ceiling. OCR's ceiling scales with frame count, because OCR samples one
frame per second and a 189-second video is 190 frames of work against a
25-frame clip's 25; a single fixed ceiling could never fit both, and did not.

When OCR exceeds its ceiling the asset is NOT dropped. The frames, scenes and
transcript are already checkpointed, so OCR falls back to the representative
frames - one per scene - and the asset completes with

    ocr_coverage = SAMPLED

which is the exception to Rule 8, and a considered one: an asset whose text
was read at four scene boundaries is genuinely catalogued, just not
exhaustively, and recording that as FAILED would hide a usable asset instead
of describing it. What Rule 8 actually forbids is silence, so coverage is
recorded on EVERY OCR run - FULL, SAMPLED or NONE - and surfaced by the view
test_v1.asset_ocr_coverage. Before that field existed, a video with no text
in it and a video that was never read both held zero ocr_observations and
were indistinguishable.

Usage:
    python scripts/step2_analyse.py --dry-run   # computes everything, writes nothing
    python scripts/step2_analyse.py --commit    # writes to test_v1 and Storage
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORT_DRY = PROJECT_ROOT / "reports" / "step2_dryrun.md"
REPORT_COMMIT = PROJECT_ROOT / "reports" / "step2_commit.md"

PROCESSOR_VERSION = "step2_analyse/1.0.0"

# A NEW config version. Step 1 rows keep step1_2026_09_22_a, so the two steps
# stay distinguishable in every stamped table.
STEP2_CONFIG_VERSION = "step2_2026_09_22_d"
STEP2_SNAPSHOT_NOTE = (
    "Step 2, cuts_and_drift, scene guard corrected, plus per-stage watchdog "
    "budgets (ocr_stage_base_seconds, ocr_stage_seconds_per_frame, "
    "stage_ceiling_seconds, ocr_sampled_max_frames). Step 1 rows keep "
    "step1_2026_09_22_a."
)

# Versions whose Step 2 output is still current.
#
# _d differs from _c by four keys and nothing else, and all four are watchdog
# policy: how long a stage is allowed to take before it is killed, and how
# many frames the degraded retry reads. None of them changes what is
# measured, what is extracted or what is stored - a frame cut under _c is the
# same frame cut under _d. So the rows written under _c are not stale and
# must not be recomputed; twenty-four assets would be reprocessed to arrive
# at byte-identical results, and the unique constraint on
# (asset_id, timestamp_seconds) would refuse them anyway.
#
# Anything that DID change a measurement would need a version that is not in
# this list, which is the point of keeping the list explicit rather than
# matching on a prefix.
STEP2_CONFIG_LINEAGE = ("step2_2026_09_22_c", "step2_2026_09_22_d")

# analysis_runs.status has no PARTIAL. Anything incomplete is FAILED, with the
# error_code saying which kind of incomplete it was.
STATUS_OK = "COMPLETED"
STATUS_BAD = "FAILED"


class AssetFailed(Exception):
    """One asset could not be processed. Nothing is written for it."""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_db_config(conn):
    with conn.cursor() as cur:
        cur.execute("select key, value from test_v1.pipeline_config")
        return {k: v for k, v in cur.fetchall()}


def need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: {}".format(key))
    return cfg[key]


# The same fingerprint snapshot_config() computes, but read-only: sha256 over
# the pretty-printed {key: {value, value_type}} map of pipeline_config.
CURRENT_FINGERPRINT_SQL = """
select encode(extensions.digest(convert_to(jsonb_pretty(
         (select jsonb_object_agg(key,
                   jsonb_build_object('value', value, 'value_type', value_type))
            from test_v1.pipeline_config)), 'UTF8'), 'sha256'), 'hex')
"""


def take_snapshot(conn, version, note):
    """Take the snapshot, or reuse an identical one already taken.

    Re-running a step must not fail merely because its snapshot exists, but
    reusing one blindly would let the config drift out from under a version
    name. So the current config is fingerprinted first and compared: equal
    means reuse, different means the config changed after that snapshot was
    taken and a human has to choose a new version. Returns (fingerprint,
    'created'|'reused')."""
    with conn.cursor() as cur:
        cur.execute(CURRENT_FINGERPRINT_SQL)
        current = cur.fetchone()[0]
        cur.execute("select fingerprint from test_v1.config_snapshots "
                    "where config_version = %s", (version,))
        existing = cur.fetchone()
        if existing:
            if existing[0] != current:
                raise SystemExit(
                    "config snapshot {!r} already exists but no longer matches "
                    "pipeline_config.\n"
                    "  snapshot fingerprint: {}\n"
                    "  config fingerprint:   {}\n"
                    "  The config changed after that snapshot was taken. Pick "
                    "a new version name rather than\n  stamping rows with a "
                    "version whose payload is not what ran."
                    .format(version, existing[0], current))
            return existing[0], "reused"
        cur.execute("select test_v1.snapshot_config(%s, %s)", (version, note))
        return cur.fetchone()[0], "created"


# --------------------------------------------------------------------------
# what to process
# --------------------------------------------------------------------------

ASSET_SQL = """
select a.id                         as asset_id,
       a.content_hash,
       sf.metadata->>'step1_item'   as item,
       a.file_name,
       a.media_kind,
       sf.google_file_id,
       sf.relative_path,
       sf.parent_visible,
       f.source_name                as folder_name,
       f.discovery_status,
       m.duration_seconds,
       m.display_width_px, m.display_height_px,
       m.fps, m.has_audio
from test_v1.assets a
join test_v1.asset_technical_metadata m on m.asset_id = a.id
join test_v1.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
join test_v1.source_files sf on sf.id = s.source_file_id
join test_v1.source_folders f on f.id = sf.source_folder_id
order by a.media_kind desc,
         (regexp_replace(sf.metadata->>'step1_item', '\\D', '', 'g'))::int
"""


def load_assets(conn):
    with conn.cursor() as cur:
        cur.execute(ASSET_SQL)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def item_sort(item):
    digits = re.sub(r"\D", "", item or "")
    return (item[:1] if item else "", int(digits) if digits else 0)


def current_step2_counts(conn):
    """Rows already present in each Step 2 table."""
    from preflight import STEP2_TABLES
    out = {}
    with conn.cursor() as cur:
        for t in STEP2_TABLES:
            cur.execute("select count(*) from test_v1." + t)
            out[t] = cur.fetchone()[0]
    conn.rollback()
    return out


# One attempt per (asset, step): the highest attempt_number, and among equals
# the one written last. Earlier attempts stay in the table as the record of
# what was tried and why it failed, but they are history, not status.
LATEST_ATTEMPT_SQL = """
    select distinct on (r.asset_id, r.step)
           r.asset_id, r.step, r.status, r.error_code, r.error_message,
           r.attempt_number
    from test_v1.analysis_runs r
    where r.step like 'STEP2%%' and r.config_version = any(%s)
    order by r.asset_id, r.step, r.attempt_number desc, r.created_at desc
"""


def completed_assets(conn, config_version=None):
    """Assets that already carry a full set of Step 2 runs under the current
    config LINEAGE, every one COMPLETED on its LATEST attempt. A resumed run
    skips these instead of repeating twenty minutes of work per file.

    The lineage, not a single version, because adding the watchdog budget
    keys forced a new config version onto rows written after the change while
    the rows written before it remain perfectly current. Asking only about
    the newest version would declare all twenty-four assets unprocessed and
    send the run at work it must not repeat.

    An asset whose latest attempt at any step FAILED is NOT complete and is
    retried: a dropped connection or a scene-rate rejection must not be frozen
    in as finished. A superseded FAILED attempt, on the other hand, must not
    keep an asset out of "done" for ever - that is what the first run's
    watchdog rows would otherwise do.
    """
    # What a finished asset looks like depends on what it IS. A still has no
    # frames, no scenes and no soundtrack, so it only ever produces an OCR
    # run: demanding all five of a photograph made every image permanently
    # incomplete, and every run re-read all nine of them. Harmless while they
    # held no text - and a duplicate-row bug the first time one did.
    want = {"VIDEO": {"STEP2_SCENES", "STEP2_KEYFRAMES", "STEP2_DERIVATIVES",
                      "STEP2_TRANSCRIPT", "STEP2_OCR"},
            "IMAGE": {"STEP2_OCR"}}
    with conn.cursor() as cur:
        cur.execute("""
            select l.asset_id,
                   array_agg(distinct l.step) as steps,
                   bool_and(l.status = 'COMPLETED') as all_ok,
                   min(sf.metadata->>'step1_item') as item,
                   min(m.media_kind) as media_kind
            from ({}) l
            join test_v1.asset_sources s
              on s.asset_id = l.asset_id and s.is_first_discovered_source
            join test_v1.source_files sf on sf.id = s.source_file_id
            join test_v1.asset_technical_metadata m on m.asset_id = l.asset_id
            group by l.asset_id
        """.format(LATEST_ATTEMPT_SQL), (list(STEP2_CONFIG_LINEAGE),))
        rows = cur.fetchall()
    conn.rollback()
    ids, items = set(), set()
    for asset_id, steps, all_ok, item, media_kind in rows:
        expected = want.get(media_kind)
        if expected and all_ok and set(steps) == expected:
            ids.add(asset_id)
            items.add(item)
    return items, ids


# --------------------------------------------------------------------------
# 7. patient name and session date from the folder path
# --------------------------------------------------------------------------

# Paths in this library look like 'LIM TJUN CHERN/2025.03.11/DSC03426.JPG'.
# Deterministic parse: a date-like path segment gives the session date, and
# the segment before it gives the patient name. No inference, no guessing -
# a path with no date segment yields nothing and says so.
DATE_SEGMENT_RE = re.compile(r"^(\d{4})[.\-_/](\d{1,2})[.\-_/](\d{1,2})$")


def parse_relative_path(relative_path, parent_visible):
    """Returns a dict. Absence is recorded as a reason, never as a blank."""
    out = {"relative_path": relative_path,
           "patient_name": None, "session_date": None,
           "parse_status": None, "parse_note": None,
           "path_from_drive_scan": bool(parent_visible)}

    parts = [p for p in (relative_path or "").replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        out["parse_status"] = "NO_PATH_STRUCTURE"
        out["parse_note"] = ("the file sits at the root of its folder, so the "
                             "path carries no patient or session")
        return out

    date_idx = None
    for i, seg in enumerate(parts[:-1]):
        m = DATE_SEGMENT_RE.match(seg.strip())
        if m:
            y, mo, d = (int(x) for x in m.groups())
            try:
                out["session_date"] = dt.date(y, mo, d).isoformat()
                date_idx = i
            except ValueError:
                out["parse_status"] = "BAD_DATE_SEGMENT"
                out["parse_note"] = "segment {!r} looks like a date but is not one".format(seg)
                return out
            break

    if date_idx is None:
        out["parse_status"] = "NO_DATE_SEGMENT"
        out["parse_note"] = "no path segment matches a date"
        return out

    if date_idx == 0:
        out["parse_status"] = "DATE_WITHOUT_PATIENT"
        out["parse_note"] = "the date is the first segment, so no patient precedes it"
        return out

    out["patient_name"] = parts[date_idx - 1].strip()
    out["parse_status"] = "OK"
    if not parent_visible:
        out["parse_note"] = (
            "PARTIAL: parent_visible is false for this file, so relative_path "
            "came from the manifest, not from a Drive scan. The parse is "
            "correct but the end-to-end case is not exercised until "
            "ALL PATIENT REVIEW is shared.")
    return out


# --------------------------------------------------------------------------
# 1 + 2. frames
# --------------------------------------------------------------------------

def planned_frame_times(duration, interval, origin, max_frames):
    """t = k * interval for k = 0,1,2,..., dropping any sample at or beyond
    the duration. This is what frame_sample_origin=interval_start means."""
    if origin != "interval_start":
        raise SystemExit(
            "frame_sample_origin={!r} is not implemented. Only "
            "'interval_start' is.".format(origin))
    times, k = [], 0
    while True:
        t = round(k * interval, 3)
        if t >= duration:
            break
        times.append(t)
        k += 1
        if len(times) > max_frames:
            raise AssetFailed(
                "frame plan exceeded max_frames_per_asset={}".format(max_frames))
    return times


def target_size(disp_w, disp_h, box_long, box_short):
    """Fit inside a box whose long edge is box_long and short edge box_short,
    orientation-agnostic. A 720x1280 portrait inside 1280x720 is unchanged."""
    long_edge, short_edge = max(disp_w, disp_h), min(disp_w, disp_h)
    factor = min(box_long / long_edge, box_short / short_edge, 1.0)
    w = max(2, int(round(disp_w * factor / 2)) * 2)
    h = max(2, int(round(disp_h * factor / 2)) * 2)
    return w, h


def extract_frames(video_path, out_dir, times, size, quality, interval,
                   deadline=None):
    """One seek per planned time, so the frame really is the one at t.

    The obvious single-pass form, -vf fps=1/interval, does NOT implement this
    config: its output count is round(duration/interval), not floor+1. On a
    189.267s video it yields 189 frames where t = k*1 for k = 0..189 requires
    190, and on a 12.003s video it yields 12 where the rule requires 13. So
    each frame is seeked to individually. ffmpeg applies the display matrix
    itself, so output is already in display orientation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files, kept_times, notes = [], [], []
    for i, t in enumerate(times):
        dest = out_dir / "f_{:06d}.jpg".format(i)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", "{:.3f}".format(t), "-i", str(video_path),
               "-frames:v", "1",
               # JPEG is a full-range format. V9 carries full-range YUV and
               # ffmpeg 9 refuses the non-standard combination unless the
               # output range is stated, so it is stated for every video
               # rather than special-cased - the frames stay comparable.
               "-vf", "scale={}:{}:out_range=pc".format(size[0], size[1]),
               "-pix_fmt", "yuvj420p",
               "-q:v", str(jpeg_qscale(quality)), str(dest)]
        if deadline:
            deadline.check("extract_frames")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if deadline:
            deadline.heartbeat("frames", i + 1, len(times))
        if proc.returncode != 0 or not dest.exists():
            if i == len(times) - 1:
                # The last planned sample can fall inside the declared
                # duration and still have no frame behind it: V9 is 12.003s
                # and its final coded frame ends before t=12.000. Dropping
                # the sample is the honest result, and it is recorded rather
                # than silently absorbed. A gap anywhere else is a failure.
                notes.append(
                    "final sample t={:.3f}s dropped: no coded frame at or "
                    "after it, although the declared duration extends "
                    "past it".format(t))
                break
            raise AssetFailed(
                "frame extraction failed at t={:.3f}: {}".format(
                    t, (proc.stderr or "").strip()[:200] or "no frame produced"))
        files.append(dest)
        kept_times.append(t)
    return files, kept_times, notes


def jpeg_qscale(quality):
    """ffmpeg -q:v is 2 (best) to 31 (worst); config gives 0-100 quality."""
    q = max(1, min(100, int(quality)))
    return max(2, min(31, int(round(31 - (q / 100.0) * 29))))


# --------------------------------------------------------------------------
# 3 + 4. scenes
# --------------------------------------------------------------------------

SHOWINFO_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


def detect_cuts(video_path, threshold):
    """Hard cuts via ffmpeg's scene metric. Deterministic, no model."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
           "-filter:v", "select='gt(scene,{})',showinfo".format(threshold),
           "-an", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    # showinfo writes to stderr; a non-zero rc here is a real failure.
    if proc.returncode != 0:
        raise AssetFailed("ffmpeg cut detection failed: {}".format(
            (proc.stderr or "").strip()[:300]))
    cuts = []
    for line in (proc.stderr or "").splitlines():
        if "showinfo" not in line:
            continue
        m = SHOWINFO_PTS_RE.search(line)
        if m:
            cuts.append(round(float(m.group(1)), 3))
    return sorted(set(cuts))


def scene_score_profile(video_path, probe_threshold=0.02, top=12):
    """Diagnostic only. Reports the largest frame-to-frame scene scores the
    file contains, so 'cuts found nothing' can be shown as evidence rather
    than asserted. Never used to place a boundary - the run uses
    scene_change_threshold and nothing else."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
           "-filter:v",
           "select='gt(scene,{})',metadata=print:file=-".format(probe_threshold),
           "-an", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception:
        return []
    scores, pending_t = [], None
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            pending_t = float(m.group(1))
        m2 = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m2 and pending_t is not None:
            scores.append((round(pending_t, 3), round(float(m2.group(1)), 4)))
            pending_t = None
    scores.sort(key=lambda x: -x[1])
    return scores[:top]


def frame_histogram(path, downscale_px, bins):
    """Per-channel colour histogram of one frame, L1-normalised.

    Deterministic and cheap: reduce to a fixed square, bin each RGB channel,
    normalise. Pinned to config values (scene_drift_downscale_px,
    scene_drift_histogram_bins) so the feature does not depend on a library
    default and a re-run reproduces the same boundaries."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((downscale_px, downscale_px),
                                      Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    out = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
        total = h.sum()
        out.append(h / total if total else h.astype(float))
    return np.concatenate(out)


def histogram_tvd(a, b):
    """Total-variation distance, averaged over the three channels, in [0, 1].

    Chosen in config over a greyscale layout measure because it is
    translation-invariant: panning across a static room must not read as the
    situation changing."""
    import numpy as np
    return float(np.abs(a - b).sum() / (2.0 * 3.0))


def detect_drift(frames, cuts, cfg, duration):
    """Anchored drift boundaries within a shot.

    A segment keeps an anchor frame. Every later frame is compared to that
    anchor, not to its neighbour, so a slow change accumulates instead of
    hiding under a per-frame noise floor. The distance must stay above
    scene_drift_threshold for scene_drift_sustain_seconds before a boundary
    is recorded, which is what stops a hand crossing the lens becoming a
    scene. The boundary is placed at the FIRST crossing, because that is when
    the change began. A hard cut forces a boundary and re-anchors.

    Returns (boundary_times, diagnostic_rows).
    """
    if not frames:
        return [], []
    thr = float(need(cfg, "scene_drift_threshold"))
    sustain = float(need(cfg, "scene_drift_sustain_seconds"))
    px = int(need(cfg, "scene_drift_downscale_px"))
    bins = int(need(cfg, "scene_drift_histogram_bins"))
    metric = need(cfg, "scene_drift_metric")
    if metric != "histogram_tvd":
        raise SystemExit(
            "scene_drift_metric={!r} is not implemented. Only "
            "'histogram_tvd' is.".format(metric))

    hists = [frame_histogram(Path(f["path"]), px, bins) for f in frames]
    times = [f["timestamp"] for f in frames]
    cutset = sorted(c for c in cuts if 0.0 < c < duration)

    boundaries, diag = [], []
    anchor = 0
    crossing_since = None
    for i in range(1, len(frames)):
        t = times[i]

        # A hard cut between the previous frame and this one wins outright.
        if any(times[i - 1] < c <= t for c in cutset):
            boundaries.append(round(t, 3))
            anchor, crossing_since = i, None
            diag.append((round(t, 3), None, "cut_reanchor"))
            continue

        d = histogram_tvd(hists[i], hists[anchor])
        diag.append((round(t, 3), round(d, 4), "above" if d > thr else "below"))

        if d > thr:
            if crossing_since is None:
                crossing_since = t
            elif t - crossing_since >= sustain:
                # Boundary at the first crossing, not at confirmation.
                boundaries.append(round(crossing_since, 3))
                anchor = next(j for j, tt in enumerate(times)
                              if tt >= crossing_since)
                crossing_since = None
        else:
            crossing_since = None

    return sorted(set(boundaries)), diag


def build_scenes(duration, boundaries, cfg, mode):
    """Scenes that cover [0, duration] exactly: no gaps, no overlaps.

    boundaries is a list of (time, source) where source is 'cut' or 'drift'.
    The schema forbids overlap but says nothing about gaps, so contiguity is
    enforced here and verified again in SQL afterwards."""
    duration = round(float(duration), 3)
    min_dur = float(need(cfg, "scene_min_duration_seconds"))
    cut_min = float(need(cfg, "scene_cut_detection_min_duration_seconds"))

    if duration < cut_min:
        return [{"index": 0, "start": 0.0, "end": duration,
                 "boundary_source": "start",
                 "detection_method": "{}:too_short_for_cut_detection".format(mode),
                 "confidence": None}]

    # Keep only interior boundaries; 0 and duration are the outer edges. A cut
    # and a drift at the same instant collapse to one, and the cut wins.
    interior = {}
    for t, src in boundaries:
        t = round(float(t), 3)
        if not (0.0 < t < duration):
            continue
        if interior.get(t) != "cut":
            interior[t] = src

    starts = [(0.0, "start")] + sorted(interior.items())
    scenes = []
    for i, (s, src) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else duration
        scenes.append({"start": round(s, 3), "end": round(e, 3),
                       "boundary_source": src})

    # Merge anything shorter than scene_min_duration_seconds into its
    # predecessor, so a flicker does not become a scene.
    merged = []
    for sc in scenes:
        if merged and (sc["end"] - sc["start"]) < min_dur:
            merged[-1]["end"] = sc["end"]
        else:
            merged.append(dict(sc))
    # A too-short FIRST scene absorbs the one after it instead.
    while len(merged) > 1 and (merged[0]["end"] - merged[0]["start"]) < min_dur:
        merged[1]["start"] = merged[0]["start"]
        merged[1]["boundary_source"] = "start"
        merged.pop(0)

    for i, sc in enumerate(merged):
        sc["index"] = i
        if mode == "cuts_and_drift":
            sc["detection_method"] = (
                "cuts_and_drift:ffmpeg_scene_gt_{}+{}_gt_{}_sustain_{}s".format(
                    need(cfg, "scene_change_threshold"),
                    need(cfg, "scene_drift_metric"),
                    need(cfg, "scene_drift_threshold"),
                    need(cfg, "scene_drift_sustain_seconds")))
        else:
            sc["detection_method"] = "cuts:ffmpeg_scene_gt_{}".format(
                need(cfg, "scene_change_threshold"))
        sc["confidence"] = None
    return merged


def verify_scene_coverage(scenes, duration):
    """Returns (covered_seconds, problems). Coverage is reported as a number,
    never as a yes."""
    duration = round(float(duration), 3)
    problems = []
    if not scenes:
        return 0.0, ["no scenes"]
    if abs(scenes[0]["start"]) > 1e-9:
        problems.append("scene 0 starts at {} not 0.000".format(scenes[0]["start"]))
    if abs(scenes[-1]["end"] - duration) > 0.0005:
        problems.append("last scene ends at {} not {}".format(
            scenes[-1]["end"], duration))
    covered = 0.0
    for i, sc in enumerate(scenes):
        if sc["end"] <= sc["start"]:
            problems.append("scene {} is not positive length".format(i))
        covered += sc["end"] - sc["start"]
        if i + 1 < len(scenes):
            gap = scenes[i + 1]["start"] - sc["end"]
            if abs(gap) > 0.0005:
                problems.append(
                    "gap of {:.3f}s between scene {} end {} and scene {} start {}"
                    .format(gap, i, sc["end"], i + 1, scenes[i + 1]["start"]))
    return round(covered, 3), problems


# A rate needs enough footage for the rate to mean anything. Below this, the
# bound on over-segmentation is scene_min_duration_seconds instead.
SCENE_RATE_MIN_DURATION_SECONDS = 60.0


def check_scene_rate(scenes, duration, cfg):
    """Guard over-segmentation - the regression case was 42 scenes in 18.467s.

    The per-minute ceiling applies only above 60 seconds. Below that the
    arithmetic rejects correct detections: allowance is duration/5 seconds, so
    a 9.567s clip is capped at one scene and a 23.667s clip at four. V8 found
    2 and V6 found 5, both right, both rejected. Under 60s the bound is
    scene_min_duration_seconds, which the merge step has already enforced, so
    what reaches here cannot contain a scene shorter than that. The rate is
    computed and reported either way."""
    per_min = round(len(scenes) / (duration / 60.0), 2) if duration else 0.0
    ceiling = float(need(cfg, "scene_max_count_per_minute"))
    min_dur = float(need(cfg, "scene_min_duration_seconds"))

    if duration <= SCENE_RATE_MIN_DURATION_SECONDS:
        shortest = min((s["end"] - s["start"] for s in scenes), default=0.0)
        if len(scenes) > 1 and shortest < min_dur - 1e-9:
            raise AssetFailed(
                "shortest scene {:.3f}s is under scene_min_duration_seconds="
                "{} ({} scenes in {:.3f}s)".format(shortest, min_dur,
                                                   len(scenes), duration))
        return per_min

    if len(scenes) > 1 and per_min > ceiling:
        raise AssetFailed(
            "scene rate {:.1f}/min exceeds scene_max_count_per_minute={} "
            "({} scenes in {:.3f}s)".format(per_min, ceiling, len(scenes),
                                            duration))
    return per_min


# --------------------------------------------------------------------------
# 5. speech
# --------------------------------------------------------------------------

_WHISPER = {}


def whisper_model(cfg):
    if "m" not in _WHISPER:
        from faster_whisper import WhisperModel
        _WHISPER["m"] = WhisperModel(
            need(cfg, "transcription_model"),
            device="cpu",
            compute_type=need(cfg, "whisper_compute_type"))
    return _WHISPER["m"]


def extract_audio(video_path, wav_path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
           "-f", "wav", str(wav_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssetFailed("audio extraction failed: {}".format(
            (proc.stderr or "").strip()[:300]))


def transcribe(video_path, tmp_dir, cfg, duration):
    """Returns a dict with status and a list of chunks. Music with no speech
    is EMPTY_NO_SPEECH with empty text, which is a recorded reason and not a
    missing row."""
    wav = tmp_dir / "audio.wav"
    try:
        extract_audio(video_path, wav)
        model = whisper_model(cfg)
        lang = need(cfg, "transcription_language")
        segments, info = model.transcribe(
            str(wav),
            beam_size=int(need(cfg, "whisper_beam_size")),
            vad_filter=bool(need(cfg, "whisper_vad_filter")),
            language=lang if lang else None)
        chunks = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            chunks.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "language": info.language,
                # avg_logprob is a log probability; exp() puts it on 0-1 so it
                # satisfies the column's range check. Recorded as a confidence
                # proxy, not as a calibrated probability.
                "confidence": round(min(1.0, max(0.0,
                                   math.exp(getattr(seg, "avg_logprob", -1.0)))), 4),
            })
        if not chunks:
            return {"status": "EMPTY_NO_SPEECH",
                    "language": info.language,
                    "language_probability": round(float(
                        getattr(info, "language_probability", 0.0) or 0.0), 4),
                    "chunks": [{"start": 0.0, "end": round(duration, 3),
                                "text": "", "language": info.language,
                                "confidence": None}],
                    "reason": "voice activity filter found no speech"}
        return {"status": "COMPLETE", "language": info.language,
                "language_probability": round(float(
                    getattr(info, "language_probability", 0.0) or 0.0), 4),
                "chunks": chunks, "reason": None}
    except AssetFailed:
        raise
    except Exception as exc:
        return {"status": "FAILED", "language": None,
                "language_probability": None,
                "chunks": [{"start": 0.0, "end": round(duration, 3),
                            "text": "", "language": None, "confidence": None}],
                "reason": "{}:{}".format(type(exc).__name__, str(exc)[:200])}
    finally:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------
# 6. on-screen text
# --------------------------------------------------------------------------

_OCR = {}


def ocr_reader(cfg):
    if "r" not in _OCR:
        import torch
        torch.set_num_threads(int(need(cfg, "ocr_torch_threads")))
        import easyocr
        langs = list(need(cfg, "ocr_languages"))
        _OCR["r"] = easyocr.Reader(langs, gpu=False, verbose=False)
    return _OCR["r"]


def classify_text(bbox, frame_w, frame_h):
    """Deterministic geometry only.

    A burnt-in caption sits low in the frame and runs wide; that is a shape,
    and shapes are measurable. Whether a label belongs to a device is NOT
    measurable without interpreting the picture, so nothing here claims
    DEVICE_LABEL - that would need the vision model this step is forbidden."""
    try:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
    except Exception:
        return "OTHER"
    top, bottom = min(ys), max(ys)
    width = max(xs) - min(xs)
    centre_y = (top + bottom) / 2.0
    if frame_h and centre_y > 0.70 * frame_h and frame_w and width > 0.40 * frame_w:
        return "CAPTION"
    return "OTHER"


def ocr_frame(path, reader, cfg, frame_w, frame_h):
    floor = float(need(cfg, "ocr_confidence_floor"))
    try:
        results = reader.readtext(str(path))
    except Exception as exc:
        raise AssetFailed("ocr failed on {}: {}:{}".format(
            path.name, type(exc).__name__, str(exc)[:160]))
    out = []
    for bbox, text, conf in results:
        raw = (text or "").strip()
        if not raw:
            continue
        conf = float(conf)
        accepted = conf >= floor
        out.append({
            "raw_text": raw,
            "normalized_text": " ".join(raw.upper().split()),
            "text_type": classify_text(bbox, frame_w, frame_h),
            "confidence": round(min(1.0, max(0.0, conf)), 4),
            "accepted_for_search": accepted,
            "rejection_reason": None if accepted else
                "confidence {:.4f} below ocr_confidence_floor {}".format(conf, floor),
            "bounding_box": {"points": [[float(p[0]), float(p[1])] for p in bbox]},
        })
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def near_duplicate_rate(per_frame_text):
    """Fraction of consecutive frame pairs whose accepted on-screen text is
    the same, plus the near-identical fraction at Jaccard >= 0.9. Both are
    measurements for sizing a full-library run, not a reason to sample less
    here - this run reads every frame."""
    pairs = len(per_frame_text) - 1
    if pairs <= 0:
        return {"pairs": 0, "identical": 0, "identical_pct": None,
                "near_identical": 0, "near_identical_pct": None,
                "frames_with_text": sum(1 for s in per_frame_text if s)}
    identical = near = 0
    for a, b in zip(per_frame_text, per_frame_text[1:]):
        if a == b:
            identical += 1
            near += 1
            continue
        union = a | b
        if union and len(a & b) / len(union) >= 0.9:
            near += 1
    return {"pairs": pairs, "identical": identical,
            "identical_pct": round(100.0 * identical / pairs, 1),
            "near_identical": near,
            "near_identical_pct": round(100.0 * near / pairs, 1),
            "frames_with_text": sum(1 for s in per_frame_text if s)}


def scene_for_time(scenes, t):
    for sc in scenes:
        if sc["start"] <= t < sc["end"]:
            return sc
    return scenes[-1]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rmtree(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


# --------------------------------------------------------------------------
# stage budgets
# --------------------------------------------------------------------------
#
# The ceiling used to be one number for a whole asset, derived from duration.
# That was wrong in a way arithmetic could have caught before the run did:
# OCR samples one frame per second, so its cost scales with FRAME COUNT, and
# a 189-second video carries 190 frames against a 25-frame clip's 25. One
# ceiling cannot fit both. IMG_1147 could not have finished inside the
# 340-359s that the short clips were given, at any per-frame cost.
#
# So OCR is budgeted per frame and every other stage gets a fixed ceiling,
# and all of it comes from pipeline_config rather than from constants here
# (Rule 6), so the policy is a row to change and not a release.
#
# Measured on this machine, 720x1280 clinic footage, easyocr on CPU:
#
#     V12    4 frames    22s     5.5 s/frame
#     V1     6 frames    50s     8.3 s/frame
#     V2    25 frames   166s     6.6 s/frame
#     V15   25 frames   176s     7.0 s/frame
#     V4    81 frames   541s     6.7 s/frame
#     V5    84 frames   559s     6.7 s/frame
#     V7   190 frames  1019s     5.4 s/frame
#
# The budget has to sit above the WORST of those, not the average, or the
# fix turns every slow-but-healthy asset into a degraded one.

def ocr_budget(frame_count, cfg):
    """The OCR stage's ceiling: base + per_frame * frames.

    frame_count is the number of frames OCR will actually be asked to read,
    so the sampled retry gets a budget sized to the sample and not to the
    video it came from."""
    return (float(need(cfg, "ocr_stage_base_seconds"))
            + float(need(cfg, "ocr_stage_seconds_per_frame")) * frame_count)


def fixed_budget(cfg):
    """Every stage that is not OCR. Download, frame extraction, cut
    detection and transcription do not scale with frame count the way OCR
    does, and none of them has ever been the slow one - they were only ever
    the stage a hang happened to be caught in."""
    return float(need(cfg, "stage_ceiling_seconds"))


def planned_frame_count(asset):
    """What OCR will be asked to read, known before a frame is cut: the
    sampler plans floor(duration) + 1 frames at one per second."""
    d = float(asset.get("duration_seconds") or 0.0)
    return math.floor(d) + 1 if d else 1


def asset_deadline(asset, cfg):
    """A backstop for the whole asset, used only when the worker dies before
    it has told the parent which stage it is in. The per-stage budgets are
    the real control; this stops an asset that hangs before it says anything
    from hanging for ever."""
    return fixed_budget(cfg) * 3 + ocr_budget(planned_frame_count(asset), cfg)


def representative_frames(frames, scenes, cap):
    """The frames worth reading when there is not time to read them all:
    the first frame of every scene, plus frame 0, capped.

    One per scene rather than a flat sample, because a scene boundary is
    where the picture changed, which is where new on-screen text appears.
    Frame 0 is always in, because it is the frame the keyframe rows mark
    is_representative and the one a catalogue thumbnail shows."""
    by_scene, seen = [], set()
    for sc in scenes or []:
        for fr in frames:
            if sc["start"] <= fr["timestamp"] < sc["end"]:
                if fr["index"] not in seen:
                    seen.add(fr["index"])
                    by_scene.append(fr)
                break
    if frames and frames[0]["index"] not in seen:
        seen.add(frames[0]["index"])
        by_scene.insert(0, frames[0])
    by_scene.sort(key=lambda f: f["index"])
    cap = max(1, int(cap))
    if len(by_scene) <= cap:
        return by_scene
    # Too many scenes to read even one frame each: keep frame 0 and spread
    # the rest evenly over the scenes rather than taking the first few.
    step = len(by_scene) / float(cap)
    picked = [by_scene[min(len(by_scene) - 1, int(i * step))] for i in range(cap)]
    out, seen = [], set()
    for fr in picked:
        if fr["index"] not in seen:
            seen.add(fr["index"])
            out.append(fr)
    return out


# Coverage vocabulary. Written into analysis_runs.metadata for every OCR run
# and surfaced by the test_v1.asset_ocr_coverage view, so "no text" and
# "never looked" can never again be the same answer.
OCR_FULL = "FULL"          # every extracted frame was read
OCR_SAMPLED = "SAMPLED"    # the budget ran out; representative frames read
OCR_NONE = "NONE"          # the stage ran and could not read anything


class StageOverBudget(Exception):
    """One stage exceeded its ceiling. Caught where degrading is possible."""

    def __init__(self, stage, elapsed, limit):
        self.stage, self.elapsed, self.limit = stage, elapsed, limit
        super().__init__(
            "stage {!r} exceeded its {:.0f}s ceiling ({:.0f}s elapsed)"
            .format(stage, limit, elapsed))


class NullClock:
    """Used when nothing is enforcing budgets - the dry-run path and tests."""

    def open(self, stage, limit):
        pass

    def check(self, stage=None):
        pass

    def heartbeat(self, *a, **k):
        pass


def _worker_callbacks():
    def on_progress(stage, done, total, elapsed):
        print("      {}  {} {}/{}  {:.0f}s elapsed".format(
            dt.datetime.now().strftime("%H:%M:%S"), stage, done, total,
            elapsed), flush=True)

    def on_log(msg):
        print("    {}".format(msg), flush=True)

    return on_progress, on_log


def run_in_worker(worker, asset, kind, cfg, tmp_dir, media, limit, item):
    """Hand one asset to the media worker and hold each stage to its ceiling.

    Everything that ever hung - the Drive read, ffmpeg, faster-whisper,
    easyocr - happens on the other side of this call, in a process the
    supervisor can kill. What comes back is either a finished measurement or
    a failure; there is no third outcome where the run is still waiting.

    If OCR is what gets killed, the asset is NOT lost. The worker checkpoints
    the frames, scenes and transcript before OCR opens, so this can spend a
    second, smaller budget reading the representative frames in a fresh
    worker and return a complete-but-SAMPLED asset. A video with its text
    read at four scene boundaries is worth having; a video dropped for
    running long is not.
    """
    from step2_supervisor import WorkerDied

    job = {"kind": kind, "asset": asset, "cfg": cfg,
           "tmp_dir": str(tmp_dir), "media_name": media.name}
    on_progress, on_log = _worker_callbacks()

    def on_stage(stage, budget):
        print("      stage {!r}  ceiling {:.0f}s".format(stage, budget),
              flush=True)

    try:
        outcome = worker.run(job, limit, item, on_progress=on_progress,
                             on_log=on_log, on_stage=on_stage)
    except WorkerDied as exc:
        # The worker died without being told to. That is not this asset's
        # fault in the way a bad file is, but the asset still did not finish,
        # and Rule 8 says anything incomplete is FAILED.
        raise AssetFailed("media worker lost: {}".format(exc))

    tag = outcome[0]
    if tag == "ok":
        return outcome[1]
    if tag == "killed":
        _, reason, stage, checkpoint = outcome
        if kind == "VIDEO" and stage in ("ocr", "ocr_sampled") and checkpoint:
            return salvage_ocr(worker, cfg, item, checkpoint, reason)
        raise AssetFailed(reason + " Abandoned so the run continues.")
    # An unexpected exception inside the worker arrives already labelled
    # "unexpected:Type:message", and its traceback has already been printed on
    # the worker's stderr. Re-raising it as something unanticipated here would
    # only add this process's traceback, which points at this line and not at
    # the fault, and would label the reason twice.
    raise AssetFailed(outcome[1])


def salvage_ocr(worker, cfg, item, checkpoint, reason):
    """OCR was killed. Read the representative frames in a fresh worker and
    finish the asset as SAMPLED - or, failing that, as NONE.

    The frames are still on disk: the run does not sweep an asset's temp
    directory until the asset is done with, precisely so this is possible.
    """
    frames = checkpoint["frames"]
    reps = representative_frames(frames, checkpoint["scenes"],
                                 need(cfg, "ocr_sampled_max_frames"))
    print("    OCR killed; retrying {} representative frame(s) of {}".format(
        len(reps), len(frames)), flush=True)

    out = dict(checkpoint)
    out.update({"observations": [], "ocr_seconds": 0.0,
                "ocr_seconds_per_frame": None,
                "ocr_frames_total": len(frames), "ocr_frames_processed": 0,
                "ocr_coverage": OCR_NONE, "ocr_degraded_reason": reason,
                "near_duplicate": near_duplicate_rate([]),
                "download_attempts": checkpoint.get("download_attempts", 1)})

    on_progress, on_log = _worker_callbacks()
    job = {"kind": "OCR_ONLY", "cfg": cfg, "frames": reps,
           "size": list(checkpoint["frame_size"])}
    try:
        outcome = worker.run(job, ocr_budget(len(reps) or 1, cfg), item,
                             on_progress=on_progress, on_log=on_log)
    except Exception as exc:                       # the worker would not start
        out["ocr_degraded_reason"] = "{} The sampled retry could not run: {}" \
            .format(reason, exc)
        return out

    if outcome[0] == "ok":
        got = outcome[1]
        out["observations"] = got["observations"]
        out["ocr_frames_processed"] = got["ocr_frames_processed"]
        out["ocr_coverage"] = (OCR_SAMPLED if got["ocr_frames_processed"]
                               else OCR_NONE)
        out["ocr_degraded_reason"] = (
            "{} On-screen text was read from {} representative frame(s) "
            "instead of all {}.".format(reason, got["ocr_frames_processed"],
                                        len(frames)))
    else:
        out["ocr_degraded_reason"] = "{} The sampled retry also failed: {}" \
            .format(reason, outcome[1])
    return out


def download_with_retry(s1, service, file_id, dest, attempts=4,
                        socket_timeout=180):
    """Drive downloads drop. V5 died mid-transfer on a socket the host
    aborted - a transport failure, not a fact about the file, so it is
    retried rather than recorded as an asset failure. A partial file is
    deleted before each retry so a resumed download cannot be spliced onto
    a truncated one."""
    import socket
    last = None
    previous = socket.getdefaulttimeout()
    for attempt in range(1, attempts + 1):
        try:
            if dest.exists():
                dest.unlink()
            # httplib2 under googleapiclient exposes no per-call timeout here,
            # so the socket default is pinned for the transfer. Without this a
            # stalled Drive read blocks forever instead of failing the asset.
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
                print("    download attempt {} failed ({}); retrying in {}s"
                      .format(attempt, type(exc).__name__, wait), flush=True)
                time.sleep(wait)
    raise AssetFailed("download failed after {} attempts: {}:{}".format(
        attempts, type(last).__name__, str(last)[:200]))


def decode_check(path, expect_w, expect_h):
    """A byte-match proves the transfer; it does not prove the JPEG is valid.
    Decode one frame per video and confirm it opens at the expected size."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        got = im.size
    if got != (expect_w, expect_h):
        raise AssetFailed(
            "decoded frame is {}x{}, expected {}x{}".format(
                got[0], got[1], expect_w, expect_h))
    return got


# --------------------------------------------------------------------------
# per-asset processing (no database, no storage)
# --------------------------------------------------------------------------

def process_video(asset, video_path, tmp_dir, cfg, do_ocr=True,
                  deadline=None, on_checkpoint=None):
    """Everything measurable about one video. Raises AssetFailed.

    Each stage opens its own budget on the clock, so the watchdog knows what
    it is timing and can size OCR against the frame count. on_checkpoint, if
    given, is called with everything measured BEFORE on-screen text is read:
    if OCR then has to be killed outright, the scenes, the transcript and the
    frames survive the kill and the asset can still complete."""
    deadline = deadline or NullClock()
    duration = float(asset["duration_seconds"])
    interval = float(need(cfg, "frame_interval_seconds"))
    origin = need(cfg, "frame_sample_origin")
    tol = int(need(cfg, "frame_count_tolerance"))
    max_frames = int(need(cfg, "max_frames_per_asset"))
    disp_w = int(asset["display_width_px"])
    disp_h = int(asset["display_height_px"])
    fps = float(asset["fps"])

    # --- frames -------------------------------------------------------
    times = planned_frame_times(duration, interval, origin, max_frames)
    size = target_size(disp_w, disp_h,
                       int(need(cfg, "frame_width_px")),
                       int(need(cfg, "frame_height_px")))
    frame_dir = tmp_dir / "frames"
    deadline.open("extract_frames", fixed_budget(cfg))
    files, kept_times, frame_notes = extract_frames(
        video_path, frame_dir, times, size,
        need(cfg, "frame_jpeg_quality"), interval, deadline=deadline)

    expected_floor = int(math.floor(duration / interval))
    if abs(len(files) - expected_floor) > tol:
        raise AssetFailed(
            "frame count {} deviates from floor(duration/interval)={} by more "
            "than frame_count_tolerance={}".format(
                len(files), expected_floor, tol))
    if len(files) != len(kept_times):
        raise AssetFailed(
            "extracted {} frames but kept {} timestamps".format(
                len(files), len(kept_times)))

    frames = []
    for i, (t, path) in enumerate(zip(kept_times, files)):
        frames.append({"index": i, "timestamp": round(t, 3), "path": str(path),
                       "width": size[0], "height": size[1],
                       "source_frame_index": int(round(t * fps))})

    # A byte-identical upload is not proof of a valid image. Decode one.
    decode_check(Path(frames[0]["path"]), size[0], size[1])

    # --- scenes -------------------------------------------------------
    mode = need(cfg, "scene_segmentation_mode")
    if mode not in ("cuts", "cuts_and_drift"):
        raise SystemExit("unknown scene_segmentation_mode: {!r}".format(mode))
    deadline.open("scenes", fixed_budget(cfg))
    cuts = detect_cuts(video_path, need(cfg, "scene_change_threshold"))
    boundaries = [(c, "cut") for c in cuts]
    drift, drift_diag = [], []
    if mode == "cuts_and_drift":
        # Drift is measured on the frames already extracted, so the boundaries
        # derive from exactly the images that get stored - not a second,
        # differently-sampled decode.
        drift, drift_diag = detect_drift(frames, cuts, cfg, duration)
        boundaries += [(d, "drift") for d in drift]
    scenes = build_scenes(duration, boundaries, cfg, mode)
    coverage, problems = verify_scene_coverage(scenes, duration)
    rate = check_scene_rate(scenes, duration, cfg)
    # Evidence for the report when cuts finds nothing: what the largest
    # scores actually were, against the threshold that was applied.
    score_profile = scene_score_profile(video_path) if len(scenes) == 1 else []

    # --- speech -------------------------------------------------------
    if bool(need(cfg, "transcription_enabled")) and asset["has_audio"]:
        deadline.open("transcribe", fixed_budget(cfg))
        tr = transcribe(video_path, tmp_dir, cfg, duration)
    else:
        tr = {"status": "EXCLUDED", "language": None,
              "language_probability": None,
              "chunks": [{"start": 0.0, "end": round(duration, 3), "text": "",
                          "language": None, "confidence": None}],
              "reason": "no audio stream" if not asset["has_audio"]
                        else "transcription_enabled is false"}

    # --- on-screen text -----------------------------------------------
    # Everything above this line is measured and, from here on, must survive.
    # The checkpoint hands it to the parent before OCR starts, so that killing
    # OCR costs OCR and not the asset.
    checkpoint = {"frames": frames, "raw_cuts": cuts, "scenes": scenes,
                  "coverage": coverage, "coverage_problems": problems,
                  "scene_rate_per_min": rate, "transcript": tr,
                  "frame_size": size, "expected_floor": expected_floor,
                  "raw_drift": drift, "drift_diag": drift_diag,
                  "planned_frames": len(times), "frame_notes": frame_notes,
                  "scene_score_profile": score_profile}
    if on_checkpoint:
        on_checkpoint(checkpoint)

    ocr = {"observations": [], "ocr_seconds": 0.0,
           "ocr_frames_total": len(frames), "ocr_frames_processed": 0,
           "ocr_coverage": OCR_NONE, "ocr_degraded_reason": None,
           "near_duplicate": near_duplicate_rate([])}
    if do_ocr and bool(need(cfg, "ocr_enabled")):
        ocr = read_on_screen_text(frames, scenes, size, cfg, deadline)
    else:
        ocr["ocr_degraded_reason"] = "ocr_enabled is false"

    out = dict(checkpoint)
    out.update(ocr)
    out["ocr_seconds_per_frame"] = (
        round(ocr["ocr_seconds"] / ocr["ocr_frames_processed"], 2)
        if ocr["ocr_frames_processed"] else None)
    out["ocr_seconds"] = round(ocr["ocr_seconds"], 2)
    return out


def read_on_screen_text(frames, scenes, size, cfg, deadline):
    """Read every frame if the budget allows it; if it does not, read the
    representative frames instead and say so.

    Abandoning the asset was the old behaviour and it is the wrong trade: an
    asset with on-screen text read from four frames of 190 is worth far more
    than one with none, PROVIDED the catalogue records which it is. That
    proviso is the whole of ocr_coverage.
    """
    reader = ocr_reader(cfg)
    observations, per_frame_text = [], []
    done_indices = set()
    ocr_seconds = 0.0

    def read(batch, stage):
        nonlocal ocr_seconds
        for n_done, fr in enumerate(batch, 1):
            deadline.check(stage)
            t_start = time.time()
            found = ocr_frame(Path(fr["path"]), reader, cfg, size[0], size[1])
            deadline.heartbeat(stage, n_done, len(batch))
            ocr_seconds += time.time() - t_start
            for ob in found:
                ob["timestamp"] = fr["timestamp"]
                ob["_frame_index"] = fr["index"]
                observations.append(ob)
            done_indices.add(fr["index"])
            per_frame_text.append(
                frozenset(ob["normalized_text"] for ob in found
                          if ob["accepted_for_search"]))

    coverage, reason = OCR_FULL, None
    deadline.open("ocr", ocr_budget(len(frames), cfg))
    try:
        read(frames, "ocr")
    except StageOverBudget as exc:
        # The stage is over budget, not the asset. Fall back to the frames
        # that carry the most information per second spent: one per scene.
        # Frames already read are kept - they are real observations that have
        # already been paid for - and only the ones still missing are read.
        coverage, reason = OCR_SAMPLED, str(exc)
        reps = representative_frames(
            frames, scenes, need(cfg, "ocr_sampled_max_frames"))
        todo = [fr for fr in reps if fr["index"] not in done_indices]
        deadline.open("ocr_sampled", ocr_budget(len(todo) or 1, cfg))
        try:
            read(todo, "ocr_sampled")
        except StageOverBudget as exc2:
            reason = "{}; the sampled retry also ran out: {}".format(
                str(exc), exc2)

    if not done_indices:
        coverage = OCR_NONE

    return {"observations": observations, "ocr_seconds": ocr_seconds,
            "ocr_frames_total": len(frames),
            "ocr_frames_processed": len(done_indices),
            "ocr_coverage": coverage, "ocr_degraded_reason": reason,
            # How much of the per-second sampling is redundant. Two
            # consecutive frames of a static shot read the same text, so this
            # is what a full-library run could skip. Reported, never acted on.
            "near_duplicate": near_duplicate_rate(per_frame_text)}


def process_image(asset, image_path, tmp_dir, cfg, do_ocr=True):
    """Stills get on-screen text only: no frames, no scenes, no speech. The
    OCR run is also where this asset's path parse is recorded."""
    box_w = int(need(cfg, "frame_width_px"))
    box_h = int(need(cfg, "frame_height_px"))
    disp_w = int(asset["display_width_px"])
    disp_h = int(asset["display_height_px"])
    size = target_size(disp_w, disp_h, box_w, box_h)

    # Pillow, not ffmpeg. ffmpeg decodes HEIC through a complex filtergraph
    # and then refuses a simple -vf on the same stream, which is what failed
    # on P9. Step 1 already reads stills with Pillow + pillow-heif, so this
    # is the same decoder that produced the dimensions being scaled to.
    small = tmp_dir / "ocr_input.jpg"
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        with Image.open(image_path) as im:
            im.load()
            im = im.convert("RGB").resize(size, Image.LANCZOS)
            im.save(small, "JPEG",
                    quality=int(need(cfg, "frame_jpeg_quality")))
    except Exception as exc:
        raise AssetFailed("image downscale failed: {}:{}".format(
            type(exc).__name__, str(exc)[:250]))

    observations = []
    coverage, reason, processed = OCR_NONE, "ocr_enabled is false", 0
    if do_ocr and bool(need(cfg, "ocr_enabled")):
        reader = ocr_reader(cfg)
        for ob in ocr_frame(small, reader, cfg, size[0], size[1]):
            ob["timestamp"] = None
            observations.append(ob)
        # A still is one frame, so reading it at all is reading all of it.
        # It still records coverage, because a reader asking "was this looked
        # at?" must get the same kind of answer for a photograph as for a
        # video - that is the whole point of the column.
        coverage, reason, processed = OCR_FULL, None, 1
    try:
        small.unlink(missing_ok=True)
    except OSError:
        pass
    return {"observations": observations, "ocr_size": size,
            "ocr_frames_total": 1, "ocr_frames_processed": processed,
            "ocr_coverage": coverage, "ocr_degraded_reason": reason}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="compute everything, write nothing")
    mode.add_argument("--commit", action="store_true",
                      help="write to test_v1 and upload keyframes")
    ap.add_argument("--only", default=None,
                    help="comma-separated items (V1,V7) for a narrower run")
    args = ap.parse_args()
    committing = bool(args.commit)

    from db import (Reconnecting, SecretMissing, WrongTarget)
    from preflight import (Step2Preflight, PreflightFailed, PHASE_PRE_RUN,
                           PHASE_RESUME, KEYFRAME_BUCKET)
    from step2_writer import Step2Writer
    import step1_ingest as s1

    # --- connect and gate ----------------------------------------------
    try:
        handle = Reconnecting()
    except (SecretMissing, WrongTarget) as exc:
        raise SystemExit("step2 cannot start.\n{}".format(exc))
    conn, hint = handle.conn, handle.hint
    print("database:   {}".format(hint))

    # The probe goes through the handle, so the gate repairs a connection the
    # pooler reaped during a long asset instead of failing on it. Step 2 only
    # touches the database at the end of each asset, so the connection sits
    # idle for the whole of the next asset's processing - sixteen minutes of
    # OCR for the 189-second video, which is well past any pooler's patience.
    def probe(sql):
        c = handle.ensure()
        with c.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            row = cur.fetchone()
            c.rollback()
            return dict(zip(cols, row))

    pf = Step2Preflight(probe, target_hint=hint)

    baseline = current_step2_counts(handle.ensure())
    done_items, done_ids = completed_assets(handle.ensure(),
                                            STEP2_CONFIG_VERSION)
    resuming = any(baseline.values())
    try:
        pf.assert_ready(PHASE_RESUME if resuming else PHASE_PRE_RUN,
                        baseline=baseline)
    except PreflightFailed as exc:
        handle.close()
        raise SystemExit(str(exc))
    print("pre-flight: PASSED (phase={})".format(
        "resume" if resuming else "pre_run"))
    if resuming:
        print("resume:     {} asset(s) already complete: {}".format(
            len(done_items),
            ", ".join(sorted(done_items, key=item_sort)) or "(none)"))
        print("            inherited rows: {}".format(
            ", ".join("{}={}".format(k, v)
                      for k, v in sorted(baseline.items()) if v)))

    cfg = load_db_config(handle.ensure())
    # temp_directory and media_kind_by_extension are local-only by design -
    # Step 1 established that and the database deliberately does not hold
    # them. Everything else comes from pipeline_config.
    file_cfg = s1.load_file_config()
    s1.assert_configs_agree(file_cfg, cfg)
    for k in s1.LOCAL_ONLY_CONFIG_KEYS:
        cfg[k] = file_cfg[k]
    print("config:     {} keys from test_v1.pipeline_config + {} local-only"
          .format(len(cfg) - len(s1.LOCAL_ONLY_CONFIG_KEYS),
                  len(s1.LOCAL_ONLY_CONFIG_KEYS)))
    handle.conn.rollback()

    # --- storage ---------------------------------------------------------
    store = None
    if committing:
        from storage import KeyframeStore, StorageKeyMissing
        try:
            store = KeyframeStore(
                need(cfg, "keyframe_storage_bucket"),
                verify_readback=bool(need(cfg, "storage_verify_readback")))
        except StorageKeyMissing as exc:
            handle.close()
            raise SystemExit(
                "--commit cannot start: keyframes must be uploaded and no "
                "service role key is available.\n{}".format(exc))
        if store.bucket != KEYFRAME_BUCKET:
            handle.close()
            raise SystemExit("bucket mismatch: config says {!r}, pre-flight "
                             "checks {!r}".format(store.bucket, KEYFRAME_BUCKET))
        print("storage:    bucket {!r}, read-back verify {}".format(
            store.bucket, store.verify_readback))

    # --- new config snapshot --------------------------------------------
    config_version = STEP2_CONFIG_VERSION
    if committing:
        c = handle.ensure()
        fp, how = take_snapshot(c, config_version, STEP2_SNAPSHOT_NOTE)
        c.commit()
        print("snapshot:   {} fingerprint {} ({})".format(
            config_version, (fp or "")[:12], how))
    else:
        print("snapshot:   would take {} (dry run takes none)".format(
            config_version))

    # --- what to process --------------------------------------------------
    assets = load_assets(handle.ensure())
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        assets = [a for a in assets if (a["item"] or "").upper() in want]
    videos = [a for a in assets if a["media_kind"] == "VIDEO"]
    images = [a for a in assets if a["media_kind"] == "IMAGE"]
    print("assets:     {} video, {} image".format(len(videos), len(images)))
    print("")

    # The parent no longer touches Drive or ffmpeg or the models: everything
    # that can hang happens in the worker, where it can be killed. The service
    # is still built once here so the report can name the account and so a bad
    # credential fails before the first asset rather than inside a child.
    service, sa_email = s1.build_drive_service()
    del service

    from step2_supervisor import AssetWorker
    worker = AssetWorker(log=lambda m: print(m, flush=True))
    # A Ctrl-C or an unhandled error must not leave a worker - and its ffmpeg
    # children - alive holding a patient's media open.
    import atexit
    atexit.register(worker.stop)

    temp_root = PROJECT_ROOT / need(cfg, "temp_directory")
    temp_root.mkdir(parents=True, exist_ok=True)

    writer = None
    if committing:
        writer = Step2Writer(handle, pf, cfg, config_version, store)
        writer.baseline = dict(baseline)

    results, failures = [], []

    skipped = []
    for asset in videos + images:
        item = asset["item"]
        kind = asset["media_kind"]
        if committing and asset["asset_id"] in done_ids:
            skipped.append(item)
            print("[{}] {} ({}) ... already complete, skipped".format(
                item, asset["file_name"], kind.lower()), flush=True)
            continue
        limit = asset_deadline(asset, cfg)
        tmp_dir = temp_root / "step2_{}".format(item)
        rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        media = tmp_dir / asset["file_name"]
        started = dt.datetime.now(dt.timezone.utc)
        print("[{}] {} ({}) ...".format(item, asset["file_name"], kind.lower()),
              flush=True)

        path_parse = parse_relative_path(asset["relative_path"],
                                         asset["parent_visible"])
        rec = {"item": item, "kind": kind, "file_name": asset["file_name"],
               "asset_id": str(asset["asset_id"]), "path_parse": path_parse,
               "status": "OK", "reason": None}

        try:
            out = run_in_worker(worker, asset, kind, cfg, tmp_dir, media,
                                limit, item)
            if out.get("download_attempts", 1) > 1:
                rec["download_attempts"] = out["download_attempts"]

            if kind == "VIDEO":
                rec.update({
                    "duration": float(asset["duration_seconds"]),
                    "frames": len(out["frames"]),
                    "expected_floor": out["expected_floor"],
                    "planned_frames": out["planned_frames"],
                    "frame_notes": out["frame_notes"],
                    "frame_size": "{}x{}".format(*out["frame_size"]),
                    "raw_cuts": out["raw_cuts"],
                    "raw_drift": out["raw_drift"],
                    "drift_diag": out["drift_diag"],
                    "scene_score_profile": out["scene_score_profile"],
                    "scenes": len(out["scenes"]),
                    "scene_bounds": [(s["start"], s["end"]) for s in out["scenes"]],
                    "coverage": out["coverage"],
                    "coverage_problems": out["coverage_problems"],
                    "scene_rate_per_min": out["scene_rate_per_min"],
                    "transcript_status": out["transcript"]["status"],
                    "language": out["transcript"].get("language"),
                    "language_probability": out["transcript"].get("language_probability"),
                    "transcript_chunks": len(out["transcript"]["chunks"]),
                    "transcript_reason": out["transcript"].get("reason"),
                    "transcript_text": " ".join(
                        c["text"] for c in out["transcript"]["chunks"]).strip(),
                    "ocr_observations": len(out["observations"]),
                    "ocr_accepted": sum(1 for o in out["observations"]
                                        if o["accepted_for_search"]),
                    "ocr_seconds": out["ocr_seconds"],
                    "ocr_seconds_per_frame": out["ocr_seconds_per_frame"],
                    "ocr_coverage": out["ocr_coverage"],
                    "ocr_frames_total": out["ocr_frames_total"],
                    "ocr_frames_processed": out["ocr_frames_processed"],
                    "ocr_degraded_reason": out["ocr_degraded_reason"],
                    "near_duplicate": out["near_duplicate"],
                    "ocr_top": sorted(
                        {o["normalized_text"] for o in out["observations"]
                         if o["accepted_for_search"]})[:40],
                })
                print("    frames {} (floor {})  scenes {}  coverage {:.3f}/{:.3f}s"
                      "  transcript {}  ocr {} [{} {}/{} frames]".format(
                          rec["frames"], rec["expected_floor"], rec["scenes"],
                          rec["coverage"], rec["duration"],
                          rec["transcript_status"], rec["ocr_observations"],
                          rec["ocr_coverage"], rec["ocr_frames_processed"],
                          rec["ocr_frames_total"]),
                      flush=True)
                if rec["ocr_coverage"] != OCR_FULL:
                    print("    OCR DEGRADED: {}".format(
                        rec["ocr_degraded_reason"]), flush=True)
                if out["coverage_problems"]:
                    for p in out["coverage_problems"]:
                        print("    COVERAGE PROBLEM: {}".format(p))

                if committing:
                    now = dt.datetime.now(dt.timezone.utc)
                    writer.write_scenes(asset, out["scenes"], out["coverage"],
                                        out["coverage_problems"], started, now,
                                        path_parse)
                    if out["coverage_problems"]:
                        raise AssetFailed("scene coverage incomplete: {}".format(
                            "; ".join(out["coverage_problems"])))
                    uploads = writer.upload_frames(asset, out["frames"])
                    up = writer.write_keyframes(asset, out["frames"],
                                                out["scenes"], uploads,
                                                started,
                                                dt.datetime.now(dt.timezone.utc))
                    rec["stored"] = up["stored"]
                    rec["derivatives"] = up["derivatives"]
                    if up["failed"]:
                        rec["status"] = "PARTIAL"
                        rec["reason"] = "{} frame(s) not stored".format(
                            len(up["failed"]))
                    writer.write_transcript(asset, out["transcript"], started,
                                            dt.datetime.now(dt.timezone.utc))
                    for ob in out["observations"]:
                        fr = out["frames"][ob["_frame_index"]]
                        ob["keyframe_id"] = fr.get("keyframe_id")
                        ob["scene_id"] = fr.get("scene_id")
                    writer.write_ocr(asset, out["observations"], started,
                                     dt.datetime.now(dt.timezone.utc),
                                     path_parse, coverage=out)
                    print("    wrote: {} scenes, {} keyframes, {} derivatives, "
                          "{} transcript chunks, {} ocr".format(
                              len(out["scenes"]), len(out["frames"]),
                              up["derivatives"],
                              len(out["transcript"]["chunks"]),
                              len(out["observations"])), flush=True)
            else:
                rec.update({
                    "ocr_observations": len(out["observations"]),
                    "ocr_accepted": sum(1 for o in out["observations"]
                                        if o["accepted_for_search"]),
                    "ocr_coverage": out["ocr_coverage"],
                    "ocr_frames_total": out["ocr_frames_total"],
                    "ocr_frames_processed": out["ocr_frames_processed"],
                    "ocr_degraded_reason": out["ocr_degraded_reason"],
                    "ocr_top": sorted(
                        {o["normalized_text"] for o in out["observations"]
                         if o["accepted_for_search"]})[:40],
                })
                print("    ocr {} observation(s); path parse {}".format(
                    rec["ocr_observations"], path_parse["parse_status"]),
                    flush=True)
                if committing:
                    writer.write_ocr(asset, out["observations"], started,
                                     dt.datetime.now(dt.timezone.utc),
                                     path_parse, coverage=out)

        except BaseException as exc:
            # The catch-all. BaseException, not Exception, so that a
            # MemoryError or a KeyboardInterrupt inside one asset is still
            # recorded rather than tearing the run down silently - except
            # KeyboardInterrupt itself, which is re-raised because a human
            # asking to stop should be obeyed.
            anticipated = isinstance(exc, AssetFailed)
            rec["status"] = "FAILED"
            rec["reason"] = (str(exc) if anticipated else
                             "unexpected:{}:{}".format(type(exc).__name__, exc))
            failures.append(rec)
            print("    FAILED  {}{}".format("" if anticipated else "unexpected: ",
                                            str(exc)[:300]), flush=True)
            if not anticipated:
                traceback.print_exc()

            # Say so in the database, whatever went wrong. An asset that is
            # merely absent is indistinguishable from one never attempted.
            if writer is not None:
                ok, why = writer.write_failure(
                    asset, "STEP2_SCENES", rec["reason"], started,
                    dt.datetime.now(dt.timezone.utc),
                    error_code="WATCHDOG" if "watchdog" in rec["reason"]
                               else "ASSET_FAILED")
                if not ok:
                    rec["reason"] += " | and the FAILED row could not be "\
                                     "written: {}".format(why)
                    print("    (could not record the failure: {})".format(why),
                          flush=True)

            if isinstance(exc, KeyboardInterrupt):
                rmtree(tmp_dir)
                raise
        finally:
            # Rule 4: media is never kept, and neither are extracted frames.
            rmtree(tmp_dir)

        results.append(rec)

    # The worker is shut down BEFORE the sweep: it holds the media open, and
    # a file cannot be deleted on Windows while a live handle points at it.
    worker.stop()

    # nothing of the patients' may survive the run
    rmtree(temp_root / "step2_tmp")
    leftovers = [p for p in temp_root.rglob("*") if p.is_file()]
    for p in leftovers:
        print("WARNING: sweeping leftover {}".format(p))
        try:
            p.unlink()
        except OSError:
            pass

    print("")
    if skipped:
        print("skipped {} already-complete asset(s): {}".format(
            len(skipped), ", ".join(skipped)))
    print("processed {}  ok {}  failed {}".format(
        len(results), sum(1 for r in results if r["status"] == "OK"),
        len(failures)))
    print("media worker: {} spawn(s), {} killed on the ceiling".format(
        worker.spawns, worker.kills))

    write_report(results, cfg, config_version, committing, sa_email, store)
    path = REPORT_COMMIT if committing else REPORT_DRY
    print("wrote {}".format(path))
    handle.close()
    if handle.reconnects:
        print("database reconnects during the run: {}".format(
            handle.reconnects))
    return 1 if failures else 0


def write_report(results, cfg, config_version, committing, sa_email, store):
    L = []
    L.append("# Step 2 - {}".format("commit report" if committing
                                    else "dry run (nothing was written)"))
    L.append("")
    L.append("- Config version: `{}`".format(config_version))
    L.append("- Engines: ffmpeg frames and cuts, faster-whisper `{}` speech, "
             "`{}` on-screen text. No AI vision model.".format(
                 cfg.get("transcription_model"), cfg.get("ocr_engine")))
    L.append("- Service account: `{}`".format(sa_email))
    if store is not None:
        L.append("- Storage: bucket `{}`, uploaded {}, read-back verified {}"
                 .format(store.bucket, store.uploaded, store.verified))
    L.append("")

    vids = [r for r in results if r["kind"] == "VIDEO"]
    imgs = [r for r in results if r["kind"] == "IMAGE"]

    L.append("## Videos")
    L.append("")
    L.append("| Item | File | Duration | Frames | floor(d) | Scenes | Coverage |"
             " Rate/min | Transcript | Lang | OCR | Status |")
    L.append("|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---|")
    for r in vids:
        L.append("| {} | `{}` | {:.3f} | {} | {} | {} | {}/{:.3f} | {} | {} | {} "
                 "| {} | {} |".format(
                     r["item"], r["file_name"], r.get("duration", 0),
                     r.get("frames", "-"), r.get("expected_floor", "-"),
                     r.get("scenes", "-"), r.get("coverage", "-"),
                     r.get("duration", 0), r.get("scene_rate_per_min", "-"),
                     r.get("transcript_status", "-"), r.get("language", "-"),
                     r.get("ocr_observations", "-"), r["status"]))
    L.append("")

    L.append("## Scene boundaries")
    L.append("")
    for r in vids:
        if not r.get("scene_bounds"):
            continue
        bounds = ", ".join("[{:.3f}, {:.3f}]".format(a, b)
                           for a, b in r["scene_bounds"])
        L.append("- **{}** `{}` - {} scene(s): {}".format(
            r["item"], r["file_name"], r["scenes"], bounds))
        if r.get("raw_cuts"):
            L.append("  - raw cut candidates: {}".format(r["raw_cuts"]))
        if r.get("raw_drift"):
            L.append("  - drift boundaries: {}".format(r["raw_drift"]))
        dd = r.get("drift_diag") or []
        if dd:
            vals = [d for _t, d, _k in dd if d is not None]
            if vals:
                L.append("  - drift distance from anchor: min {:.4f}, "
                         "max {:.4f}, {} of {} samples above threshold"
                         .format(min(vals), max(vals),
                                 sum(1 for _t, d, k in dd if k == "above"),
                                 len(vals)))
        if r.get("scene_score_profile"):
            L.append("  - largest scene scores in the file (threshold {}): {}"
                     .format(cfg.get("scene_change_threshold"),
                             ", ".join("{}s={}".format(t, v)
                                       for t, v in r["scene_score_profile"])))
        if r.get("coverage_problems"):
            L.append("  - **COVERAGE PROBLEMS**: {}".format(
                "; ".join(r["coverage_problems"])))
        for n in (r.get("frame_notes") or []):
            L.append("  - frame note: {}".format(n))
    L.append("")

    L.append("## Transcripts")
    L.append("")
    for r in vids:
        L.append("- **{}** `{}` - {} ({} chunk(s), language {})".format(
            r["item"], r["file_name"], r.get("transcript_status"),
            r.get("transcript_chunks"), r.get("language")))
        if r.get("transcript_reason"):
            L.append("  - reason: {}".format(r["transcript_reason"]))
        txt = (r.get("transcript_text") or "").strip()
        if txt:
            L.append("  - text: {}".format(txt[:600]))
    L.append("")

    L.append("## On-screen text accepted for search")
    L.append("")
    for r in results:
        top = r.get("ocr_top") or []
        if not top:
            continue
        L.append("- **{}** `{}` - {} observation(s), {} accepted: {}".format(
            r["item"], r["file_name"], r.get("ocr_observations"),
            r.get("ocr_accepted"), "; ".join(top)))
    L.append("")

    L.append("## OCR cost and redundancy (for sizing a full-library run)")
    L.append("")
    L.append("| Item | Frames | OCR seconds | s/frame | Frames with text | "
             "Consecutive pairs identical | Near-identical (Jaccard >= 0.9) |")
    L.append("|---|---:|---:|---:|---:|---|---|")
    tot_f = tot_s = 0
    for r in vids:
        nd = r.get("near_duplicate") or {}
        if r.get("ocr_seconds") is None:
            continue
        tot_f += r.get("frames") or 0
        tot_s += r.get("ocr_seconds") or 0
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["item"], r.get("frames"), r.get("ocr_seconds"),
            r.get("ocr_seconds_per_frame"), nd.get("frames_with_text"),
            "{} of {} ({}%)".format(nd.get("identical"), nd.get("pairs"),
                                    nd.get("identical_pct")),
            "{} of {} ({}%)".format(nd.get("near_identical"), nd.get("pairs"),
                                    nd.get("near_identical_pct"))))
    if tot_f:
        L.append("")
        L.append("**Overall: {} frames, {:.0f}s of OCR, {:.2f}s per frame.**"
                 .format(tot_f, tot_s, tot_s / tot_f))
    L.append("")

    L.append("## Patient name and session date from the folder path")
    L.append("")
    L.append("This schema has no patients or sessions table, so the parse is "
             "recorded in the metadata of each asset's Step 2 analysis_runs "
             "row rather than as a first-class record. That is a schema gap, "
             "not a parsing result - see the report notes.")
    L.append("")
    L.append("| Item | relative_path | Patient | Session date | Parse | "
             "Path from Drive scan |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        pp = r["path_parse"]
        L.append("| {} | `{}` | {} | {} | {} | {} |".format(
            r["item"], pp["relative_path"], pp["patient_name"] or "-",
            pp["session_date"] or "-", pp["parse_status"],
            "yes" if pp["path_from_drive_scan"] else "**no (from manifest)**"))
    L.append("")

    bad = [r for r in results if r["status"] != "OK"]
    if bad:
        L.append("## Assets not fully COMPLETED")
        L.append("")
        for r in bad:
            L.append("- **{}** `{}` - {}: {}".format(
                r["item"], r["file_name"], r["status"], r["reason"]))
        L.append("")

    path = REPORT_COMMIT if committing else REPORT_DRY
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
