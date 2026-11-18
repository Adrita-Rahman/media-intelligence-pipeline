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
  6 Every tunable comes from content_v1.pipeline_config, never a constant here.
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
content_v1.asset_ocr_coverage. Before that field existed, a video with no text
in it and a video that was never read both held zero ocr_observations and
were indistinguishable.

Usage:
    python scripts/step2_analyse.py --dry-run   # computes everything, writes nothing
    python scripts/step2_analyse.py --commit    # writes to content_v1 and Storage
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

REPORT_DRY = PROJECT_ROOT / "reports" / "content_v1" / "step2_dryrun.md"
REPORT_COMMIT = PROJECT_ROOT / "reports" / "content_v1" / "step2_commit.md"

# 2.2.1 (12 Oct 2026, batch 2): frame uploads are retried. step2_writer.upload_frames
# tries each frame up to UPLOAD_ATTEMPTS times, waiting 2, 4, 8, 16 s between tries,
# before the frame counts as not stored (PARTIAL_UPLOAD). Batch 2 lost exactly one
# frame (t=0, the first upload after a long OCR stretch) on B2-V15, B2-V20 and B2-V29.
# Nothing else changed: no setting, no stage, no other write.
# 2.3.0 (13 Oct 2026): --scene-review FILE - a hand-checked list of detected scenes
# to merge, per asset, applied after segmentation and before the scene-count
# checks; recorded on the STEP2_SCENES run as metadata.scene_review. Made for
# example.mp4 (31 detected scenes, limit
# scene_edited_max_count_absolute_per_asset = 30; the operator counted about 26 cuts
# and none of the 31 was short, blank or a near-duplicate, so no automatic rule
# fitted). The run refuses an asset whose detected scenes differ from what the
# reviewer saw. No setting changed; an asset not named in the file is untouched.
# 2.4.0 (13 Oct 2026): a manually set edit_class is permanent. step2_writer's
# edit confirmation (run inside every scenes write, full run or --stages scenes)
# no longer changes edit_class when edit_class_evidence has a manual_override; it
# still records its measurement, with manual_override_kept. Found when the 13 Oct
# example.mp4 (B2-V28) back to EDITED after migration 018:
# source_type was still EDITED and the raw cut check found no cut. No setting
# changed; assets without a manual override behave exactly as in 2.3.0.
# 2.5.0 (13 Oct 2026): --transcribe-raw ITEM=REASON (repeatable) transcribes a
# RAW video despite the 25 Sep raw-footage policy, as a recorded per-file
# exception: the reason goes on the STEP2_TRANSCRIPT run as
# metadata.policy_exception. example.mov (doctor-patient
# conversation; the operator, 13 Oct). transcription_edit_classes and every other
# setting are unchanged; a video not named is transcribed exactly as in 2.4.0.
# 2.6.0 (13 Oct 2026): --scene-review also applies to --stages scenes, so a
# hand-checked merge can be made on an asset Step 2 already processed without
# redoing its frames, transcript or OCR (replace_scenes re-links them by time);
# recorded as metadata.scene_review on the new STEP2_SCENES run. Made for B2-V19
# 'example.mov': scene 3 (37.2-38.0 s) is a glitch
# transition with no stored frame (the operator, 13 Oct). No setting changed.
# 2.7.0 (19 Oct 2026): --transcript-language ITEM=LANG:REASON forces the transcript
# language for one file (the forced-language path step2_transcribe already has for
# transcription_language), so language detection and the decode-score override are
# skipped for it; LANG must be in transcription_allowed_languages. The reason goes on
# the STEP2_TRANSCRIPT run in metadata.language_choice. Made for B3-V44
# example.mp4: an English conversation detected as Malay (0.68)
# and decoded as a Malay translation, every segment then excluded (user, 19 Oct).
# 2.7.0 also: --ignore-no-speech ITEM=REASON turns off only the two no_speech_prob
# checks (transcript_no_speech_prob_max and _corroborating) for one file, by passing
# 1.0 for that file's run only; every other quality check still applies. The reason
# goes on the STEP2_TRANSCRIPT run as metadata.no_speech_check_override. Made for
# B3-V44: Whisper's no_speech_prob is 0.65/0.86 on whole 30 s windows of a clear
# English conversation (user listened, 19 Oct). transcript_no_speech_prob_max in
# pipeline_config is unchanged.
# No setting changed; a video not named is transcribed exactly as in 2.6.0.
# 2.8.0 (28 Oct 2026): an EDITED video over scene_edited_max_count_absolute_per_asset
# is no longer failed when step2_scenes_v2 1.5.0 has grouped its shots into moments
# (scene_edited_consolidate_enabled/_similarity/_min_seconds; user, 28 Oct: "only
# collect the scenes where something different is happening"). The absolute cap is
# skipped for those groups, the per-minute ceiling still applies, and the run
# records metadata.scene_consolidated {shots, groups}. Made for B1-V29 MDG_6235 and
# B1-V30 MDG_0375 (compilation reels, 322 and 411 shots). Without the config keys
# every asset behaves exactly as in 2.7.0. Also: the stage re-run report no longer
# crashes on an undefined sa_email.
PROCESSOR_VERSION = "step2_analyse/2.8.0"

# Runtime-only key set by --skip-transcription. The underscore marks it as not
# a pipeline_config key; it lives only in this run's in-memory cfg.
SKIP_TRANSCRIPTION_KEY = "_run_skip_transcription_reason"
# Runtime-only key set by --scene-review (2.3.0): {step1 item: review}.
SCENE_REVIEW_KEY = "_run_scene_review"
# Runtime-only key set by --transcribe-raw (2.5.0): {step1 item: reason}.
TRANSCRIBE_RAW_KEY = "_run_transcribe_raw"
# Runtime-only key set by --transcript-language (2.7.0): {step1 item: (lang, reason)}.
TRANSCRIPT_LANGUAGE_KEY = "_run_transcript_language"
# Runtime-only key set by --ignore-no-speech (2.7.0): {step1 item: reason}.
IGNORE_NO_SPEECH_KEY = "_run_ignore_no_speech"

# A NEW config version. Step 1 rows keep step1_2026_09_22_a, so the two steps
# stay distinguishable in every stamped table.
STEP2_CONFIG_VERSION = "step2_2026_09_25_b"
# Per-asset copies a stage re-run makes before replacing rows go to
# <table>_backup_<day><BACKUP_SUFFIX>. 28 Sep: whole-table backups were taken
# first under the plain date, so the per-asset copies take the letter.
BACKUP_SUFFIX = "b"
STEP2_SNAPSHOT_NOTE = (
    "2026-09-28: step2_scenes_v2/1.2.0 heals short flashes before scoring "
    "(scene_edited_flash_*), scene_edited_confirm_threshold 0.20 -> 0.12; "
    "each re-run takes its own --config-version. "
    "step2_analyse/2.2.0: transcript - the other allowed language is decoded "
    "too and replaces the detector's choice only by "
    "transcript_language_override_margin; a span a loop guard fires on is "
    "re-decoded once without hotwords; a single-word repeat guard; speech "
    "lost to a loop makes the run PARTIAL; stage-only re-runs "
    "(--stages transcript|scenes). "
    "step2_analyse/2.1.0: scene segmentation moved to step2_scenes_v2 with one "
    "path per Step 1 edit_class - RAW (sustained change of situation, merge of "
    "alike neighbours) and EDITED (hard cuts plus gradual transitions, "
    "confirmed before/after, flashes, inserts and blank frames rejected, "
    "scenes down to scene_edited_min_duration_seconds). Earlier: "
    "step2_analyse/1.1.0: settled-state scene segmentation, frame quality "
    "scoring and the UNUSABLE_FOOTAGE flag, transcript segment filtering "
    "with REJECTED segments, OCR persistence/line-grouping/lexicon "
    "normalisation. Step 1 rows keep step1_2026_09_22_a."
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
STEP2_CONFIG_LINEAGE = ("step2_2026_09_22_c", "step2_2026_09_22_d",
                        "step2_2026_09_22_e", "step2_2026_09_23_a",
                        # 25 Sep: _a is what V16-V18 were written under; it
                        # was left out when the version was bumped, which is
                        # why a resumed run did not see them as done. _b is
                        # the transcript fix's config (four new keys).
                        "step2_2026_09_25_a", "step2_2026_09_25_b",
                        # 28 Sep: stage-only re-runs (quality, scenes, OCR),
                        # one version each; keyframes and frames unchanged.
                        "step2_2026_09_28_a", "step2_2026_09_28_b",
                        "step2_2026_09_28_c")
# Unified target (ml_test): its own Step 2 config versions. A resumed run must
# see assets written under them as done, or it rewrites them (26 Oct 2026: the
# run resumed after the iMac restarted tried every finished video again).
# _b: transcription_edit_classes gained RAW (every video with sound is
# transcribed) and the scene rule v2 (one scene RAW, more EDITED).
ML_TEST_STEP2_LINEAGE = ("ml_test_step2_2026_10_26_a", "ml_test_step2_2026_10_26_b",
                         # 5 Nov: the all-files video run writes under 11_03_a
                         # (videos.sh --config-version); without it every
                         # restart redid the whole list from the top.
                         "ml_test_step2_2026_11_03_a",
                         # 6 Nov (user: count them if processed correctly):
                         # 147 videos ran Step 2 under 10_29_a (29 Oct - 3 Nov)
                         # and were checked before this was added: scenes,
                         # keyframes, derivatives, transcript chunks and OCR
                         # present, keyframes and OCR linked to existing scenes.
                         "ml_test_step2_2026_10_29_a")

def config_lineage():
    """Config versions whose finished rows count as done for this target."""
    import target
    if target.is_unified():
        return STEP2_CONFIG_LINEAGE + ML_TEST_STEP2_LINEAGE
    return STEP2_CONFIG_LINEAGE


# analysis_runs.status has no PARTIAL. Anything incomplete is FAILED, with the
# error_code saying which kind of incomplete it was.
STATUS_OK = "COMPLETED"
STATUS_BAD = "FAILED"


class AssetFailed(Exception):
    """One asset could not be processed. Nothing is written for it."""


class SceneCapExceeded(AssetFailed):
    """Segmentation produced more scenes than the configured ceiling allows.

    Its own type so the failure carries its own error_code. "Too many scenes"
    is a different fact from "this asset blew up", and a reader of
    analysis_runs should not have to parse the message to tell them apart.
    """


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_db_config(conn):
    with conn.cursor() as cur:
        cur.execute("select key, value from content_v1.pipeline_config")
        return {k: v for k, v in cur.fetchall()}


def need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: {}".format(key))
    return cfg[key]


# The same fingerprint snapshot_config() computes, but read-only: sha256 over
# the pretty-printed {key: {value, value_type}} map of pipeline_config.
# Secret keys are left out, exactly as the ml_test snapshot_config() leaves
# them out of the payload (the unified pipeline_config also holds the DeepSeek
# queue's runner token). No content_v1 key matches the filter, so on content_v1
# this is the same fingerprint as before.
CURRENT_FINGERPRINT_SQL = """
select encode(extensions.digest(convert_to(jsonb_pretty(
         (select jsonb_object_agg(key,
                   jsonb_build_object('value', value, 'value_type', value_type))
            from content_v1.pipeline_config
           where key !~* '(^|_)(secret|api_key|password)$'
             and key not in ('runner_token', 'runner_url'))), 'UTF8'), 'sha256'), 'hex')
"""


# pipeline_config keys that only the Step 3 analyser / Step 4 embedder read. They change while
# Step 3 runs (brake_since after every hiccup, caps, prompt) and must not block Step 2.
STEP34_ONLY_KEYS_RE = (r"^(brake_since|stop_reason|run_enabled|test_only|spend_cap_usd|spend_window_hours"
                       r"|max_in_flight|max_scenes_per_call|jobs_per_invocation|calls_per_tick"
                       r"|call_timeout_seconds|stale_minutes|max_attempts|audit_rate|ref_pictures_max"
                       r"|prompt_version|system_prompt_.*|error_brake_.*|deep_.*|deepseek_.*|claude_.*"
                       r"|window_.*|embed_inline)$")


def step2_settings_unchanged(cur, version):
    """True when the config equals snapshot `version` in every key except STEP34_ONLY_KEYS_RE."""
    cur.execute("""
        select (select jsonb_object_agg(key, jsonb_build_object('value', value, 'value_type', value_type))
                  from content_v1.pipeline_config
                 where key !~* '(^|_)(secret|api_key|password)$'
                   and key not in ('runner_token', 'runner_url') and key !~ %(re)s)
             = (select jsonb_object_agg(e.key, e.value)
                  from content_v1.config_snapshots cs, jsonb_each(cs.payload) e
                 where cs.config_version = %(v)s
                   and e.key !~* '(^|_)(secret|api_key|password)$'
                   and e.key not in ('runner_token', 'runner_url') and e.key !~ %(re)s)""",
                {"re": STEP34_ONLY_KEYS_RE, "v": version})
    return bool(cur.fetchone()[0])


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
        cur.execute("select fingerprint from content_v1.config_snapshots "
                    "where config_version = %s", (version,))
        existing = cur.fetchone()
        if existing:
            if existing[0] != current and step2_settings_unchanged(cur, version):
                # 9 Nov 2026 (nightly run): only Step 3/4 keys differ (prompt, spend caps, the error
                # brake time the Step 3 supervisor resets after every hiccup). Step 2 ran with exactly
                # the snapshot's settings, so the version name still says what ran.
                return existing[0], "reused (Step 2 settings unchanged; only Step 3/4 keys differ)"
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
        cur.execute("select content_v1.snapshot_config(%s, %s)", (version, note))
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
       f.source_kind,
       f.root_path,
       f.discovery_status,
       m.duration_seconds,
       m.display_width_px, m.display_height_px,
       m.fps, m.has_audio,
       m.edit_class
from content_v1.assets a
join content_v1.asset_technical_metadata m on m.asset_id = a.id
join content_v1.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
join content_v1.source_folders f on f.id = sf.source_folder_id
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
            cur.execute("select count(*) from content_v1." + t)
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
    from content_v1.analysis_runs r
    where r.step like 'STEP2%%' and r.config_version = any(%s)
      -- 5 Nov 2026: a redo of an asset whose scenes were already written
      -- dies on the scenes unique key before writing anything, and the
      -- catch-all records that as a FAILED scenes attempt. That row says
      -- nothing about the stored result, so it does not count as status
      -- when a COMPLETED scenes attempt exists. 25 finished videos were
      -- shown as failed this way after restarts that did not skip them.
      and not (r.status = 'FAILED' and r.step = 'STEP2_SCENES'
               and r.metadata->>'recorded_by' = 'catch_all'
               -- 6 Nov: the same holds for any catch-all redo of an asset
               -- whose scenes had already COMPLETED before it (timeouts, the
               -- in-run pre-flight, a run stopped mid-asset): a failed asset
               -- writes nothing. Quality rejections (SCENE_CAP_EXCEEDED,
               -- WATCHDOG) still count. 30 videos were checked complete and
               -- linked before this was widened.
               and r.error_code = 'ASSET_FAILED'
               and exists (select 1 from content_v1.analysis_runs p
                           where p.asset_id = r.asset_id
                             and p.step = 'STEP2_SCENES'
                             and p.status = 'COMPLETED'
                             and p.created_at < r.created_at
                             and p.config_version = any(%s)))
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
                   -- A transcript is redone only by a stage-only re-run
                   -- (--stages transcript), never by a whole-asset retry. So
                   -- a transcript deliberately skipped (FAILED with
                   -- TRANSCRIPTION_SKIPPED) or knowingly PARTIAL counts as
                   -- done here; otherwise the asset's frames, scenes and OCR
                   -- would be written a second time.
                   bool_and(l.status = 'COMPLETED'
                            or (l.step = 'STEP2_TRANSCRIPT'
                                and (l.status = 'PARTIAL'
                                     or (l.status = 'FAILED'
                                         and l.error_code = 'TRANSCRIPTION_SKIPPED'))))
                     as all_ok,
                   min(sf.metadata->>'step1_item') as item,
                   min(m.media_kind) as media_kind
            from ({}) l
            join content_v1.asset_sources s
              on s.asset_id = l.asset_id and s.is_first_discovered_source
            join content_v1.source_files sf on sf.id = s.source_file_id
            join content_v1.asset_technical_metadata m on m.asset_id = l.asset_id
            group by l.asset_id
        """.format(LATEST_ATTEMPT_SQL), (list(config_lineage()),) * 2)
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

# Paths in this library look like 'LIM TJUN CHERN/2025.03.11/DSC03426.JPG':
# patient / session / file. The session folder is not always a date -
# 'FUE DAY-25.03.10' is a date with a stage on the front, and '3 MONTH' is a
# stage with no date in it at all.
#
# Two rules, applied INDEPENDENTLY, which is the correction that matters:
#
#   patient  the file's grandparent segment. Structural, so it does not
#            depend on the session folder being readable.
#   session  the file's parent segment, searched for a date and for a stage
#            label. Either may be absent without affecting the other.
#
# The old parser anchored on the date and took the segment before it, so a
# session folder it could not date yielded NO patient either. That is why
# 'DR MOHD MUZAFFAR MAN BIN MOHD MYDIN/3 MONTH/...' produced no patient name
# at all, and why brief 3.2 items 6-7 - telling two similar patients apart -
# could not be tested: one of the two patients was never identified.

# A date anywhere inside a segment, with a 4- or 2-digit year.
DATE_IN_SEGMENT_RE = re.compile(
    r"(?<![0-9])(\d{4}|\d{2})[.\-_](\d{1,2})[.\-_](\d{1,2})(?![0-9])")

# A 2-digit year is read as 20YY. The library's 4-digit dates are all 20xx,
# and 'FUE DAY-25.03.10' sits one day before the same patient's '2025.03.11',
# which is the reading that makes the two folders consistent. A year that
# would land in the future is refused rather than silently accepted.
TWO_DIGIT_YEAR_BASE = 2000


def parse_session_segment(segment):
    """Split a session folder into (date, stage, note).

    Returns date as an ISO string or None, stage as the non-date remainder or
    None. A segment that carries neither is neither - it is not an error."""
    seg = (segment or "").strip()
    if not seg:
        return None, None, "the session segment is empty"

    m = DATE_IN_SEGMENT_RE.search(seg)
    if not m:
        return None, seg, None

    y, mo, d = (int(x) for x in m.groups())
    raw_year = m.group(1)
    if len(raw_year) == 2:
        y += TWO_DIGIT_YEAR_BASE
    try:
        parsed = dt.date(y, mo, d)
    except ValueError:
        return None, seg, ("{!r} contains {!r}, which is shaped like a date "
                           "but is not one".format(seg, m.group(0)))
    if parsed > dt.date.today() + dt.timedelta(days=1):
        return None, seg, ("{!r} reads as {}, which is in the future; it is "
                           "recorded as a stage rather than guessed at"
                           .format(seg, parsed.isoformat()))

    stage = (seg[:m.start()] + " " + seg[m.end():]).strip(" -_.")
    stage = re.sub(r"\s+", " ", stage).strip(" -_.") or None
    return parsed.isoformat(), stage, None


def parse_relative_path(relative_path, parent_visible):
    """Returns a dict. Absence is recorded as a reason, never as a blank.

    Name and date are independent: a session folder that carries no date
    still yields the patient, because the patient is a different segment and
    was never the date's to withhold."""
    out = {"relative_path": relative_path,
           "patient_name": None, "session_date": None, "session_stage": None,
           "parse_status": None, "parse_note": None,
           "path_from_drive_scan": bool(parent_visible)}

    parts = [p for p in (relative_path or "").replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        out["parse_status"] = "NO_PATH_STRUCTURE"
        out["parse_note"] = ("the file sits at the root of its folder, so the "
                             "path carries no patient or session")
        return out

    notes = []

    # --- session: date and stage, each optional -----------------------
    session_date, stage, note = parse_session_segment(parts[-2])
    out["session_date"] = session_date
    out["session_stage"] = stage
    if note:
        notes.append(note)

    # --- patient: structural, independent of the above ----------------
    if len(parts) >= 3:
        out["patient_name"] = parts[-3].strip() or None
    if out["patient_name"] is None:
        notes.append("no segment precedes the session folder, so the path "
                     "carries no patient name")

    if out["patient_name"] and out["session_date"]:
        out["parse_status"] = "OK"
    elif out["patient_name"] and out["session_stage"]:
        # A stage is a real, usable fact about the session. Recording it as a
        # failure would throw away a patient name that parsed perfectly well.
        out["parse_status"] = "OK_STAGE_NO_DATE"
        notes.append("the session folder names a stage ({!r}), not a date, so "
                     "session_date is null and the stage is recorded instead"
                     .format(out["session_stage"]))
    elif out["patient_name"]:
        out["parse_status"] = "NO_SESSION_DETAIL"
        notes.append("the session folder carries neither a date nor a label")
    elif out["session_date"]:
        out["parse_status"] = "DATE_WITHOUT_PATIENT"
    else:
        out["parse_status"] = "NO_PATIENT_OR_DATE"

    if not parent_visible:
        notes.append(
            "PARTIAL: parent_visible is false for this file, so relative_path "
            "came from the manifest, not from a Drive scan. The parse is "
            "correct but the end-to-end case is not exercised until "
            "ALL PATIENT REVIEW is shared.")
    out["parse_note"] = "; ".join(notes) or None
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


def apply_scene_review(scenes, review, tol=0.05):
    """2.3.0: merge the groups of detected scenes a person marked as one scene.

    review = {"expected_scene_count": N, "merge_groups": [[i, j, ...], ...],
              "expected_starts": {"i": seconds, ...}, "reviewed_by", "date",
              "reason", ...}. Indices are the detector's 0-based scene indexes;
    each group is consecutive. The detection must be the one reviewed - same
    count, every grouped scene starting within tol seconds of what was seen -
    or the asset fails rather than merging the wrong scenes. A merged scene
    runs from its first member's start to its last member's end, so coverage
    is unchanged; no frame is touched (frames are linked by time later)."""
    n = int(review["expected_scene_count"])
    if len(scenes) != n:
        raise AssetFailed("scene review expects {} detected scenes, found {} - "
                          "not applied".format(n, len(scenes)))
    starts = {int(k): float(v) for k, v in (review.get("expected_starts") or {}).items()}
    groups = [sorted(int(i) for i in g) for g in review["merge_groups"]]
    seen = set()
    for g in groups:
        if len(g) < 2 or g != list(range(g[0], g[-1] + 1)) or seen & set(g) \
                or g[0] < 0 or g[-1] >= n:
            raise AssetFailed("scene review group {} is not a run of consecutive, "
                              "unshared scene indexes".format(g))
        seen |= set(g)
        for i in g:
            if i not in starts or abs(scenes[i]["start"] - starts[i]) > tol:
                raise AssetFailed(
                    "scene review: scene {} starts at {:.3f}s, reviewed at {} - "
                    "not applied".format(i, scenes[i]["start"], starts.get(i)))
    first = {g[0]: g for g in groups}
    out, merged = [], []
    i = 0
    while i < n:
        g = first.get(i)
        sc = dict(scenes[i])
        if g:
            sc["end"] = scenes[g[-1]]["end"]
            merged.append({"detected": g, "start": sc["start"], "end": sc["end"],
                           "removed_boundaries": [scenes[k]["start"] for k in g[1:]],
                           "note": (review.get("notes") or {}).get(str(g[0]))})
            i = g[-1] + 1
        else:
            i += 1
        sc["index"] = len(out)
        out.append(sc)
    for a, b in zip(out, out[1:]):
        if abs(a["end"] - b["start"]) > 1e-6:
            raise AssetFailed("scene review left a gap at {:.3f}s".format(a["end"]))
    record = {k: review.get(k) for k in ("method", "reviewed_by", "date", "reason")}
    record.update({"detected_scene_count": n, "scene_count_after": len(out),
                   "merged": merged, "processor": PROCESSOR_VERSION})
    return out, record


def check_scene_rate(scenes, duration, cfg, edit_path="RAW", consolidated=False):
    """Guard over-segmentation - the regression case was 42 scenes in 18.467s.

    The per-minute ceiling applies only above 60 seconds. Below that the
    arithmetic rejects correct detections: allowance is duration/5 seconds, so
    a 9.567s clip is capped at one scene and a 23.667s clip at four. V8 found
    2 and V6 found 5, both right, both rejected. Under 60s the bound is
    scene_min_duration_seconds, which the merge step has already enforced, so
    what reaches here cannot contain a scene shorter than that. The rate is
    computed and reported either way.

    The RATE is not a sufficient guard on its own, and V7 is the proof: 14
    scenes in 189.267s is 4.44/min, comfortably inside a ceiling of 12, and
    the run recorded COMPLETED while being thirteen scenes wrong. A rate
    ceiling cannot bind on long footage, because length buys headroom. So
    scene_max_count_absolute_per_asset is checked first and at every duration.
    It existed in config and in step2_scenes.check_ceiling(), but nothing on
    the live path ever read it - only step2_tune.py did, which does not write
    analysis_runs."""
    per_min = round(len(scenes) / (duration / 60.0), 2) if duration else 0.0
    # EDITED footage is cut by an editor: short shots and many of them are
    # normal there (V16 is 12 scenes in 39 s, V17 opens with a 0.97 s shot),
    # so it has its own floor and ceiling. RAW keeps the original ones, which
    # are what stop a continuous take being chopped up.
    edited = (edit_path == "EDITED")
    ceiling = float(need(cfg, "scene_edited_max_count_per_minute" if edited
                         else "scene_max_count_per_minute"))
    min_dur = float(need(cfg, "scene_edited_min_duration_seconds" if edited
                         else "scene_min_duration_seconds"))
    absolute = int(need(cfg, "scene_edited_max_count_absolute_per_asset" if edited
                        else "scene_max_count_absolute_per_asset"))

    # 2.8.0: step2_scenes_v2 1.5.0 has already grouped an over-cap EDITED
    # video's shots into moments (scene_edited_consolidate_*). The groups are
    # each at least scene_edited_consolidate_min_seconds long, so the count may
    # pass the absolute cap; the per-minute ceiling below still applies.
    if len(scenes) > absolute and not consolidated:
        raise SceneCapExceeded(
            "{} scenes exceeds scene_max_count_absolute_per_asset={} "
            "({:.2f}/min over {:.3f}s)".format(len(scenes), absolute,
                                               per_min, duration))

    if duration <= SCENE_RATE_MIN_DURATION_SECONDS:
        shortest = min((s["end"] - s["start"] for s in scenes), default=0.0)
        if len(scenes) > 1 and shortest < min_dur - 1e-9:
            raise AssetFailed(
                "shortest scene {:.3f}s is under scene_min_duration_seconds="
                "{} ({} scenes in {:.3f}s)".format(shortest, min_dur,
                                                   len(scenes), duration))
        return per_min

    if len(scenes) > 1 and per_min > ceiling:
        raise SceneCapExceeded(
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
        import mlx_whisper_engine as MW
        if MW.selected():   # STEP2_WHISPER_ENGINE=mlx: Whisper on the GPU (2 Nov 2026)
            _WHISPER["m"] = MW.MLXWhisperModel(need(cfg, "transcription_model"))
            return _WHISPER["m"]
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
    """Speech, via step2_transcribe - which is where the rules live.

    This function used to hold its own inline implementation, and that
    implementation read six config keys and ignored eleven. Every safeguard
    the operator had configured - the language allowlist, the quality gates,
    the repetition limits, the reject share, condition_on_previous_text -
    existed in pipeline_config and in step2_transcribe.py, and nothing on the
    live path ever called any of it. That is how 3.7 seconds of a camera
    pointed at clothing came to be stored as fluent French, COMPLETE.

    So the live path now calls the module. The returned dict keeps the shape
    the writer expects, with per-chunk statuses added.
    """
    import step2_transcribe as T

    raw = T.transcribe_asset(sys.modules[__name__], video_path, tmp_dir, cfg,
                             duration, True)
    res = T.assemble(raw, duration, cfg)
    status, error_code, message = T.run_status(res, cfg)
    return {
        "status": res["outcome"],
        "run_status": status,
        "error_code": error_code,
        "error_message": message,
        "language": res.get("language"),
        "language_probability": res.get("language_probability"),
        "language_choice": res.get("language_choice"),
        "vocabulary_applied": res.get("vocabulary_applied"),
        "kept_segments": res.get("kept_segments"),
        "excluded_segments": res.get("excluded_segments"),
        "excluded_real_speech": res.get("excluded_real_speech"),
        "speech_ratio": res.get("speech_ratio"),
        "clamped_chunks": res.get("clamped_chunks"),
        "chunks": res["chunks"],
        "reason": res.get("reason"),
        "span_retries": res.get("span_retries") or [],
        "language_overridden": res.get("language_overridden"),
        "language_decode_scores": res.get("language_decode_scores"),
        "loop_lost_segments": res.get("loop_lost_segments"),
        "loop_lost_seconds": res.get("loop_lost_seconds"),
    }


def speech(asset, video_path, tmp_dir, cfg, duration, deadline=None):
    """The transcript for one video, whichever way it is obtained. Shared by
    process_video and the stage-only re-run so the two cannot drift apart."""
    # Policy (2026-09-25): only edited videos are transcribed. A RAW take is
    # recorded as fully processed - COMPLETED, one EXCLUDED chunk saying why -
    # because not transcribing it is the intended outcome, not a failure.
    allowed = [str(c).upper() for c in need(cfg, "transcription_edit_classes")]
    exception = (cfg.get(TRANSCRIBE_RAW_KEY) or {}).get(asset.get("item"))
    if exception and str(asset.get("edit_class") or "").upper() not in allowed:
        # 2.5.0: a recorded per-file exception (--transcribe-raw); the policy
        # itself and transcription_edit_classes are unchanged
        tr = speech(dict(asset, edit_class=allowed[0]), video_path, tmp_dir,
                    {k: v for k, v in cfg.items() if k != TRANSCRIBE_RAW_KEY},
                    duration, deadline)
        tr["policy_exception"] = {"reason": exception,
                                  "edit_class": asset.get("edit_class"),
                                  "policy": "raw footage: transcription not required "
                                            "(policy 2026-09-25)"}
        return tr
    if str(asset.get("edit_class") or "").upper() not in allowed:
        why = "raw footage: transcription not required (policy 2026-09-25)"
        return {"status": "EXCLUDED", "language": None,
                "language_probability": None, "run_status": "COMPLETED",
                "chunks": [{"start": 0.0, "end": round(duration, 3), "text": "",
                            "language": None, "confidence": None,
                            "transcription_status": "EXCLUDED",
                            "failure_reason": why}],
                "reason": why, "policy_skip": True,
                "edit_class": asset.get("edit_class")}
    skip_reason = cfg.get(SKIP_TRANSCRIPTION_KEY)
    if skip_reason:
        # The run row is FAILED with error_code TRANSCRIPTION_SKIPPED: the
        # step did not happen, and analysis_runs has no status for
        # "deferred". completed_assets() still counts it as done.
        why = "skipped for this run by --skip-transcription: {}".format(
            skip_reason)
        return {"status": "EXCLUDED", "language": None,
                "language_probability": None,
                "chunks": [{"start": 0.0, "end": round(duration, 3), "text": "",
                            "language": None, "confidence": None,
                            "transcription_status": "EXCLUDED",
                            "failure_reason": why}],
                "reason": why, "error_code": "TRANSCRIPTION_SKIPPED",
                "skipped_by_flag": True, "skip_reason": skip_reason}
    if bool(need(cfg, "transcription_enabled")) and asset["has_audio"]:
        if deadline is not None:
            deadline.open("transcribe", fixed_budget(cfg))
        # 2.7.0: a recorded per-file language (--transcript-language) and a
        # recorded per-file exception to the no_speech_prob checks (--ignore-no-speech)
        forced = (cfg.get(TRANSCRIPT_LANGUAGE_KEY) or {}).get(asset.get("item"))
        no_speech = (cfg.get(IGNORE_NO_SPEECH_KEY) or {}).get(asset.get("item"))
        if no_speech:
            limits = {k: cfg.get(k) for k in ("transcript_no_speech_prob_max",
                                              "transcript_no_speech_prob_corroborating")}
            tr = speech(asset, video_path, tmp_dir,
                        dict({k: v for k, v in cfg.items() if k != IGNORE_NO_SPEECH_KEY},
                             transcript_no_speech_prob_max=1.0,
                             transcript_no_speech_prob_corroborating=1.0),
                        duration)
            tr["no_speech_check_override"] = {
                "reason": no_speech, "flag": "--ignore-no-speech (step2_analyse 2.7.0)",
                "limits_in_config": limits, "limits_used": 1.0}
            return tr
        if forced:
            tr = transcribe(video_path, tmp_dir, dict(cfg, transcription_language=forced[0]),
                            duration)
            tr["language_choice"] = dict(
                tr.get("language_choice") or {}, forced_by="--transcript-language",
                forced_reason=forced[1],
                note="language forced to {!r} for this file by --transcript-language "
                     "(step2_analyse 2.7.0): {}".format(forced[0], forced[1]))
            return tr
        return transcribe(video_path, tmp_dir, cfg, duration)
    why = ("no audio stream" if not asset["has_audio"]
           else "transcription_enabled is false")
    # transcription_status is NOT NULL: this chunk could never have been
    # written without it.
    return {"status": "EXCLUDED", "language": None,
            "language_probability": None,
            "run_status": "COMPLETED",
            "chunks": [{"start": 0.0, "end": round(duration, 3), "text": "",
                        "language": None, "confidence": None,
                        "transcription_status": "EXCLUDED",
                        "failure_reason": why}],
            "reason": why}


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


def classify_text(bbox, frame_w, frame_h, normalized_text=None, cfg=None,
                  lexicon=None):
    """Deterministic classification from two sources, neither of which reads
    the picture.

    GEOMETRY gives CAPTION. A burnt-in caption is a wide block parked in one
    of the frame's safe bands - top or bottom. The original rule looked only
    at the bottom, which is why it never fired once: the caption in V9
    ('POV: THE PATIENT FORGOT', 0.61 of the frame wide) sits at centre_y 0.15,
    in the TOP band. Measured, not assumed.

    A LEXICON gives DEVICE_LABEL and UI. Whether a reading names a device is
    not something geometry can answer - ULTRAFORMER sits at centre_y 0.68,
    LASEMD at 0.37 and HELIOS at 0.10, so there is no shape to find. It is
    also not something that needs the picture interpreted: it needs a list of
    the devices this clinic owns, matched exactly. That list is reference
    data, not inference, and lives outside the code.

    The match is EXACT on normalized text. The same footage produced FIRMX,
    IRM, FRM and FR alongside SYLFIRM; fuzzy-matching those would be guessing
    at what the camera nearly saw, so they stay OTHER. A garbled reading is
    not a confirmed device name.

    SIGNAGE and DOCUMENT are left unassigned: nothing in this data
    distinguishes a sign on a wall or a page of text from any other reading
    without interpreting the picture, which Rule 2 forbids.
    """
    lexicon = lexicon or {}
    if normalized_text:
        key = " ".join((normalized_text or "").upper().split())
        for text_type, terms in lexicon.items():
            if key in terms:
                return text_type

    try:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
    except Exception:
        return "OTHER"
    if not frame_h or not frame_w:
        return "OTHER"
    centre_y = (min(ys) + max(ys)) / 2.0
    width = max(xs) - min(xs)

    min_w = float(_cfg_or(cfg, "ocr_caption_min_width_fraction", 0.40))
    top_band = float(_cfg_or(cfg, "ocr_caption_top_band_fraction", 0.25))
    bottom_band = float(_cfg_or(cfg, "ocr_caption_bottom_band_fraction", 0.70))
    if width > min_w * frame_w and (centre_y < top_band * frame_h
                                    or centre_y > bottom_band * frame_h):
        return "CAPTION"
    return "OTHER"


def _cfg_or(cfg, key, default):
    """Config when there is config, the documented default when there is not.
    Reclassification runs without a full config in some call paths."""
    try:
        return cfg[key] if cfg and key in cfg else default
    except TypeError:
        return default


# ocr_observations.text_type's allowed values.
TEXT_TYPES = ("DEVICE_LABEL", "CAPTION", "SIGNAGE", "DOCUMENT", "UI", "OTHER")


def build_lexicon(cfg, treatments=None):
    """{text_type: set(normalized terms)}.

    Prefers the treatments reference table when it holds rows, because brief
    section 6 makes that the authority on what this clinic treats with. Falls
    back to pipeline_config's operator-curated list while the reference data
    is still missing, so the classifier is not blocked on a copy that needs
    credentials nobody has handed over yet.
    """
    out = {}
    # ocr_known_terms holds the whole OCR vocabulary by group. Only groups
    # named after a text type classify; CLINIC_NAME is vocabulary, not a type.
    raw = need(cfg, "ocr_known_terms") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    for text_type, terms in raw.items():
        if text_type not in TEXT_TYPES:
            continue
        out[text_type] = {" ".join(str(t).upper().split()) for t in terms}
    for name in (treatments or []):
        key = " ".join(str(name).upper().split())
        if key:
            out.setdefault("DEVICE_LABEL", set()).add(key)
    return out


def ocr_frame(path, reader, cfg, frame_w, frame_h, lexicon=None):
    floor = float(need(cfg, "ocr_confidence_floor"))
    if lexicon is None:
        lexicon = build_lexicon(cfg)
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
            "text_type": classify_text(
                bbox, frame_w, frame_h,
                normalized_text=" ".join(raw.upper().split()),
                cfg=cfg, lexicon=lexicon),
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
#     V11    15 frames    54.0s   3.60 s/frame
#     V10    11 frames    44.0s   4.00 s/frame
#     V14    13 frames    60.7s   4.67 s/frame
#     V7    190 frames   907.8s   4.78 s/frame
#     V13    12 frames    60.0s   5.00 s/frame
#     V3     25 frames   146.7s   5.87 s/frame
#     V9     12 frames    72.0s   6.00 s/frame
#     V4     81 frames   495.1s   6.11 s/frame
#     V15    25 frames   176.7s   7.07 s/frame
#     V6     24 frames   173.1s   7.21 s/frame
#     V2     25 frames   183.3s   7.33 s/frame
#
# These are the OCR STAGE alone, separated from download, extraction and
# transcription by scripts/step2_stage_timings.py. Dividing an asset's total
# runtime by its frame count instead gives 5.4-8.3 s/frame, which charges OCR
# for work it did not do; the tighter range above is the honest one.
#
# The budget has to sit above the WORST of those, not the average, or the
# fix turns every slow-but-healthy asset into a degraded one. At 12 s/frame
# the closest any video comes to its ceiling is V2, at 50.9%.

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
# and surfaced by the content_v1.asset_ocr_coverage view, so "no text" and
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


# Which measured stages each analysis_runs row is accountable for. The rows
# are per STEP, the budgets are per STAGE, and they are not the same carving:
# STEP2_SCENES is the product of getting the bytes, cutting the frames and
# finding the boundaries, so its window spans all three.
STEP_STAGES = {
    "STEP2_SCENES": ("download", "extract_frames", "scenes"),
    "STEP2_TRANSCRIPT": ("transcribe",),
    "STEP2_OCR": ("ocr", "ocr_sampled"),
}


def stage_windows(timings):
    """Real (started_at, completed_at) per STEP2 step, from the worker's own
    per-stage clock.

    Every row used to carry the moment the ASSET started, which made the five
    rows indistinguishable and their durations nested rather than sequential:
    STEP2_SCENES appeared to have taken 1019s on IMG_1147 when 1019s was the
    whole measurement phase and cut detection was a small part of it. A
    budget cannot be audited against a number that means something else.
    """
    by_stage = {}
    for t in timings or []:
        if t.get("completed_at") is None:
            continue
        by_stage.setdefault(t["stage"], []).append(t)
    out = {}
    for step, stages in STEP_STAGES.items():
        spans = [t for s in stages for t in by_stage.get(s, [])]
        if spans:
            out[step] = (min(t["started_at"] for t in spans),
                         max(t["completed_at"] for t in spans))
    return out


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
        out["ocr_engine"] = got.get("ocr_engine")
        out["ocr_summary"] = got.get("ocr_summary")
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


def is_local(asset):
    """True when the original sits on a mounted share (source_kind LOCAL, e.g.
    the NAS) rather than in Drive."""
    return asset.get("source_kind") == "LOCAL"


def fetch_original(s1, get_service, asset, dest):
    """Put the original's bytes at dest. LOCAL: a verbatim copy from
    root_path + relative_path (the share is only read). Drive: the retried
    download, as before. get_service() is called only for Drive assets, so a
    NAS-only run never needs Drive credentials. Returns the attempt count."""
    if is_local(asset):
        import os
        import shutil
        import target
        if dest.exists():
            dest.unlink()
        cached = target.cached_original_path(asset["content_hash"], asset["file_name"])
        if target.keep_originals() and not cached.exists():
            # Fill the cache once: copy from the share, verify, then keep.
            try:
                src = s1.local_original_path(asset["root_path"], asset["relative_path"])
            except s1.FileFailed as exc:
                raise AssetFailed("original not readable on the share: {}".format(exc))
            cached.parent.mkdir(parents=True, exist_ok=True)
            part = cached.with_name(cached.name + ".part")
            shutil.copyfile(src, part)
            if sha256_file(part) != asset["content_hash"]:
                part.unlink()
                raise AssetFailed("copy from the share does not match Step 1's content_hash")
            os.replace(part, cached)
        if target.keep_originals():
            # A hard link: no second copy on disk, and deleting the working
            # file afterwards leaves the cached one in place.
            try:
                os.link(cached, dest)
            except OSError:
                shutil.copyfile(cached, dest)
            return 1
        try:
            src = s1.local_original_path(asset["root_path"], asset["relative_path"])
        except s1.FileFailed as exc:
            raise AssetFailed("original not readable on the share: {}".format(exc))
        shutil.copyfile(src, dest)
        return 1
    if not asset.get("google_file_id"):
        raise AssetFailed("no google_file_id and not a LOCAL source: the "
                          "original's bytes cannot be retrieved")
    return download_with_retry(s1, get_service(), asset["google_file_id"], dest)


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
    # Segmentation runs on its own full-frame-rate decode (step2_scenes_v2),
    # not on the 1-per-second stored frames: an editor's dissolve lasts 0.2 to
    # 0.4 s and falls between two stored frames. The path is chosen by Step 1's
    # edit_class - RAW and EDITED footage fail in opposite directions.
    mode = need(cfg, "scene_segmentation_mode")
    if mode != "edit_class_v2":
        raise SystemExit(
            "scene_segmentation_mode={!r}: this version of step2_analyse "
            "implements only 'edit_class_v2'.".format(mode))
    import step2_scenes_v2 as SV2
    deadline.open("scenes", fixed_budget(cfg))
    try:
        seg = SV2.segment(video_path, duration, asset.get("edit_class"), cfg)
    except SV2.SceneError as exc:
        raise AssetFailed("scene segmentation failed: {}".format(exc))
    scenes = seg["scenes"]
    review = (cfg.get(SCENE_REVIEW_KEY) or {}).get(asset.get("item"))
    if review:
        scenes, review_record = apply_scene_review(scenes, review)
    else:
        review_record = None
    cuts = [t for t, src, _s in seg["boundaries"] if src in ("cut", "transition")]
    drift = [t for t, src, _s in seg["boundaries"] if src == "drift"]
    raw_thr = float(need(cfg, "scene_raw_drift_threshold"))
    drift_diag = [(t, d, "above" if d > raw_thr else "below")
                  for t, d in seg["diag"]]
    coverage, problems = verify_scene_coverage(scenes, duration)
    rate = check_scene_rate(scenes, duration, cfg, seg["path"],
                            bool(seg.get("consolidated")))
    # Evidence for the report when nothing was found: the largest ffmpeg
    # scene scores in the file. Diagnostic only - never places a boundary.
    score_profile = scene_score_profile(video_path) if len(scenes) == 1 else []

    # --- speech -------------------------------------------------------
    tr = speech(asset, video_path, tmp_dir, cfg, duration, deadline)

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
                  "scene_score_profile": score_profile,
                  "scene_path": seg["path"],
                  "scene_rejected": seg["rejected"],
                  "scene_cut_check": seg.get("cut_check"),
                  "scene_review": review_record,
                  "scene_consolidated": seg.get("consolidated") or None}
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


# --------------------------------------------------------------------------
# on-screen text: the STEP2_OCR 3.0.0 engine and rules, for every path
# --------------------------------------------------------------------------
# One implementation for the full run, the OCR-only salvage, stills and the
# stage re-run: ocr_engine (RapidOCR PP-OCRv6, native + crop-upscale
# variants) and step2_ocr (coverage rule, known terms, dictionary
# correction, every-word garbage gate, text-block merge). The older inline
# easyocr reader (ocr_reader/ocr_frame above) is kept only for the lab
# scripts that call it; no pipeline path does.

OCR_REQUIRED_ENGINE = "rapidocr"
# Runtime-only (like SKIP_TRANSCRIPTION_KEY): the treatments/aliases reference
# data step2_ocr needs, loaded once by the parent and carried in the in-memory
# cfg to the media worker. Never a pipeline_config key.
OCR_REFERENCE_KEY = "_run_ocr_reference"
_OCR3 = {}


def ocr3_engine(cfg):
    """(engine callable, {"provider", "model"}) - the engine that actually
    reads, so the rows can say so. Refuses anything but the 3.0.0 choice:
    easyocr is not a fallback."""
    import ocr_engine as E
    name = need(cfg, "ocr_engine")
    if name != OCR_REQUIRED_ENGINE:
        raise SystemExit(
            "ocr_engine={!r}: on-screen text is read only with the STEP2_OCR "
            "3.0.0 engine {!r} (chosen by benchmark); there is no fallback."
            .format(name, OCR_REQUIRED_ENGINE))
    if "engine" not in _OCR3:
        _OCR3["engine"] = E.make_engine(cfg)
        _OCR3["info"] = {"provider": "apple_vision" if E.apple_vision_selected() else name,
                         "model": E.engine_version(cfg)}
    return _OCR3["engine"], dict(_OCR3["info"])


def load_ocr_reference(conn):
    """treatments and aliases, the same reference score_ocr reads."""
    with conn.cursor() as cur:
        cur.execute("select name from test_v1.treatments where is_active")
        treatments = [r[0] for r in cur.fetchall()]
        cur.execute("select alias from test_v1.treatment_aliases")
        aliases = [r[0] for r in cur.fetchall()]
    conn.rollback()
    return {"treatments": treatments, "aliases": aliases}


def load_ocr_expected(conn):
    """{asset_id: expected_onscreen_text or None} from answer_key.

    content_v1 has no answer_key table: there is no expected text for any
    content_v1 asset, which is exactly what an asset without an answer_key
    row gets in test_v1."""
    return {}


def _ocr_reference(cfg):
    ref = cfg.get(OCR_REFERENCE_KEY)
    if ref is None:
        raise SystemExit("OCR reference data was not loaded into cfg "
                         "(OCR_REFERENCE_KEY); call load_ocr_reference first")
    return ref.get("treatments") or [], ref.get("aliases") or []


def ocr3_change_scores(images_in_order):
    """step2_ocr_run's change score for each frame vs the one before, from an
    iterator of decoded frames - one frame in memory at a time."""
    import ocr_engine as E
    scores, prev = [], None
    for img in images_in_order:
        scores.append(E.change_score(prev, img))
        prev = img
    return scores


def ocr3_read(items, load_image, cfg, per_frame, deadline=None, stage="ocr"):
    """Read `items` ([{"index","timestamp",...}]) in order, appending
    {"index","timestamp","variants"} to per_frame as each finishes, so a
    budget overrun keeps what was already read. Returns the frame size."""
    import ocr_engine as E
    engine, _info = ocr3_engine(cfg)
    size = None
    for n, it in enumerate(items, 1):
        if deadline is not None:
            deadline.check(stage)
        img = load_image(it)
        size = size or {"width": int(img.shape[1]), "height": int(img.shape[0])}
        per_frame.append({"index": it["index"], "timestamp": it["timestamp"],
                          "scene_id": it.get("scene_id"),
                          "variants": E.read_frame(img, engine, cfg)})
        if deadline is not None:
            deadline.heartbeat(stage, n, len(items))
    return size


def ocr3_build(per_frame, size, cfg, usable=True):
    """step2_ocr.build_observations over what was read - rows plus summary."""
    import step2_ocr as O
    treatments, aliases = _ocr_reference(cfg)
    if not per_frame or not size:
        return [], {"accepted": 0, "lines": 0, "frames_ocrd_count": 0}
    per_frame = sorted(per_frame, key=lambda f: f["index"])
    return O.build_observations(per_frame, size, cfg,
                                build_lexicon(cfg, treatments), usable=usable,
                                treatments=treatments, aliases=aliases)


def ocr3_result(observations, coverage, cfg, expected=None):
    """The STEP2_OCR status from the score, as step2_ocr_run sets it.

    expected is answer_key.expected_onscreen_text, or None where the key has
    no opinion - then only the garbage check can fail the run. A run whose
    coverage is degraded is PARTIAL (OCR_DEGRADED) even if what it read scored
    clean: not everything was read."""
    import score_ocr as S
    import step2_ocr as O
    treatments, aliases = _ocr_reference(cfg)
    terms = O.known_terms(cfg, treatments, aliases)
    words = O.load_words(cfg)
    accepted = [o for o in observations if o["accepted_for_search"]]
    sc = S.score(expected, accepted, terms, words, cfg)
    status, code, message = sc["status"], sc["error_code"], sc["message"]
    if coverage.get("ocr_coverage") != OCR_FULL and status == "COMPLETED":
        status, code = "PARTIAL", "OCR_DEGRADED"
        message = coverage.get("ocr_degraded_reason") or "coverage {}".format(
            coverage.get("ocr_coverage"))
    return {"status": status, "error_code": code, "message": message,
            "score": {k: sc[k] for k in (
                "expected", "found", "missed", "recall", "strict_recall",
                "accepted", "precision", "garbage", "mean_confidence",
                "false_positives")}}


def _per_frame_accepted(observations):
    by = {}
    for o in observations:
        if o["accepted_for_search"]:
            by.setdefault(o["_frame_index"], set()).add(o["normalized_text"])
    return [frozenset(v) for _k, v in sorted(by.items())]


def read_on_screen_text(frames, scenes, size, cfg, deadline, unusable=()):
    """The 3.0.0 coverage rule over the frames on disk, then the 3.0.0 rules.

    If the budget runs out, the representative frames not yet read are read
    under a second budget and the run says SAMPLED. Frames already read are
    kept - they are paid for."""
    import cv2
    import step2_ocr as O
    _engine, info = ocr3_engine(cfg)
    usable_frames = [fr for fr in frames
                     if not any(a - 1e-9 <= fr["timestamp"] < b
                                for a, b in unusable)]
    load = (lambda fr: cv2.imread(str(fr["path"]), cv2.IMREAD_COLOR))
    scores = ocr3_change_scores(load(fr) for fr in usable_frames)
    chosen = [usable_frames[i] for i in O.coverage_frames(
        [fr["timestamp"] for fr in usable_frames], cfg, scores)]

    per_frame, frame_size = [], None
    t0 = time.time()
    coverage, reason = OCR_FULL, None
    deadline.open("ocr", ocr_budget(len(chosen) or 1, cfg))
    try:
        frame_size = ocr3_read(chosen, load, cfg, per_frame, deadline, "ocr")
    except StageOverBudget as exc:
        coverage, reason = OCR_SAMPLED, str(exc)
        done = {f["index"] for f in per_frame}
        reps = representative_frames(frames, scenes,
                                     need(cfg, "ocr_sampled_max_frames"))
        todo = [fr for fr in reps if fr["index"] not in done]
        deadline.open("ocr_sampled", ocr_budget(len(todo) or 1, cfg))
        try:
            frame_size = ocr3_read(todo, load, cfg, per_frame, deadline,
                                   "ocr_sampled") or frame_size
        except StageOverBudget as exc2:
            reason = "{}; the sampled retry also ran out: {}".format(
                str(exc), exc2)
    if frame_size is None and per_frame:
        img = load(frames[0])
        frame_size = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    rows, summary = ocr3_build(per_frame, frame_size, cfg)
    if not per_frame:
        coverage = OCR_NONE
    return {"observations": rows, "ocr_seconds": time.time() - t0,
            "ocr_frames_total": len(frames),
            "ocr_frames_selected": len(chosen),
            "ocr_frames_processed": len(per_frame),
            "ocr_coverage": coverage, "ocr_degraded_reason": reason,
            "ocr_engine": info, "ocr_summary": summary,
            "near_duplicate": near_duplicate_rate(_per_frame_accepted(rows))}


def ocr3_read_paths(frames, cfg, deadline=None, stage="ocr_sampled"):
    """The OCR-only salvage: named frames already on disk, 3.0.0 rules."""
    import cv2
    _engine, info = ocr3_engine(cfg)
    per_frame = []
    load = (lambda fr: cv2.imread(str(fr["path"]), cv2.IMREAD_COLOR))
    try:
        size = ocr3_read(frames, load, cfg, per_frame, deadline, stage)
    except StageOverBudget:
        size = None
        if per_frame:
            img = load(frames[0])
            size = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    rows, summary = ocr3_build(per_frame, size, cfg)
    return {"observations": rows, "ocr_frames_processed": len(per_frame),
            "ocr_engine": info, "ocr_summary": summary}


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
    engine_info, summary = None, None
    coverage, reason, processed = OCR_NONE, "ocr_enabled is false", 0
    if do_ocr and bool(need(cfg, "ocr_enabled")):
        # The 3.0.0 engine and rules, as for video. A still is one frame, so
        # it cannot persist across frames: ocr_min_consecutive_frames is 1
        # for a still (in memory only; the video rule is unchanged).
        import cv2
        still_cfg = dict(cfg, ocr_min_consecutive_frames=1)
        _engine, engine_info = ocr3_engine(still_cfg)
        per_frame = []
        fsize = ocr3_read([{"index": 0, "timestamp": 0.0}],
                          lambda _it: cv2.imread(str(small), cv2.IMREAD_COLOR),
                          still_cfg, per_frame)
        observations, summary = ocr3_build(per_frame, fsize, still_cfg)
        for ob in observations:
            ob["timestamp"] = None      # a still has no time; stored as before
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
            "ocr_coverage": coverage, "ocr_degraded_reason": reason,
            "ocr_engine": engine_info, "ocr_summary": summary}


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
                      help="write to content_v1 and upload keyframes")
    ap.add_argument("--only", default=None,
                    help="comma-separated items (V1,V7) for a narrower run")
    ap.add_argument("--skip-transcription", metavar="REASON", default=None,
                    help="this run only: do not run Whisper; every video's "
                         "transcript is written as EXCLUDED with REASON. "
                         "pipeline_config is not changed.")
    ap.add_argument("--config-version", default=STEP2_CONFIG_VERSION,
                    help="config_version stamped on this run's rows and "
                         "snapshot; a new name per re-run (default %(default)s)")
    ap.add_argument("--stages", default=None, metavar="STAGE[,STAGE]",
                    help="stage-only re-run on assets Step 2 has already "
                         "processed: any of 'scenes', 'transcript', 'ocr'. Replaces only "
                         "that stage's rows (backed up first, one transaction "
                         "per asset per stage) with a new analysis_runs attempt; "
                         "never re-extracts frames or uploads images.")
    ap.add_argument("--scene-review", default=None, metavar="FILE",
                    help="2.3.0: JSON {step1 item: hand-checked merge list}; merges "
                         "those detected scenes and records the review on the "
                         "STEP2_SCENES run. Full runs, or --stages scenes (2.6.0).")
    ap.add_argument("--transcribe-raw", action="append", default=[],
                    metavar="ITEM=REASON",
                    help="2.5.0: transcribe this RAW video despite the raw-footage "
                         "policy; REASON is recorded on the STEP2_TRANSCRIPT run "
                         "(metadata.policy_exception). Repeatable.")
    ap.add_argument("--transcript-language", action="append", default=[],
                    metavar="ITEM=LANG:REASON",
                    help="2.7.0: force the transcript language for this file (LANG in "
                         "transcription_allowed_languages); REASON is recorded on the "
                         "STEP2_TRANSCRIPT run (metadata.language_choice). Repeatable.")
    ap.add_argument("--ignore-no-speech", action="append", default=[],
                    metavar="ITEM=REASON",
                    help="2.7.0: turn off only the no_speech_prob checks for this file; "
                         "REASON is recorded on the STEP2_TRANSCRIPT run "
                         "(metadata.no_speech_check_override). Repeatable.")
    args = ap.parse_args()
    ignore_no_speech = {}
    for spec in args.ignore_no_speech:
        item, _, reason = spec.partition("=")
        if not item.strip() or not reason.strip():
            raise SystemExit("--ignore-no-speech takes ITEM=REASON; got {!r}".format(spec))
        ignore_no_speech[item.strip().upper()] = reason.strip()
    transcript_language = {}
    for spec in args.transcript_language:
        item, _, rest = spec.partition("=")
        lang, _, reason = rest.partition(":")
        if not item.strip() or not lang.strip() or not reason.strip():
            raise SystemExit("--transcript-language takes ITEM=LANG:REASON; got {!r}".format(spec))
        transcript_language[item.strip().upper()] = (lang.strip().lower(), reason.strip())
    transcribe_raw = {}
    for spec in args.transcribe_raw:
        item, _, reason = spec.partition("=")
        if not item.strip() or not reason.strip():
            raise SystemExit("--transcribe-raw takes ITEM=REASON; got {!r}".format(spec))
        transcribe_raw[item.strip().upper()] = reason.strip()
    if args.scene_review and args.stages is not None and "scenes" not in args.stages.lower():
        raise SystemExit("--scene-review needs a full run or --stages scenes")
    committing = bool(args.commit)
    if args.skip_transcription is not None and not args.skip_transcription.strip():
        raise SystemExit("--skip-transcription needs a reason")
    stages = None
    if args.stages is not None:
        stages = {s.strip().lower() for s in args.stages.split(",") if s.strip()}
        unknown = stages - set(STAGE_CHOICES)
        if not stages or unknown:
            raise SystemExit("--stages takes {}; got {!r}".format(
                " and/or ".join(STAGE_CHOICES), args.stages))

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
    print("config:     {} keys from content_v1.pipeline_config + {} local-only"
          .format(len(cfg) - len(s1.LOCAL_ONLY_CONFIG_KEYS),
                  len(s1.LOCAL_ONLY_CONFIG_KEYS)))
    handle.conn.rollback()
    # STEP2_OCR 3.0.0 needs the treatments reference in the worker too, and
    # scores against answer_key where one exists (None = no opinion).
    cfg[OCR_REFERENCE_KEY] = load_ocr_reference(handle.ensure())
    ocr_expected = load_ocr_expected(handle.ensure())
    if args.skip_transcription:
        # A per-run flag, not config: it rides in the in-memory cfg so the
        # media worker sees it, and is never written to pipeline_config.
        cfg[SKIP_TRANSCRIPTION_KEY] = args.skip_transcription.strip()
        print("transcribe: SKIPPED for this run - {}".format(
            cfg[SKIP_TRANSCRIPTION_KEY]))
    if args.scene_review:
        # 2.3.0: per-run, like --skip-transcription; never written to pipeline_config
        reviews = json.loads(Path(args.scene_review).read_text(encoding="utf-8"))
        cfg[SCENE_REVIEW_KEY] = {k: v for k, v in reviews.items() if not k.startswith("_")}
        print("scene review: {} for {}".format(args.scene_review,
                                               ", ".join(sorted(cfg[SCENE_REVIEW_KEY]))))
    if transcript_language:
        allowed_langs = [str(x).lower() for x in need(cfg, "transcription_allowed_languages")]
        for item, (lang, reason) in sorted(transcript_language.items()):
            if lang not in allowed_langs:
                raise SystemExit("--transcript-language {}: {!r} is not in "
                                 "transcription_allowed_languages {}".format(item, lang, allowed_langs))
            print("transcript language: {} forced to {} - {}".format(item, lang, reason))
        cfg[TRANSCRIPT_LANGUAGE_KEY] = transcript_language
    if ignore_no_speech:
        for item, reason in sorted(ignore_no_speech.items()):
            print("no-speech checks off: {} - {}".format(item, reason))
        cfg[IGNORE_NO_SPEECH_KEY] = ignore_no_speech
    if transcribe_raw:
        cfg[TRANSCRIBE_RAW_KEY] = transcribe_raw
        for item, reason in sorted(transcribe_raw.items()):
            print("transcribe raw: {} - {}".format(item, reason))

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
    config_version = args.config_version
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

    if stages:
        return run_stages(stages, videos, images, handle, pf, cfg,
                          config_version, committing, store, baseline,
                          ocr_expected)

    # The parent no longer touches Drive or ffmpeg or the models: everything
    # that can hang happens in the worker, where it can be killed. The service
    # is still built once here so the report can name the account and so a bad
    # credential fails before the first asset rather than inside a child.
    if any(not is_local(a) for a in videos + images):
        service, sa_email = s1.build_drive_service()
        del service
    else:
        sa_email = "n/a (every original is on a LOCAL share)"

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
                    "scene_path": out.get("scene_path"),
                    "scene_rejected": out.get("scene_rejected") or [],
                    "scene_review": out.get("scene_review"),
                    "scene_consolidated": out.get("scene_consolidated"),
                    "edit_class": asset.get("edit_class"),
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
                    # Search text only: an EXCLUDED chunk (a loop, noise) is
                    # kept in the table for review but is not the transcript.
                    "transcript_text": " ".join(
                        c["text"] for c in out["transcript"]["chunks"]
                        if c.get("transcription_status") == "COMPLETE").strip(),
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
                    # Each row is stamped with the window its OWN work
                    # occupied, not with the moment the asset began.
                    win = stage_windows(out.get("stage_timings"))
                    sc_from, sc_to = win.get("STEP2_SCENES", (started, None))
                    writer.write_scenes(asset, out["scenes"], out["coverage"],
                                        out["coverage_problems"], sc_from,
                                        sc_to or dt.datetime.now(dt.timezone.utc),
                                        path_parse,
                                        stage_timings=out.get("stage_timings"),
                                        extra_meta={"scene_review": out["scene_review"]}
                                        if out.get("scene_review") else None)
                    if out["coverage_problems"]:
                        raise AssetFailed("scene coverage incomplete: {}".format(
                            "; ".join(out["coverage_problems"])))
                    # The upload is the parent's own work and no stage covers
                    # it, so it is timed here rather than inherited.
                    up_from = dt.datetime.now(dt.timezone.utc)
                    uploads = writer.upload_frames(asset, out["frames"])
                    up = writer.write_keyframes(asset, out["frames"],
                                                out["scenes"], uploads,
                                                up_from,
                                                dt.datetime.now(dt.timezone.utc))
                    rec["stored"] = up["stored"]
                    rec["derivatives"] = up["derivatives"]
                    if up["failed"]:
                        rec["status"] = "PARTIAL"
                        rec["reason"] = "{} frame(s) not stored".format(
                            len(up["failed"]))
                    tr_from, tr_to = win.get(
                        "STEP2_TRANSCRIPT",
                        (started, dt.datetime.now(dt.timezone.utc)))
                    writer.write_transcript(asset, out["transcript"], tr_from,
                                            tr_to)
                    for ob in out["observations"]:
                        fr = out["frames"][ob["_frame_index"]]
                        ob["keyframe_id"] = fr.get("keyframe_id")
                        ob["scene_id"] = fr.get("scene_id")
                    ocr_from, ocr_to = win.get(
                        "STEP2_OCR",
                        (started, dt.datetime.now(dt.timezone.utc)))
                    writer.write_ocr(asset, out["observations"], ocr_from,
                                     ocr_to, path_parse, coverage=out,
                                     result=ocr3_result(
                                         out["observations"], out, cfg,
                                         ocr_expected.get(asset["asset_id"])))
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
                    win = stage_windows(out.get("stage_timings"))
                    o_from, o_to = win.get(
                        "STEP2_OCR",
                        (started, dt.datetime.now(dt.timezone.utc)))
                    writer.write_ocr(asset, out["observations"], o_from, o_to,
                                     path_parse, coverage=out,
                                     stage_timings=out.get("stage_timings"),
                                     result=ocr3_result(
                                         out["observations"], out, cfg,
                                         ocr_expected.get(asset["asset_id"])))

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
                if isinstance(exc, SceneCapExceeded):
                    failure_code = "SCENE_CAP_EXCEEDED"
                elif "watchdog" in rec["reason"]:
                    failure_code = "WATCHDOG"
                else:
                    failure_code = "ASSET_FAILED"
                ok, why = writer.write_failure(
                    asset, "STEP2_SCENES", rec["reason"], started,
                    dt.datetime.now(dt.timezone.utc),
                    error_code=failure_code)
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
    if os.environ.get("STEP2_SHARED_WRITERS") == "1":
        # runs side by side share temp_root: sweep only this run's own item
        # folders, never the frames another run is still reading (3 Nov 2026:
        # the whole-folder sweep at the end of each chunk deleted them)
        leftovers = [p for a in videos + images
                     for p in (temp_root / "step2_{}".format(a["item"])).rglob("*")
                     if p.is_file()]
    else:
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
    if committing and writer is not None:
        import writer as step1_writer
        step1_writer.close_batch_row(
            handle.ensure(), writer.batch_id, len(results),
            sum(1 for r in results if r["status"] == "OK"), len(failures))

    write_report(results, cfg, config_version, committing, sa_email, store)
    path = REPORT_COMMIT if committing else REPORT_DRY
    print("wrote {}".format(path))
    handle.close()
    if handle.reconnects:
        print("database reconnects during the run: {}".format(
            handle.reconnects))
    return 1 if failures else 0


# --------------------------------------------------------------------------
# stage-only re-runs (--stages transcript|scenes)
# --------------------------------------------------------------------------

STAGE_CHOICES = ("scenes", "transcript", "ocr")   # this order when several are asked:
# OCR last, so its rows take the scene links a scenes re-run just wrote


def stage_state(conn, asset_id):
    """What an asset's transcript and scenes look like in the database now:
    the 'before' of a stage re-run."""
    with conn.cursor() as cur:
        cur.execute("""
            select status, error_code, metadata->>'outcome',
                   metadata->>'language', attempt_number, config_version
            from content_v1.analysis_runs
            where asset_id = %s and step = 'STEP2_TRANSCRIPT'
            order by attempt_number desc, created_at desc limit 1
        """, (asset_id,))
        run = cur.fetchone()
        cur.execute("""
            select start_seconds, end_seconds, transcript_text, language,
                   transcription_status
            from content_v1.asset_transcript_chunks
            where asset_id = %s order by start_seconds
        """, (asset_id,))
        chunks = cur.fetchall()
        cur.execute("select start_seconds, end_seconds from content_v1.asset_scenes "
                    "where asset_id = %s order by scene_index", (asset_id,))
        scenes = cur.fetchall()
        cur.execute("select count(*) from content_v1.asset_keyframes "
                    "where asset_id = %s", (asset_id,))
        keyframes = cur.fetchone()[0]
    conn.rollback()
    langs = sorted({c[3] for c in chunks
                    if c[4] == "COMPLETE" and c[3]}) or [None]
    return {
        "run_status": run[0] if run else None,
        "run_error_code": run[1] if run else None,
        "outcome": run[2] if run else None,
        "language": (run[3] if run and run[3] else None) or "/".join(
            x for x in langs if x) or None,
        "chunks": len(chunks),
        "text": " ".join(c[2] for c in chunks
                         if c[4] == "COMPLETE" and c[2]).strip(),
        "scenes": [(float(a), float(b)) for a, b in scenes],
        "keyframes": keyframes,
    }


def _transcript_status(tr):
    status = tr.get("run_status") or ("COMPLETED" if tr["status"] in (
        "COMPLETE", "EMPTY_NO_SPEECH") else "FAILED")
    return status, tr.get("error_code")


def _what_changed(before, after):
    notes = []
    if (before["language"] or None) != (after["language"] or None):
        notes.append("language {} -> {}".format(before["language"],
                                                after["language"]))
    b_st = "{}/{}".format(before["run_status"], before["outcome"])
    a_st = "{}/{}".format(after["run_status"], after["outcome"])
    if b_st != a_st:
        notes.append("status {} -> {}".format(b_st, a_st))
    if before["text"] == after["text"]:
        notes.append("text identical")
    else:
        notes.append("text changed")
    return notes


def stage_ocr_read(asset, cfg, conn, store):
    """Re-read one video's on-screen text from its STORED keyframes with the
    STEP2_OCR 3.0.0 engine and rules. Nothing is extracted or uploaded: each
    keyframe is fetched from the bucket, checked against its derivative's
    sha256 and decoded in memory. Frames inside a stored quality span are
    skipped, as step2_ocr_run does."""
    import hashlib
    import cv2
    import numpy as np
    import step2_ocr as O
    aid = asset["asset_id"]
    with conn.cursor() as cur:
        cur.execute("""
            select k.id, k.scene_id, k.timestamp_seconds, d.storage_path,
                   d.content_hash
            from content_v1.asset_keyframes k
            join content_v1.asset_derivatives d on d.id = k.derivative_id
            where k.asset_id = %s order by k.timestamp_seconds
        """, (aid,))
        kfs = [{"keyframe_id": r[0], "scene_id": r[1], "t": float(r[2]),
                "path": r[3], "sha": r[4]} for r in cur.fetchall()]
        cur.execute("select start_seconds, end_seconds "
                    "from content_v1.asset_quality_spans where asset_id = %s",
                    (aid,))
        spans = [(float(a), float(b)) for a, b in cur.fetchall()]
    conn.rollback()
    if not kfs:
        raise AssetFailed("no stored keyframes to read")
    usable = [f for f in kfs
              if not any(a - 1e-9 <= f["t"] < b for a, b in spans)]
    data = {}

    def fetch(f):
        key = f["path"]
        if key not in data:
            b = store.get(key)
            if hashlib.sha256(b).hexdigest() != f["sha"]:
                raise AssetFailed("keyframe {} does not match its stored "
                                  "sha256".format(key))
            data[key] = b
        return cv2.imdecode(np.frombuffer(data[key], np.uint8),
                            cv2.IMREAD_COLOR)

    scores = ocr3_change_scores(fetch(f) for f in usable)
    idx = O.coverage_frames([f["t"] for f in usable], cfg, scores)
    chosen = [dict(usable[i], index=i, timestamp=usable[i]["t"]) for i in idx]
    _engine, info = ocr3_engine(cfg)
    per_frame = []
    size = ocr3_read(chosen, fetch, cfg, per_frame)
    rows, summary = ocr3_build(per_frame, size, cfg)
    for r in rows:
        f = usable[r["_frame_index"]]
        r["keyframe_id"], r["scene_id"] = f["keyframe_id"], f["scene_id"]
    coverage = {"ocr_frames_total": len(kfs), "ocr_frames_selected": len(chosen),
                "ocr_frames_processed": len(per_frame),
                "ocr_frames_skipped_unusable": len(kfs) - len(usable),
                "ocr_coverage": OCR_FULL if per_frame else OCR_NONE,
                "ocr_degraded_reason": None, "ocr_engine": info,
                "ocr_summary": summary}
    return rows, coverage


def stored_ocr(conn, asset_id):
    with conn.cursor() as cur:
        cur.execute("""
            select raw_text, normalized_text, confidence, timestamp_seconds,
                   accepted_for_search, provider, model, config_version
            from content_v1.ocr_observations where asset_id = %s
            order by timestamp_seconds, normalized_text
        """, (asset_id,))
        rows = [{"raw_text": r[0], "normalized_text": r[1],
                 "confidence": float(r[2]) if r[2] is not None else 0.0,
                 "timestamp": float(r[3]) if r[3] is not None else None,
                 "accepted_for_search": r[4], "provider": r[5], "model": r[6],
                 "config_version": r[7]} for r in cur.fetchall()]
    conn.rollback()
    return rows


def ocr_review(rows, result, expected):
    """What the report needs about one set of OCR rows: accepted readings,
    the ones no expected line accounts for, and garbled ones - the scorer's
    'not real text' (which catches run-together words: they are not words)
    plus single letters and two-character fragments, which it lets through."""
    def norm(s):
        return " ".join((s or "").upper().split())
    exp = [norm(e) for e in (expected or [])]
    acc = [r for r in rows if r["accepted_for_search"]]
    garbage = set((result.get("score") or {}).get("garbage") or [])
    by_text = {}
    for r in acc:
        by_text.setdefault(r["normalized_text"], []).append(r)
    unexpected, garbled = [], []
    for text, rs in sorted(by_text.items()):
        times = sorted({r["timestamp"] for r in rs if r["timestamp"] is not None})
        conf = round(max(r["confidence"] for r in rs), 3)
        n = norm(text)
        if not any(n and (n in e or e in n) for e in exp):
            unexpected.append({"text": text, "times": times, "max_conf": conf})
        if text in garbage or (len(n.replace(" ", "")) <= 2
                               and not n.replace(".", "").isdigit()):
            garbled.append({"text": text, "times": times, "max_conf": conf,
                            "why": "not real text (scorer)" if text in garbage
                            else "single letter / fragment"})
    return {"accepted": len(acc), "distinct_accepted": len(by_text),
            "unexpected": unexpected, "garbled": garbled}


def run_stages(stages, videos, images, handle, pf, cfg, config_version,
               committing, store, baseline, ocr_expected=None):
    """Replace one or both of a video's re-computable stages.

    Scenes need only the video and step2_scenes_v2; the transcript needs only
    the audio. Neither needs frames, so none are extracted and no image is
    uploaded: keyframes, derivatives and their Storage objects stay exactly
    as they are, and OCR is only re-linked to the new scenes. Runs in this
    process - no worker - because nothing here is the kind of stage that
    hangs (frame extraction and OCR)."""
    import step1_ingest as s1
    import step2_scenes_v2 as SV2
    from step2_writer import Step2Writer

    order = [s for s in STAGE_CHOICES if s in stages]
    if images:
        print("stages:     {} image(s) ignored - a still has no scenes or "
              "transcript".format(len(images)))
    print("stages:     {} ({}) on {} video(s)".format(
        ", ".join(order), "COMMIT" if committing else "dry run", len(videos)))
    drive = {}

    def get_service():
        # Only Drive originals need the service; a NAS-only run never builds it.
        if "s" not in drive:
            drive["s"], drive["sa"] = s1.build_drive_service()
        return drive["s"]
    temp_root = PROJECT_ROOT / need(cfg, "temp_directory")
    temp_root.mkdir(parents=True, exist_ok=True)
    day = dt.datetime.now().date()

    writer = None
    if committing:
        writer = Step2Writer(handle, pf, cfg, config_version, store)
        writer.baseline = dict(baseline)
        writer.backup_suffix = BACKUP_SUFFIX
    ocr_store = None
    if "ocr" in order:
        from storage import KeyframeStore
        ocr_store = store or KeyframeStore(need(cfg, "keyframe_storage_bucket"),
                                           verify_readback=False)
    ocr_expected = ocr_expected or {}
    needs_media = bool({"scenes", "transcript"} & set(order))

    results = []
    for asset in videos:
        item = asset["item"]
        before = stage_state(handle.ensure(), asset["asset_id"])
        rec = {"item": item, "file_name": asset["file_name"],
               "duration": float(asset["duration_seconds"]),
               "before": before, "status": "OK", "reason": None}
        print("[{}] {} ...".format(item, asset["file_name"]), flush=True)
        if before["keyframes"] == 0 or not before["scenes"]:
            rec.update(status="REFUSED",
                       reason="no Step 2 rows yet - run the full pipeline first")
            results.append(rec)
            print("    REFUSED  {}".format(rec["reason"]), flush=True)
            continue
        tmp_dir = temp_root / "stage_{}".format(item)
        rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        media = tmp_dir / asset["file_name"]
        duration = float(asset["duration_seconds"])
        try:
            if needs_media:
                fetch_original(s1, get_service, asset, media)
                if sha256_file(media) != asset["content_hash"]:
                    raise AssetFailed("downloaded bytes do not match Step 1's "
                                      "content_hash")

            if "scenes" in order:
                t0 = dt.datetime.now(dt.timezone.utc)
                seg = SV2.segment(media, duration, asset.get("edit_class"), cfg)
                scenes = seg["scenes"]
                # 2.6.0: a manual scene review applies to a scenes re-run too
                review = (cfg.get(SCENE_REVIEW_KEY) or {}).get(asset.get("item"))
                review_record = None
                if review:
                    scenes, review_record = apply_scene_review(scenes, review)
                    rec["scene_review"] = review_record
                coverage, problems = verify_scene_coverage(scenes, duration)
                check_scene_rate(scenes, duration, cfg, seg["path"],
                                 bool(seg.get("consolidated")))
                if problems:
                    raise AssetFailed("scene coverage incomplete: {}".format(
                        "; ".join(problems)))
                rec["scenes_after"] = [(s["start"], s["end"]) for s in scenes]
                rec["scene_path"] = seg["path"]
                if committing:
                    rec["scenes_written"] = writer.replace_scenes(
                        asset, scenes, coverage, t0,
                        dt.datetime.now(dt.timezone.utc), day,
                        extra_meta={"scene_path": seg["path"],
                                    "scene_rejected": seg["rejected"][:100],
                                    "cut_check": seg.get("cut_check"),
                                    **({"scene_review": review_record}
                                       if review_record else {}),
                                    **({"scene_consolidated": seg["consolidated"]}
                                       if seg.get("consolidated") else {})})
                print("    scenes {} -> {}".format(len(before["scenes"]),
                                                   len(scenes)), flush=True)

            if "transcript" in order:
                t0 = dt.datetime.now(dt.timezone.utc)
                tr = speech(asset, media, tmp_dir, cfg, duration)
                status, code = _transcript_status(tr)
                after = {
                    "run_status": status, "run_error_code": code,
                    "outcome": tr["status"], "language": tr.get("language"),
                    "chunks": len(tr["chunks"]),
                    "text": " ".join(c["text"] for c in tr["chunks"]
                                     if c.get("transcription_status")
                                     == "COMPLETE" and c["text"]).strip()}
                rec["transcript_after"] = after
                rec["changes"] = _what_changed(before, after)
                rec["transcript_detail"] = {
                    k: tr.get(k) for k in (
                        "reason", "error_message", "vocabulary_applied",
                        "language_choice", "language_overridden",
                        "no_speech_check_override",
                        "language_decode_scores", "span_retries",
                        "loop_lost_segments", "loop_lost_seconds",
                        "kept_segments", "excluded_segments",
                        "excluded_real_speech", "speech_ratio")}
                rec["transcript_chunks"] = [
                    {"start": c["start"], "end": c["end"],
                     "status": c.get("transcription_status"),
                     "text": c["text"], "reason": c.get("failure_reason")}
                    for c in tr["chunks"]]
                if committing:
                    rec["transcript_written"] = writer.replace_transcript(
                        asset, tr, t0, dt.datetime.now(dt.timezone.utc), day)
                print("    transcript {} {}/{} -> {} {}/{}  ({})".format(
                    before["language"], before["run_status"], before["outcome"],
                    after["language"], after["run_status"], after["outcome"],
                    "; ".join(rec["changes"])), flush=True)

            if "ocr" in order:
                t0 = dt.datetime.now(dt.timezone.utc)
                exp = ocr_expected.get(asset["asset_id"])
                before_rows = stored_ocr(handle.ensure(), asset["asset_id"])
                before_res = ocr3_result(before_rows, {"ocr_coverage": OCR_FULL},
                                         cfg, exp)
                rows, cov = stage_ocr_read(asset, cfg, handle.ensure(),
                                           ocr_store)
                after_res = ocr3_result(rows, cov, cfg, exp)
                rec["ocr"] = {
                    "expected": exp,
                    "before": dict(ocr_review(before_rows, before_res, exp),
                                   status=before_res["status"],
                                   error_code=before_res["error_code"],
                                   score=before_res["score"],
                                   provider=sorted({r["provider"] or "-" for r in before_rows}),
                                   model=sorted({r["model"] or "-" for r in before_rows})),
                    "after": dict(ocr_review(rows, after_res, exp),
                                  status=after_res["status"],
                                  error_code=after_res["error_code"],
                                  message=after_res["message"],
                                  score=after_res["score"],
                                  engine=cov["ocr_engine"],
                                  frames=[cov["ocr_frames_total"],
                                          cov["ocr_frames_selected"],
                                          cov["ocr_frames_processed"],
                                          cov["ocr_frames_skipped_unusable"]],
                                  rejected_by=(cov["ocr_summary"] or {}).get("rejected_by")),
                    "accepted_rows": [
                        {"t": r["timestamp"], "text": r["normalized_text"],
                         "raw": r["raw_text"], "conf": r["confidence"],
                         "type": r["text_type"]}
                        for r in rows if r["accepted_for_search"]],
                }
                if committing:
                    rec["ocr_written"] = writer.replace_ocr(
                        asset, rows, t0, dt.datetime.now(dt.timezone.utc), day,
                        cov, after_res)
                b, a = rec["ocr"]["before"], rec["ocr"]["after"]
                print("    ocr {} accepted ({}) -> {} accepted ({}); found {}/{}; "
                      "garbled {} -> {}".format(
                          b["accepted"], b["status"], a["accepted"], a["status"],
                          len(a["score"]["found"]), a["score"]["expected"],
                          len(b["garbled"]), len(a["garbled"])), flush=True)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            rec["status"] = "FAILED"
            rec["reason"] = "{}: {}".format(type(exc).__name__, exc)
            print("    FAILED  {}".format(rec["reason"][:300]), flush=True)
        finally:
            rmtree(tmp_dir)
        results.append(rec)

    handle.close()
    sa_email = drive.get("sa", "n/a (every original is on a LOCAL share)")
    path = write_stage_report(results, order, cfg, config_version, committing,
                              sa_email)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print("")
    print("stage re-run: {} ok, {} failed, {} refused".format(
        n_ok, sum(1 for r in results if r["status"] == "FAILED"),
        sum(1 for r in results if r["status"] == "REFUSED")))
    print("wrote {}".format(path))
    return 0 if n_ok == len(results) else 1


def write_stage_report(results, order, cfg, config_version, committing,
                       sa_email):
    tag = "_".join(order)
    path = PROJECT_ROOT / "reports" / "content_v1" / "step2_stage_{}_{}.md".format(
        tag, "commit" if committing else "dryrun")
    L = ["# Step 2 stage re-run ({}) - {}".format(
            ", ".join(order), "commit" if committing
            else "dry run (nothing was written)"), "",
         "- Config version: `{}`   processor: `{}`".format(
             config_version, PROCESSOR_VERSION),
         "- Service account: `{}`".format(sa_email)]
    if cfg.get(SKIP_TRANSCRIPTION_KEY):
        L.append("- **Transcription skipped** (--skip-transcription): {}".format(
            cfg[SKIP_TRANSCRIPTION_KEY]))
    L.append("")
    if "transcript" in order:
        L += ["## Transcript: before / after", "",
              "| Item | File | Before lang | Before status | Before text (120) "
              "| After lang | After status | After text (120) | What changed |",
              "|---|---|---|---|---|---|---|---|---|"]

        def cell(t):
            return (t or "").replace("|", "\\|").replace("\n", " ")[:120]
        for r in results:
            b, a = r["before"], r.get("transcript_after")
            if a is None:
                L.append("| {} | `{}` | {} | {}/{} | {} | - | {} | - | {} |".format(
                    r["item"], r["file_name"], b["language"], b["run_status"],
                    b["outcome"], cell(b["text"]), r["status"],
                    cell(r.get("reason"))))
                continue
            L.append("| {} | `{}` | {} | {}/{} | {} | {} | {}/{}{} | {} | {} |"
                     .format(r["item"], r["file_name"], b["language"],
                             b["run_status"], b["outcome"], cell(b["text"]),
                             a["language"], a["run_status"], a["outcome"],
                             " [{}]".format(a["run_error_code"])
                             if a["run_error_code"] else "",
                             cell(a["text"]), "; ".join(r.get("changes") or [])))
        L.append("")
        L.append("## Transcript detail")
        L.append("")
        for r in results:
            d = r.get("transcript_detail")
            if not d:
                continue
            ch = d.get("language_choice") or {}
            L.append("- **{}** `{}` - detector {} (raw {}), chosen {}{}; "
                     "decode scores {}; vocabulary {}; kept {} excluded {} "
                     "(real speech {}, lost to loop {} / {}s)".format(
                         r["item"], r["file_name"],
                         ch.get("detector_chosen") or ch.get("detected"),
                         ch.get("detected_probability"),
                         ch.get("chosen"),
                         " (OVERRIDDEN)" if d.get("language_overridden") else "",
                         json.dumps(d.get("language_decode_scores")),
                         d.get("vocabulary_applied"), d.get("kept_segments"),
                         d.get("excluded_segments"),
                         d.get("excluded_real_speech"),
                         d.get("loop_lost_segments"), d.get("loop_lost_seconds")))
            for sr in d.get("span_retries") or []:
                L.append("  - span retry {} {}-{}s: kept {} (loop {}s -> {}s): "
                         "{!r} -> {!r}".format(
                             sr["language"], sr["span"][0], sr["span"][1],
                             sr["kept"], sr["loop_seconds_before"],
                             sr["loop_seconds_after"], sr["text_before"][:80],
                             sr["text_after"][:80]))
            for c in r.get("transcript_chunks") or []:
                if c["status"] != "COMPLETE":
                    L.append("  - {} {:.2f}-{:.2f}s {!r}: {}".format(
                        c["status"], c["start"], c["end"], c["text"][:80],
                        c["reason"]))
        L.append("")
    if "scenes" in order:
        L += ["## Scenes: before / after", "",
              "| Item | File | Before | After | Path |", "|---|---|---:|---:|---|"]
        for r in results:
            L.append("| {} | `{}` | {} | {} | {} |".format(
                r["item"], r["file_name"], len(r["before"]["scenes"]),
                len(r.get("scenes_after") or []) if "scenes_after" in r
                else r["status"], r.get("scene_path") or "-"))
        L.append("")
    if "ocr" in order:
        def times(ts):
            return ", ".join("{:g}".format(t) for t in ts[:12]) + (
                " ..." if len(ts) > 12 else "")
        L += ["## OCR: before / after", "",
              "| Item | File | Accepted before | Status before | Accepted after "
              "| Status after | Expected | Found | Missed | Unexpected (distinct) "
              "| Garbled before -> after |",
              "|---|---|---:|---|---:|---|---:|---:|---|---:|---|"]
        for r in results:
            o = r.get("ocr")
            if not o:
                L.append("| {} | `{}` | | | | {} | | | | | |".format(
                    r["item"], r["file_name"], r["status"]))
                continue
            b, a = o["before"], o["after"]
            L.append("| {} | `{}` | {} | {} | {} | {}{} | {} | {} | {} | {} | {} -> {} |"
                     .format(r["item"], r["file_name"], b["accepted"], b["status"],
                             a["accepted"], a["status"],
                             " [{}]".format(a["error_code"]) if a["error_code"] else "",
                             a["score"]["expected"], len(a["score"]["found"]),
                             "; ".join(a["score"]["missed"]) or "-",
                             len(a["unexpected"]), len(b["garbled"]),
                             len(a["garbled"])))
        L.append("")
        L.append("## OCR detail")
        L.append("")
        for r in results:
            o = r.get("ocr")
            if not o:
                continue
            b, a = o["before"], o["after"]
            acc_texts = {x["text"] for x in o["accepted_rows"]}
            L.append("- **{}** `{}` - engine before {} / {}; after {} (frames "
                     "total/selected/read/skipped-unusable {}); rejected by {}"
                     .format(r["item"], r["file_name"], b["provider"], b["model"],
                             a["engine"].get("model"), a["frames"],
                             a["rejected_by"]))
            for term in ("LUTRONIC", "CLASSYS"):
                if term in [str(e).upper() for e in (o["expected"] or [])]:
                    hit = sorted({x["t"] for x in o["accepted_rows"]
                                  if x["text"] == term})
                    L.append("  - {} accepted: {}{}".format(
                        term, "YES" if hit else "NO",
                        " at {} s".format(times(hit)) if hit else ""))
            if a["score"]["missed"]:
                L.append("  - missed: {}".format("; ".join(a["score"]["missed"])))
            for u in a["unexpected"]:
                L.append("  - accepted, not expected: {!r} at {} s (max conf {})"
                         .format(u["text"], times(u["times"]), u["max_conf"]))
            for g in a["garbled"]:
                L.append("  - garbled ({}): {!r} at {} s (max conf {})".format(
                    g["why"], g["text"], times(g["times"]), g["max_conf"]))
        L.append("")
    fails = [r for r in results if r["status"] != "OK"]
    if fails:
        L += ["## Not processed", ""]
        L += ["- {} {}: {}".format(r["item"], r["status"], r["reason"])
              for r in fails]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(results, indent=1, default=str, ensure_ascii=False),
        encoding="utf-8")
    return path


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
    if cfg.get(SKIP_TRANSCRIPTION_KEY):
        L.append("- **Transcription skipped for this run** (--skip-transcription): "
                 "{}. Transcripts are EXCLUDED; pipeline_config unchanged."
                 .format(cfg[SKIP_TRANSCRIPTION_KEY]))
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
        if r.get("scene_path"):
            L.append("  - scene path: {} (Step 1 edit_class={})".format(
                r["scene_path"], r.get("edit_class")))
        if r.get("raw_cuts"):
            L.append("  - cut/transition boundaries: {}".format(r["raw_cuts"]))
        for t, why in (r.get("scene_rejected") or []):
            L.append("  - rejected {:.3f}s: {}".format(t, why))
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
