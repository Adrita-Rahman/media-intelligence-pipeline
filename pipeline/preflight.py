#!/usr/bin/env python3
"""
Pre-flight assertions for every write into content_v1.

Purpose: a write must never land in a database that is not the intended one,
and never on top of state we did not expect. On 22 Sep 2026 a session read
project-a (ref OTHER_PROJECT_REF_1) while believing it was reading
the test project. Nothing was written. This module is the guard that makes
that class of mistake impossible rather than lucky.

The guard is a fingerprint, not a name. A project ref can be passed in wrongly;
these five facts are true of the intended database and of no other:

  1. content_v1.config_snapshots has config_version 'step1_2026_09_22_a'
     whose fingerprint starts 38ff26cf
  2. content_v1.pipeline_config has exactly 74 rows and contains
     'detect_format_from_content'
  3. content_v1.source_files has a 'parent_visible' column
  4. every table in REQUIRED_TABLES exists in content_v1, and nothing else
     does except dated *_backup_YYYYMMDD copies
  5. content_v1.assets is at the expected row count for this point in the run

For contrast, the database that was read by mistake has 28 tables, 22 assets,
35 config keys and no parent_visible column. Every one of the five fails there.

Checks 1-4 are invariant for a given config generation, not for all time. When
the config legitimately changes - four watchdog keys were added on 22 Sep 2026
- the expected count is updated here deliberately, which is the point: drift
that nobody intended still stops a run.

Check 5 is the only one that changes during a run. Checks 1-4 are invariant and
are re-asserted before EVERY write, per the standing instruction. Check 5 is
phase-aware, because 'assets is empty' cannot remain true once this run has
legitimately begun inserting assets:

    PHASE_PRE_RUN  -> assets must be exactly 0
    PHASE_IN_RUN   -> assets must equal the number of rows this run has
                      written so far, and no more. A row that appeared from
                      anywhere else is an unexpected-state failure.

Usage:
    from preflight import Preflight, PHASE_PRE_RUN, PHASE_IN_RUN
    pf = Preflight(fetch_one)          # fetch_one(sql) -> dict
    pf.assert_ready(PHASE_PRE_RUN)     # gate before the run starts
    ...
    pf.assert_ready(PHASE_IN_RUN, assets_written=n)   # before each write

Standalone (read-only, writes nothing):
    python scripts/preflight.py
"""
from __future__ import annotations

import os
import re

import target

# --------------------------------------------------------------------------
# The intended target. Chosen by PIPELINE_TARGET (target.py), never passed in
# by a caller, so a caller still cannot quietly aim this at another project.
# content_v1 (the default) is project-b / content_v1; ml_test is
# project-a / ml_test, which refuses project-b instead.
# SQL below is written against content_v1 and rewritten to the target schema
# by db.py; the constants here are what the messages and checks compare to.
# --------------------------------------------------------------------------
EXPECTED_PROJECT_REF = target.profile()["project_ref"]
EXPECTED_PROJECT_NAME = {"content_v1": "project-b",
                         "ml_test": "project-a"}[target.name()]
EXPECTED_SCHEMA = target.schema()

EXPECTED_SNAPSHOT = "step1_2026_09_22_a"
EXPECTED_FINGERPRINT_PREFIX = "38ff26cf"
# 41 -> 45 -> 49 on 22 Sep 2026, then 84 on 23 Sep for 1.1.0: the watchdog
# budget keys,
# (ocr_stage_base_seconds, ocr_stage_seconds_per_frame, stage_ceiling_seconds,
# ocr_sampled_max_frames) were added so the OCR ceiling could scale with frame
# count, then four more for text_type classification (the device lexicon and
# the caption bands). The count is asserted, not ignored, precisely so that
# config drifting by accident still stops a run - a deliberate change updates
# this line.
# 74 became 122 across the 2.0.0 fixes: 11 transcription keys so the language
# allowlist, the quality gates and the repetition rule could actually be read,
# 23 quality_* keys for keyframe scoring, 13 ocr_* keys for vocabulary
# matching and acceptance, and the validator's own thresholds. The count is
# asserted, not ignored, so config drifting by accident still stops a run.
# 128 -> 135 on 24 Sep: the seven capture_* keys of the capture-date rule
# (scripts/capture_date.py) - clinic hours, synced devices, offset detection
# residual, plausibility offsets, stored clock corrections, uncertain ranges
# and folder-date authority.
# 135 -> 151 on 24 Sep for STEP2_OCR 3.0.0: 21 ocr_* keys added (engine,
# models, benchmark record, one vocabulary, crop-upscale, coverage, units,
# device context, dictionary correction, line/block merge, variant
# overlap) and 5 removed (the two old vocabulary keys, the digit-count
# bare-number rule, the caption-only block gap, the IoU pooling key).
# 151 -> 172 on 25 Sep for step2_analyse 2.1.0: the 21 scene_v2 keys
# (scene_edited_*, scene_raw_*, scene_blank_*, scene_signature_px,
# scene_unclassified_mode) added by step2_scene_fix/scene_config_v2.sql.
# 172 -> 173 on 25 Sep for scene fix v2: scene_edited_gradual_contrast added.
# 173 -> 177 on 25 Sep for the transcript fix: transcript_language_override_margin,
# transcript_word_run_min, transcript_word_share_max,
# transcript_word_share_min_count (scripts/transcript_fix_config.sql).
# 177 -> 176 on 25 Sep: transcription_edit_classes added (RAW footage is not
# transcribed), the two transcript_word_share_* keys dropped (the check fired
# on real English). Backup: pipeline_config_backup_20260925.
# 176 -> 179 on 28 Sep for the V16 scene fix: scene_edited_flash_max_seconds,
# scene_edited_flash_min_distance, scene_edited_flash_return_max_distance
# (step2_scenes_v2 1.2.0 heals short flashes). Backup:
# pipeline_config_backup_20260928.
# 179 -> 190 on 28 Sep for Step 1 (scripts/step1_open_items_run.py migrate):
# path_rule_version, path_patient_folder_level, path_session_folder_level,
# path_session_patterns, clinic_timezone, capture_video_rule_version,
# capture_video_apple_creationdate_keys, capture_video_confidence,
# capture_video_edited_reason, capture_video_same_patient_sets,
# capture_video_filename_epoch_ms_regex.
# 190 -> 194 on 28 Sep for OCR: ocr_persistence_capped_by_scene_samples,
# ocr_short_words, ocr_single_frame_caption_min_confidence,
# ocr_single_frame_caption_bands (ocr_wordlist_path and ocr_known_terms
# changed in place).
# content_v1 (29 Sep 2026, content_v1 run of Task 2): test_v1's 195 keys plus
# short_video_single_scene_seconds, retry_max_attempts and
# require_validation_before_completed.
# 198 -> 203 on 29 Sep 2026 (content_v1_step2_fixes_20260929.sql): the four
# scene_edited_mask_* keys (off) and transcript_vocabulary_hint.
# 203 -> 205 on 29 Sep 2026 (content_v1_step2_fixes_20260929c.sql):
# scene_raw_cut_check and transcription_word_timestamps.
# 205 -> 213 on 30 Sep 2026 (scripts_content_v1/ocr_fix_20260930.py, STEP2_OCR
# 3.1.0, config content_v1_step2_2026_09_30_a): ocr_domain_terms,
# ocr_contractions, ocr_token_edge_punctuation, ocr_quantity_words,
# ocr_quantity_number_max_run, ocr_standalone_fragment_max_letters,
# ocr_line_split_min_gap_heights, ocr_logo_lockup (ocr_known_terms changed in
# place: DR LEW KAR MIN).
# 213 -> 230 on 30 Sep 2026 (scripts_content_v1/step3_vision.py configure,
# config content_v1_step3_2026_09_30_a): the 17 vision_* keys for STEP3_VISION.
# 230 -> 236 on 30 Sep 2026 (config content_v1_step3_2026_09_30_b): Step 3 in the
# Claude Code session - vision_execution_mode, vision_long_scene_seconds,
# vision_moment_seconds, vision_long_scene_frames_max, vision_image_token_divisor,
# vision_image_tokens_max (vision_model -> claude-opus-5-5 changed in place).
# 236 -> 237 on 30 Sep 2026 (config content_v1_step3_2026_09_30_c):
# vision_format_activities (main-activity rule); LAB_COAT added to the clothing check.
# 237 -> 241 on 1 Oct 2026 (config content_v1_step3_2026_10_01_d): vision_device_rule,
# vision_speech_min_seconds, vision_no_speech_confidence,
# vision_speech_below_min_confidence.
# 241 -> 243 on 1 Oct 2026 (config content_v1_step3_2026_10_01_e):
# vision_machine_type_max_confidence, vision_spoken_language_manual.
# 243 -> 244 on 5 Oct 2026 (config content_v1_step3_2026_10_05_g, step3_vision
# 1.5.1): vision_linked_name_rule.
# 244 -> 245 with tests/keyframe_selection/sql/001_keyframe_selection_columns.sql
# (key keyframe_selection, keyframe_selection/1.4.1). Run that SQL before the
# next pipeline run; until then this check reports 244 != 245.
# +1 on 6 Oct 2026 (config content_v1_step3_2026_10_06_h, step3_vision 1.6.0):
# vision_caption_rule -> 245 live; 246 once 001 has run (until then 245 != 246).
# +2 on 6 Oct 2026 (config content_v1_step3_2026_10_06_k, step3_vision 1.6.3):
# vision_sex_min_confidence, vision_brand_card_patterns -> 247 live; 248 once
# 001 has run (until then 247 != 248).
# Step 2b switch (6 Oct 2026, applied): SQL 001 added keyframe_selection (248)
# and the step3_vision 1.7.0 config (content_v1_step3_2026_10_06_m) added
# vision_frame_source -> 249.
# Migration 007 (7 Oct 2026, step3_vision 1.8.0, config _n) added
# vision_camera_crew_patterns, vision_duplicate_person_seconds,
# vision_no_people_rule -> 252.
# Migration 008 (7 Oct 2026, Step 4) added embedding_model,
# embedding_dimensions, embedding_function, search_text_max_words,
# search_text_recipe_version, search_text_language_rule -> 258.
# Migration 009 (7 Oct 2026, Step 5) added 8 search_* keys -> 266.
#
# 7 Oct 2026 (after 010): check 2 NO LONGER compares an exact row count - it
# broke at 252, 258 and 266 each time a later step added its own keys. Check 2
# is now: every key of the LATEST Step 3 config snapshot (content_v1_step3_*,
# by taken_at - that snapshot holds every key Steps 1-3 used when it was
# taken) is still in pipeline_config, plus REQUIRED_CONFIG_KEYS below. Keys
# added by Step 4 / Step 5 are allowed. A Step 3 config change takes a new
# snapshot, so nothing here needs bumping.
# EXPECTED_CONFIG_ROWS is kept ONLY because step2_ocr_run.py (the 30 Sep OCR
# rollout tool) imports it for its own count check; preflight does not use it.
EXPECTED_CONFIG_ROWS = 268
EXPECTED_CONFIG_KEY = "detect_format_from_content"
STEP1_3_SNAPSHOT_LIKE = r"content_v1\_step3\_%"
REQUIRED_CONFIG_KEYS = (EXPECTED_CONFIG_KEY, "keyframe_storage_bucket", "keyframe_selection",
                        "vision_model", "vision_prompt_version", "vision_frame_source")
# the probe columns behind check 2 (Step 1 and Step 2 probes share them)
CONFIG_KEYS_PROBE = """
  (select config_version from content_v1.config_snapshots
     where config_version like '{like}'
     order by taken_at desc limit 1)                               as step13_snapshot,
  (select coalesce(string_agg(k, ', ' order by k), '')
     from (select jsonb_object_keys(payload) as k from content_v1.config_snapshots
            where config_version = (select config_version from content_v1.config_snapshots
                                     where config_version like '{like}'
                                     order by taken_at desc limit 1)
           union select unnest(array[{req}])) need
     where not exists (select 1 from content_v1.pipeline_config c
                       where c.key = need.k))                      as cfg_missing_keys,
""".replace("{like}", STEP1_3_SNAPSHOT_LIKE).replace(
    "{req}", ", ".join("'{}'".format(k) for k in REQUIRED_CONFIG_KEYS))


def config_key_checks(row, check):
    """Check 2 (Steps 1-3 config): the latest Step 3 snapshot exists and every
    key it holds, plus REQUIRED_CONFIG_KEYS, is in pipeline_config. Extra keys
    (Step 4 / 5) are fine."""
    check("2. latest Step 3 config snapshot present",
          bool(row.get("step13_snapshot")),
          "a config_snapshots row like {!r}".format(STEP1_3_SNAPSHOT_LIKE),
          repr(row.get("step13_snapshot")))
    check("2a. pipeline_config has every key Steps 1-3 need",
          row.get("cfg_missing_keys") == "",
          "every key of {} and {} present ({} rows now; extra keys allowed)".format(
              row.get("step13_snapshot"), ", ".join(REQUIRED_CONFIG_KEYS), row.get("cfg_rows")),
          "missing: {}".format(row.get("cfg_missing_keys")))

# Check 4 used to assert an exact base-table count (27). Every migration
# since has left a dated backup table behind, so the count said nothing
# about whether the schema was right - it only said how many backups had
# been taken. The check is now: every table the pipeline needs exists, and
# nothing else exists except backups named <table>_backup_YYYYMMDD. A
# stray table is still a failure; so is one that was deliberately removed
# coming back (semantic_layer_catalog, dropped 24 Sep 2026 for issue #8).
REQUIRED_TABLES = frozenset({
    # content_v1's 28 base tables as of 29 Sep 2026 (content_v1_close_remaining_gaps).
    "activities", "analysis_runs", "asset_derivatives", "asset_keyframes",
    "asset_media_probe", "asset_quality_spans", "asset_quarantine",
    "asset_scenes", "asset_sources", "asset_summary",
    "asset_technical_metadata", "asset_transcript_chunks", "assets",
    "batches", "carousel_members", "config_snapshots", "devices",
    "ocr_observations", "persons", "pipeline_config", "scan_runs",
    "scene_activities", "scene_moments", "scene_persons", "scene_summary",
    "source_files", "source_folders", "sync_runs",
    # 12 Oct 2026 (user OK): Step 6 thumbnails table, added 8 Oct 2026.
    "keyframe_thumbnails",
})
# The unified template (ml_test) has its own fixed table set: template's 24
# core tables, content_v1's L1/L3/L4/L7 layers and 5 new ones - 46 in all, read
# from the live schema on 23 Oct 2026. Same rule: all present, nothing else.
UNIFIED_REQUIRED_TABLES = frozenset({
    "activities", "aesthetic_items", "ai_jobs", "analysis_runs", "appearances",
    "asset_analysis", "asset_derivatives", "asset_keyframes",
    "asset_media_probe", "asset_private_names", "asset_quality_spans",
    "asset_quarantine", "asset_sources", "asset_technical_metadata",
    "asset_transcript_chunks", "batches", "before_after_cases",
    "carousel_members", "config_snapshots", "devices", "ingest_tokens",
    "journey_items", "journey_stages", "library_info", "media_assets",
    "media_embeddings", "media_scenes", "norwood_stages", "ocr_observations",
    "people", "person_face_embeddings", "pipeline_config", "scan_runs",
    "scene_activities", "scene_analysis", "scene_embeddings", "scene_moments",
    "scene_person_detail", "scene_persons", "scene_review",
    "source_file_private", "source_files", "source_folders", "sync_runs",
    "treatments", "vocab_map",
    # added 2 Nov 2026: raw/edited labels (007) and people found in OCR/speech (006)
    "video_edit_class", "people_name_patterns", "people_text_hits",
    # added 9 Nov 2026: the reference library (010, 011), made 6-7 Nov; found by the
    # first nightly test, whose Step 1 refused to write while these were unknown
    "ref_appearance", "ref_aspect_groups", "ref_aspect_items", "ref_aspects",
    "ref_asset_branch", "ref_asset_text", "ref_branch_rules", "ref_branches",
    "ref_categories", "ref_lookup", "ref_object_files", "ref_objects",
    "ref_people_files", "ref_point_items", "ref_point_text_hits", "ref_points",
    # added 11 Nov 2026: Content Desk thumbnails (016, made 10 Nov); found by the
    # 11 Nov nightly run, whose Step 1 refused to write while it was unknown
    "desk_thumbs",
    # added 18 Nov 2026: the record of every nightly run (018, user OK); batches 76-88
    # refused Step 1 for a few minutes while it was unknown
    "nightly_runs",
})
if target.is_unified():
    REQUIRED_TABLES = UNIFIED_REQUIRED_TABLES
# One optional lowercase letter after the date: a second backup taken the same
# day (answer_key_backup_20260924b, from answer_key_verify.py for issue #9).
BACKUP_TABLE_RE = re.compile(r"_backup_\d{8}[a-z]?$")


def table_problems(names):
    """(missing required tables, unexpected non-backup tables)."""
    present = set(names)
    missing = sorted(REQUIRED_TABLES - present)
    unexpected = sorted(t for t in present - REQUIRED_TABLES
                        if not BACKUP_TABLE_RE.search(t))
    return missing, unexpected


def check_tables(row, check):
    """Check 4, shared by the Step 1 and Step 2 gates."""
    names = [n for n in (row.get("base_table_names") or "").split(",") if n]
    missing, unexpected = table_problems(names)
    row["missing_tables"] = missing
    row["unexpected_tables"] = unexpected
    check("4. required tables present",
          not missing,
          "all {} required tables in {}".format(len(REQUIRED_TABLES),
                                                EXPECTED_SCHEMA),
          "missing: {}".format(", ".join(missing)))
    check("4b. no unexpected tables",
          not unexpected,
          "only required tables and *_backup_YYYYMMDD copies",
          "unexpected: {}".format(", ".join(unexpected)))


# The project this target must never touch. Named so the failure message can
# say so. For content_v1 it is the project read by mistake on 22 Sep 2026; for
# ml_test it is content_v1's own project, so the two targets cannot cross.
KNOWN_WRONG_REF = target.profile()["refuse_refs"][0]
KNOWN_WRONG_NAME = {"content_v1": "project-a",
                    "ml_test": "project-b"}[target.name()]

PHASE_PRE_RUN = "pre_run"
PHASE_IN_RUN = "in_run"
# A resumed run starts with Step 2 rows already present from the run it is
# continuing. "Empty" is then the wrong assertion; "exactly the baseline this
# run inherited, and nothing else" is the right one.
PHASE_RESUME = "resume"


class PreflightFailed(RuntimeError):
    """Raised when any assertion fails. The caller must not write."""


PROBE_SQL = """
select
  (select count(*) from {s}.config_snapshots
     where config_version = '{snap}')                              as snap_rows,
  (select left(fingerprint, 8) from {s}.config_snapshots
     where config_version = '{snap}')                              as snap_fp8,
  (select count(*) from {s}.pipeline_config)                       as cfg_rows,
{cfgkeys}  (select count(*) from {s}.pipeline_config
     where key = '{cfgkey}')                                       as has_cfg_key,
  (select count(*) from information_schema.columns
     where table_schema = '{s}' and table_name = 'source_files'
       and column_name = 'parent_visible')                         as has_parent_visible,
  (select count(*) from information_schema.tables
     where table_schema = '{s}' and table_type = 'BASE TABLE')     as base_tables,
  (select string_agg(table_name::text, ',' order by table_name)
     from information_schema.tables
     where table_schema = '{s}' and table_type = 'BASE TABLE')     as base_table_names,
  (select count(*) from {s}.assets)                                as asset_rows,
  current_database()                                               as db_name,
  coalesce(inet_server_addr()::text, 'local')                      as server_addr
""".replace("{cfgkeys}", CONFIG_KEYS_PROBE).format(s="content_v1",   # rewritten by db.py
                                                 snap=EXPECTED_SNAPSHOT,
                                                 cfgkey=EXPECTED_CONFIG_KEY)


class Preflight:
    """Holds the probe transport. Re-probes on every assert_ready call."""

    def __init__(self, fetch_one, target_hint=None, baseline_assets=0):
        """fetch_one(sql) -> mapping of column name to value.

        target_hint is whatever identifies the connection (a URL, a DSN host).
        It is only used to fail early with a clear message; it is never trusted
        in place of the five assertions. Never pass a secret here.

        baseline_assets is the asset count a batch run starts from, declared
        by a human on the command line (step1_ingest.py
        --expect-existing-assets). It is never read from the database: a
        baseline taken from the live count would agree with any state.
        0 keeps the original first-load behaviour.
        """
        self._fetch_one = fetch_one
        self._target_hint = target_hint or ""
        self.baseline_assets = int(baseline_assets)
        self.last_probe = None

    # ------------------------------------------------------------------
    def assert_ready(self, phase, assets_written=0):
        """Raise PreflightFailed unless every assertion holds. Call before
        EVERY write, not once per run."""
        if phase not in (PHASE_PRE_RUN, PHASE_IN_RUN):
            raise ValueError("unknown phase: {!r}".format(phase))

        self._assert_hint_not_wrong_project()

        row = dict(self._fetch_one(PROBE_SQL))
        self.last_probe = row

        failures = []

        def check(label, ok, expected, actual):
            if not ok:
                failures.append("  {}\n      expected: {}\n      actual:   {}"
                                .format(label, expected, actual))

        check("1. config snapshot present",
              row["snap_rows"] == 1,
              "exactly 1 row with config_version {!r}".format(EXPECTED_SNAPSHOT),
              "{} row(s)".format(row["snap_rows"]))

        check("1b. config snapshot fingerprint",
              (row["snap_fp8"] or "") == EXPECTED_FINGERPRINT_PREFIX,
              "fingerprint starting {}".format(EXPECTED_FINGERPRINT_PREFIX),
              repr(row["snap_fp8"]))

        config_key_checks(row, check)

        check("2b. pipeline_config decision key",
              row["has_cfg_key"] == 1,
              "key {!r} present".format(EXPECTED_CONFIG_KEY),
              "present" if row["has_cfg_key"] else "ABSENT")

        check("3. source_files.parent_visible column",
              row["has_parent_visible"] == 1,
              "column present",
              "present" if row["has_parent_visible"] else "ABSENT")

        check_tables(row, check)

        base = self.baseline_assets
        if phase == PHASE_PRE_RUN and base == 0:
            check("5. assets empty before the run",
                  row["asset_rows"] == 0,
                  "0 rows in {}.assets".format(EXPECTED_SCHEMA),
                  "{} rows".format(row["asset_rows"]))
        elif phase == PHASE_PRE_RUN:
            check("5. assets at the declared baseline before the run",
                  row["asset_rows"] == base,
                  "{} rows in {}.assets (declared baseline)".format(
                      base, EXPECTED_SCHEMA),
                  "{} rows".format(row["asset_rows"]))
        else:
            check("5. assets accounted for by this run",
                  row["asset_rows"] == base + assets_written,
                  "{} rows ({} baseline + {} written by this run)".format(
                      base + assets_written, base, assets_written),
                  "{} rows".format(row["asset_rows"]))

        if failures:
            raise PreflightFailed(self._message(failures, phase, row))

    # ------------------------------------------------------------------
    def _assert_hint_not_wrong_project(self):
        hint = self._target_hint.lower()
        if KNOWN_WRONG_REF in hint or KNOWN_WRONG_NAME in hint:
            raise PreflightFailed(
                "REFUSING TO PROCEED: the connection target names a project "
                "that target {} must never write to.\n"
                "  target hint contains: {} / {}\n"
                "  Nothing may be written to it.".format(
                    target.name(), KNOWN_WRONG_REF, KNOWN_WRONG_NAME))
        if hint and EXPECTED_PROJECT_REF not in hint:
            raise PreflightFailed(
                "REFUSING TO PROCEED: the connection target does not name the "
                "intended project.\n"
                "  expected ref: {}\n"
                "  target hint:  {}".format(EXPECTED_PROJECT_REF, hint))

    # ------------------------------------------------------------------
    @staticmethod
    def _message(failures, phase, row):
        return (
            "PRE-FLIGHT FAILED ({} check(s), phase={}). NOTHING WAS WRITTEN.\n"
            "{}\n\n"
            "  A mismatch means one of two things:\n"
            "    - the connection is pointed at the wrong database, or\n"
            "    - the intended database is not in the state this run assumes.\n"
            "  Both require a human. Do not retry, and do not edit the data to "
            "make the check pass.\n"
            "  Connected to database {!r} at {!r}; target {} expects project "
            "{} ({}),\n  schema {}.".format(
                len(failures), phase, "\n".join(failures),
                row.get("db_name"), row.get("server_addr"), target.name(),
                EXPECTED_PROJECT_REF, EXPECTED_PROJECT_NAME, EXPECTED_SCHEMA))


# --------------------------------------------------------------------------
# Standalone read-only run
# --------------------------------------------------------------------------

def _describe(row):
    order = ["db_name", "server_addr", "snap_rows", "snap_fp8", "cfg_rows",
             "has_cfg_key", "has_parent_visible", "base_tables",
             "missing_tables", "unexpected_tables", "asset_rows"]
    return "\n".join("  {:<20} {}".format(k, row.get(k)) for k in order)


def main():
    import sys
    try:
        from db import connect, fetch_one_factory   # the write layer's transport
    except ImportError:
        print("preflight: no transport available yet.\n"
              "  scripts/db.py (the connection layer) has not been written, and "
              "the\n  transport decision - direct Postgres vs PostgREST - is "
              "still open.\n"
              "  The assertion logic here is transport-independent: give "
              "Preflight()\n  any fetch_one(sql) -> dict and it works.",
              file=sys.stderr)
        return 2

    from db import SecretMissing, WrongTarget

    try:
        conn, hint = connect()
    except SecretMissing as exc:
        print("preflight: cannot connect - {}".format(exc), file=sys.stderr)
        return 2
    except WrongTarget as exc:
        print("preflight: {}".format(exc), file=sys.stderr)
        return 1

    try:
        pf = Preflight(fetch_one_factory(conn), target_hint=hint)
        pf.assert_ready(PHASE_PRE_RUN)
    except PreflightFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        # Read-only probe: never leave a transaction open on the target.
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
    print("PRE-FLIGHT PASSED (phase=pre_run). Safe to write.")
    print(_describe(pf.last_probe))
    return 0


def main_step2(argv):
    """The Step 2 gate on its own, read-only, exactly as step2_analyse calls
    it (resume phase when Step 2 rows exist, baseline = what is there now).

        python scripts_content_v1/preflight.py --step2
        python scripts_content_v1/preflight.py --step2 --simulate-missing FILE
        python scripts_content_v1/preflight.py --step2 --simulate-extra FILE

    --simulate-missing hides the asset named FILE from check 5 (as if Step 1
    had not written it); --simulate-extra hides FILE's source file from the
    taken set (as if the asset had no source). Both only change the probe's
    SELECT; nothing is written."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="preflight.py --step2")
    ap.add_argument("--step2", action="store_true")
    ap.add_argument("--simulate-missing", metavar="FILE")
    ap.add_argument("--simulate-extra", metavar="FILE")
    args = ap.parse_args(argv)
    from db import Reconnecting
    import step2_analyse as s2
    handle = Reconnecting()

    def probe(sql):
        c = handle.ensure()
        with c.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            row = cur.fetchone()
        c.rollback()
        return dict(zip(cols, row))

    def lit(s):
        return "'" + s.replace("'", "''") + "'"

    try:
        pf = Step2Preflight(probe, target_hint=handle.hint)
        sql = step2_probe_sql()
        if args.simulate_missing:
            sql = step2_probe_sql("(select * from content_v1.assets where "
                                  "file_name <> {})".format(lit(args.simulate_missing)))
            print("SIMULATION (read-only): asset {!r} hidden from check 5"
                  .format(args.simulate_missing))
        if args.simulate_extra:
            sql = sql.replace(STEP1_TAKEN_SOURCE_FILES,
                              STEP1_TAKEN_SOURCE_FILES + "\n  and file_name <> "
                              + lit(args.simulate_extra))
            print("SIMULATION (read-only): source file {!r} hidden from the "
                  "taken set".format(args.simulate_extra))
        pf.probe_sql = sql
        baseline = s2.current_step2_counts(handle.ensure())
        phase = PHASE_RESUME if any(baseline.values()) else PHASE_PRE_RUN
        pf.assert_ready(phase, baseline=baseline)
    except PreflightFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        handle.close()
    r = pf.last_probe
    print("STEP 2 PRE-FLIGHT PASSED (phase={}).".format(phase))
    for k in ("taken_source_files", "expected_assets", "asset_rows",
              "missing_asset_files", "extra_asset_files", "atm_rows",
              "cfg_rows", "base_tables"):
        print("  {:<20} {}".format(k, r.get(k)))
    return 0


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------
# Same shape as the Step 1 gate: invariant checks that hold throughout, plus
# progress checks that are phase-aware. Step 1's rows are treated as fixed
# furniture - if assets or asset_technical_metadata move off 27, something has
# touched Step 1 and this run must not continue.
# 24 -> 27 on 25 Sep: V16-V18 ingested by step1_ingest_incremental.py.

# content_v1: the expected asset count is NOT a constant. It was pinned at 1
# (EXPECTED_ASSETS_AFTER_STEP1 = 1, for the first one-video run on 29 Sep)
# and went stale when content_v1 grew to 28 assets. Check 5 now derives it
# from the source of truth, the rule Step 1 writes by (writer.py
# _existing_asset / _insert_asset): one asset per distinct content_sha256
# among the source files Step 1 takes. It fails only when the assets differ
# from that, and names the files on each side.
KEYFRAME_BUCKET = target.profile()["keyframe_bucket"]

STEP2_TABLES = ["asset_scenes", "asset_keyframes", "asset_derivatives",
                "asset_transcript_chunks", "ocr_observations"]

# Source files that must have an asset: taken, hashed, present.
STEP1_TAKEN_SOURCE_FILES = """
select id, file_name, relative_path, content_sha256
from content_v1.source_files
where decision = 'TAKE' and hash_status = 'HASHED'
  and not is_folder and not trashed and not is_missing"""

ASSETS_RELATION = "content_v1.assets"


def step2_probe_sql(assets_relation=ASSETS_RELATION):
    """The Step 2 probe. `assets_relation` is what check 5 reads as the
    assets table - content_v1.assets in a real run; a filtered subquery to
    simulate a missing or extra asset without writing anything."""
    return STEP2_PROBE_SQL_TEMPLATE.replace("{assets}", assets_relation)


STEP2_PROBE_SQL_TEMPLATE = """
with a as (select id, content_hash, file_name from {assets} x),
     sf as (""" + STEP1_TAKEN_SOURCE_FILES + """)
select
  (select count(*) from sf)                                        as taken_source_files,
  (select count(distinct content_sha256) from sf)                  as expected_assets,
  (select string_agg(coalesce(sf.relative_path, sf.file_name), '; '
                     order by sf.relative_path, sf.file_name)
     from sf
     where not exists (select 1 from content_v1.asset_sources s
                       join a on a.id = s.asset_id
                       where s.source_file_id = sf.id
                         and a.content_hash = sf.content_sha256))  as missing_asset_files,
  (select string_agg(a.file_name, '; ' order by a.file_name)
     from a
     where not exists (select 1 from sf
                       join content_v1.asset_sources s on s.source_file_id = sf.id
                       where s.asset_id = a.id
                         and sf.content_sha256 = a.content_hash))  as extra_asset_files,
  (select string_agg(a.file_name, '; ' order by a.file_name)
     from a
     where not exists (select 1 from content_v1.asset_technical_metadata m
                       where m.asset_id = a.id))                   as atm_missing_files,
  (select string_agg(coalesce(x.file_name, m.asset_id::text), '; ')
     from content_v1.asset_technical_metadata m
     left join content_v1.assets x on x.id = m.asset_id
     where not exists (select 1 from a where a.id = m.asset_id))   as atm_extra_files,
  (select count(*) from a)                                         as asset_rows,
  (select count(*) from content_v1.config_snapshots
     where config_version = '{snap}')                              as snap_rows,
  (select left(fingerprint, 8) from content_v1.config_snapshots
     where config_version = '{snap}')                              as snap_fp8,
  (select count(*) from content_v1.pipeline_config)                   as cfg_rows,
{cfgkeys}  (select count(*) from content_v1.pipeline_config
     where key = '{cfgkey}')                                       as has_cfg_key,
  (select count(*) from information_schema.columns
     where table_schema = 'content_v1' and table_name = 'source_files'
       and column_name = 'parent_visible')                         as has_parent_visible,
  (select count(*) from information_schema.tables
     where table_schema = 'content_v1' and table_type = 'BASE TABLE') as base_tables,
  (select string_agg(table_name::text, ',' order by table_name)
     from information_schema.tables
     where table_schema = 'content_v1' and table_type = 'BASE TABLE') as base_table_names,
  (select count(*) from content_v1.asset_technical_metadata)          as atm_rows,
  (select count(*) from content_v1.asset_scenes)                      as scene_rows,
  (select count(*) from content_v1.asset_keyframes)                   as keyframe_rows,
  (select count(*) from content_v1.asset_derivatives)                 as derivative_rows,
  (select count(*) from content_v1.asset_transcript_chunks)           as transcript_rows,
  (select count(*) from content_v1.ocr_observations)                  as ocr_rows,
  (select count(*) from storage.buckets where name = '{bucket}')   as bucket_rows,
  (select bool_or(public) from storage.buckets
     where name = '{bucket}')                                      as bucket_public,
  current_database()                                               as db_name
""".replace("{cfgkeys}", CONFIG_KEYS_PROBE).replace("{snap}", EXPECTED_SNAPSHOT).replace(
    "{cfgkey}", EXPECTED_CONFIG_KEY).replace("{bucket}", KEYFRAME_BUCKET)

STEP2_PROBE_SQL = step2_probe_sql()

# Probe column -> the count this run has written so far.
STEP2_COUNT_COLUMN = {
    "asset_scenes": "scene_rows",
    "asset_keyframes": "keyframe_rows",
    "asset_derivatives": "derivative_rows",
    "asset_transcript_chunks": "transcript_rows",
    "ocr_observations": "ocr_rows",
}


class Step2Preflight(Preflight):
    """Checks 1-4 unchanged, plus Step 1 integrity, Step 2 progress, and the
    storage bucket. Call before EVERY write, exactly as in Step 1."""

    def assert_ready(self, phase, written=None, baseline=None):
        """written maps table name -> rows THIS run has written. baseline maps
        table name -> rows inherited from an earlier run of the same config
        version, which a resumed run must tolerate but must not exceed by
        anything it did not write itself.

        pre_run   every Step 2 table is 0
        resume    every Step 2 table equals the baseline exactly
        in_run    every Step 2 table equals baseline + written
        """
        if phase not in (PHASE_PRE_RUN, PHASE_IN_RUN, PHASE_RESUME):
            raise ValueError("unknown phase: {!r}".format(phase))
        written = written or {}
        baseline = baseline or {}

        self._assert_hint_not_wrong_project()

        row = dict(self._fetch_one(getattr(self, "probe_sql", STEP2_PROBE_SQL)))
        self.last_probe = row

        failures = []

        def check(label, ok, expected, actual):
            if not ok:
                failures.append("  {}\n      expected: {}\n      actual:   {}"
                                .format(label, expected, actual))

        # 1-4: unchanged from Step 1.
        check("1. config snapshot present", row["snap_rows"] == 1,
              "1 row with config_version {!r}".format(EXPECTED_SNAPSHOT),
              "{} row(s)".format(row["snap_rows"]))
        check("1b. config snapshot fingerprint",
              (row["snap_fp8"] or "") == EXPECTED_FINGERPRINT_PREFIX,
              "fingerprint starting {}".format(EXPECTED_FINGERPRINT_PREFIX),
              repr(row["snap_fp8"]))
        config_key_checks(row, check)
        check("2b. pipeline_config decision key", row["has_cfg_key"] == 1,
              "key {!r} present".format(EXPECTED_CONFIG_KEY),
              "present" if row["has_cfg_key"] else "ABSENT")
        check("3. source_files.parent_visible column",
              row["has_parent_visible"] == 1, "column present",
              "present" if row["has_parent_visible"] else "ABSENT")
        check_tables(row, check)

        # 5. Step 1 is intact: one asset per distinct content hash among the
        # source files Step 1 took, each linked to its source file.
        missing, extra = row["missing_asset_files"], row["extra_asset_files"]
        check("5. assets match Step 1's source files",
              row["asset_rows"] == row["expected_assets"]
              and not missing and not extra,
              "{} assets (one per distinct content_sha256 among {} taken "
              "source files)".format(row["expected_assets"],
                                     row["taken_source_files"]),
              "{} assets{}{}".format(
                  row["asset_rows"],
                  "\n      source files with no asset: {}".format(missing)
                  if missing else "",
                  "\n      assets with no taken source file: {}".format(extra)
                  if extra else ""))
        atm_missing, atm_extra = row["atm_missing_files"], row["atm_extra_files"]
        check("5b. asset_technical_metadata from Step 1 intact",
              row["atm_rows"] == row["asset_rows"]
              and not atm_missing and not atm_extra,
              "{} rows (one per asset)".format(row["asset_rows"]),
              "{} rows{}{}".format(
                  row["atm_rows"],
                  "\n      assets with no metadata row: {}".format(atm_missing)
                  if atm_missing else "",
                  "\n      metadata rows for no asset: {}".format(atm_extra)
                  if atm_extra else ""))

        # 6. Step 2 progress.
        # STEP2_SHARED_WRITERS=1 (2 Nov 2026, user: Step 2 for every file, fast):
        # several Step 2 runs write side by side (photo runs, 2-3 video runs),
        # so another run's rows legitimately appear between two checks. The
        # count must then be AT LEAST what this run inherited and wrote - a
        # lost or deleted row still fails, as does the wrong project (checks
        # 1-5) - instead of exactly that.
        shared = os.environ.get("STEP2_SHARED_WRITERS") == "1"
        for table in STEP2_TABLES:
            col = STEP2_COUNT_COLUMN[table]
            base = baseline.get(table, 0)
            if phase == PHASE_PRE_RUN and not shared:
                want, label = 0, "6. {} empty before the run".format(table)
            elif phase in (PHASE_PRE_RUN, PHASE_RESUME):
                want = base
                label = "6. {} matches the inherited baseline".format(table)
            else:
                want = base + written.get(table, 0)
                label = "6. {} accounted for by this run".format(table)
            ok = row[col] >= want if shared else row[col] == want
            check(label + (" (shared writers: at least)" if shared else ""), ok,
                  "{}{} rows".format("at least " if shared else "", want), "{} rows".format(row[col]))

        # 7. The bucket exists and is private.
        check("7. bucket {!r} exists".format(KEYFRAME_BUCKET),
              row["bucket_rows"] == 1, "1 bucket", "{}".format(row["bucket_rows"]))
        check("7b. bucket {!r} is not public".format(KEYFRAME_BUCKET),
              row["bucket_public"] is False, "public = false",
              "public = {}".format(row["bucket_public"]))

        if failures:
            raise PreflightFailed(self._message(failures, phase, row))


# --------------------------------------------------------------------------
# Step 3
# --------------------------------------------------------------------------
# Every Step 2 check still holds (Step 3 writes none of those tables), plus
# the schema Step 3 writes into: the columns added by
# content_v1_step3_prereq_columns_20260930 / content_v1_persons_reference_prep
# _20260930 / content_v1_step3_gaps_20261001, the STEP3_VISION run step
# (content_v1_step3_vision_run_step_20260930), and the activities reference
# Step 3 codes against.

STEP3_REQUIRED_COLUMNS = {
    "analysis_runs": ["input_tokens", "output_tokens", "cache_read_tokens",
                      "cache_write_tokens", "images_sent", "cost_usd",
                      "validation_status", "validation_detail"],
    "scene_persons": ["model", "model_version", "keyframe_id", "role_source",
                      "sex_source", "age_source", "ethnicity_source",
                      "person_link_source", "person_link_evidence",
                      "person_link_seconds", "person_link_confidence"],
    "scene_activities": ["model", "model_version", "keyframe_id", "device_code"],
    "scene_summary": ["model_version", "evidence_source", "evidence_keyframe_ids",
                      "language_confidence", "language_source"],
    "asset_summary": ["model_version", "evidence_source", "cover_keyframe_ids",
                      "spoken_language_confidence", "spoken_language_source"],
    "scene_moments": ["model_version"],
    "persons": ["aliases", "reference_code"],
}
# 27 from the brief + CHEMICAL_PEEL, BRAND_CARD, UNRELATED_FOOTAGE
# (content_v1_step3_gaps_20261001)
EXPECTED_ACTIVE_ACTIVITIES = 30

STEP3_PROBE_SQL = """
select
  (select coalesce(string_agg(t.tbl || '.' || t.col, ', '), '')
     from (values {pairs}) t(tbl, col)
     where not exists (select 1 from information_schema.columns c
                       where c.table_schema = 'content_v1'
                         and c.table_name = t.tbl and c.column_name = t.col))
                                                                 as missing_columns,
  (select pg_get_constraintdef(oid) like '%STEP3_VISION%'
     from pg_constraint where conname = 'analysis_runs_step_check'
      and conrelid = 'content_v1.analysis_runs'::regclass)       as step3_step_allowed,
  (select column_default is null from information_schema.columns
     where table_schema = 'content_v1' and table_name = 'persons'
       and column_name = 'person_type')                          as person_type_no_default,
  (select count(*) from content_v1.activities where active)      as active_activities,
  -- 12 Oct 2026 (user OK): S3-5 / S3-6 only look at assets whose own step
  -- passed, so assets still queued for Step 2 in a later batch do not block.
  (select count(*) from content_v1.assets a
     where (select r.status from content_v1.analysis_runs r
             where r.asset_id = a.id and r.step = 'STEP2_KEYFRAMES'
             order by r.attempt_number desc, r.created_at desc limit 1) = 'COMPLETED'
       and not exists (select 1 from content_v1.asset_keyframes k
                       where k.asset_id = a.id and k.image_status = 'STORED'))
                                                                 as assets_without_frames,
  (select count(*) from content_v1.asset_keyframes k
     where k.technical_quality_score is null
       and (select r.status from content_v1.analysis_runs r
             where r.asset_id = k.asset_id and r.step = 'STEP2_QUALITY'
             order by r.attempt_number desc, r.created_at desc limit 1) = 'COMPLETED')
                                                                 as unscored_keyframes
""".replace("{pairs}", ", ".join(
    "('{}', '{}')".format(t, c) for t, cols in sorted(STEP3_REQUIRED_COLUMNS.items())
    for c in cols))


class Step3Preflight(Step2Preflight):
    """The Step 2 gate, then the Step 3 schema: call before a Step 3 write."""

    def assert_ready(self, phase, written=None, baseline=None):
        super().assert_ready(phase, written=written, baseline=baseline)
        row = dict(self._fetch_one(STEP3_PROBE_SQL))
        self.last_step3_probe = row
        failures = []

        def check(label, ok, expected, actual):
            if not ok:
                failures.append("  {}\n      expected: {}\n      actual:   {}"
                                .format(label, expected, actual))

        check("S3-1. Step 3 columns present", not row["missing_columns"],
              "all present", "missing: " + (row["missing_columns"] or ""))
        check("S3-2. analysis_runs.step allows STEP3_VISION",
              row["step3_step_allowed"] is True, "allowed",
              repr(row["step3_step_allowed"]))
        check("S3-3. persons.person_type has no default",
              row["person_type_no_default"] is True, "no default",
              repr(row["person_type_no_default"]))
        check("S3-4. activities reference",
              row["active_activities"] == EXPECTED_ACTIVE_ACTIVITIES,
              "{} active codes".format(EXPECTED_ACTIVE_ACTIVITIES),
              "{}".format(row["active_activities"]))
        check("S3-5. every asset with STEP2_KEYFRAMES COMPLETED has a stored keyframe",
              row["assets_without_frames"] == 0, "0 assets without one",
              "{}".format(row["assets_without_frames"]))
        check("S3-6. every keyframe of an asset with STEP2_QUALITY COMPLETED has a quality score",
              row["unscored_keyframes"] == 0, "0 unscored",
              "{}".format(row["unscored_keyframes"]))
        if failures:
            raise PreflightFailed(self._message(failures, phase, row))


def main_step3(argv):
    """The Step 3 gate on its own, read-only.

        python scripts_content_v1/preflight.py --step3"""
    import sys
    from db import Reconnecting
    import step2_analyse as s2
    handle = Reconnecting()

    def probe(sql):
        c = handle.ensure()
        with c.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            row = cur.fetchone()
        c.rollback()
        return dict(zip(cols, row))

    try:
        pf = Step3Preflight(probe, target_hint=handle.hint)
        baseline = s2.current_step2_counts(handle.ensure())
        phase = PHASE_RESUME if any(baseline.values()) else PHASE_PRE_RUN
        pf.assert_ready(phase, baseline=baseline)
    except PreflightFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        handle.close()
    print("STEP 3 PRE-FLIGHT PASSED (phase={}).".format(phase))
    for k, v in pf.last_step3_probe.items():
        print("  {:<24} {}".format(k, v))
    return 0


def selftest():
    """Check 2 (config keys, 7 Oct 2026) on fake probe rows, then the live
    probe (read only). python scripts_content_v1/preflight.py --selftest"""
    fails = 0

    def case(name, row, want_ok):
        nonlocal fails
        failures = []
        config_key_checks(row, lambda label, ok, exp, act: None if ok else failures.append(label))
        ok = (not failures) == want_ok
        fails += 0 if ok else 1
        print("  {:<4} {:<62} {}".format("ok" if ok else "FAIL", name, failures or "passes"))

    base = {"step13_snapshot": "content_v1_step3_2026_10_07_n", "cfg_missing_keys": "", "cfg_rows": 252}
    print("config key check (fake probe rows):")
    case("every Step 1-3 key present, 252 rows", base, True)
    case("Step 4 / 5 keys added (266 rows) - no longer a failure", dict(base, cfg_rows=266), True)
    case("a Step 1-3 key missing (vision_model)", dict(base, cfg_missing_keys="vision_model"), False)
    case("no Step 3 config snapshot", dict(base, step13_snapshot=None), False)
    print("live probe (read only):")
    from db import Reconnecting
    handle = Reconnecting()
    try:
        c = handle.ensure()
        with c.cursor() as cur:
            cur.execute(STEP2_PROBE_SQL)
            row = dict(zip([d.name for d in cur.description], cur.fetchone()))
        c.rollback()
    finally:
        handle.close()
    case("live: {} keys of {} present ({} rows)".format(
        "all" if row["cfg_missing_keys"] == "" else "NOT all", row["step13_snapshot"],
        row["cfg_rows"]), row, True)
    print("selftest {}".format("PASSED" if not fails else "FAILED ({})".format(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys as _sys
    if "--selftest" in _sys.argv[1:]:
        raise SystemExit(selftest())
    if "--step3" in _sys.argv[1:]:
        raise SystemExit(main_step3(_sys.argv[1:]))
    if "--step2" in _sys.argv[1:]:
        raise SystemExit(main_step2(_sys.argv[1:]))
    raise SystemExit(main())
