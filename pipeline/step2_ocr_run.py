#!/usr/bin/env python3
"""
STEP2_OCR 3.0.0 - read on-screen text, score it, and set the status from the
score. Only STEP2_OCR is written: ocr_observations (per video, replaced) and
one analysis_runs row per video. Frames, scenes, transcripts and quality
spans are not touched. Photos are out of scope (none has an
expected_onscreen_text; their STEP2_OCR runs hold no rows) and are left as
they are.

  capture   reads each video's stored keyframes straight from the bucket
            (sha256-checked, decoded in memory, never written to disk),
            chooses frames by the coverage rule, runs the configured engine
            and variants, and caches readings as text + boxes only.
  backup    ocr_observations -> ocr_observations_backup_20260924 (RLS on).
  run       in ONE transaction: the config changes below, a new snapshot,
            observations built from the cache with and without
            normalisation, both scored by score_ocr, status from the
            normalised score, rows written. Without --commit everything runs
            and is rolled back; the plan's sha256 must match between two
            such runs and the commit.
  verify    reads back and checks the database against the plan.
  report    writes reports/ocr_results.md (before/after, both columns).

    python scripts/step2_ocr_run.py capture
    python scripts/step2_ocr_run.py backup --commit
    python scripts/step2_ocr_run.py run --plan work/ocr_v3/plan_1.json
    python scripts/step2_ocr_run.py run --plan work/ocr_v3/plan_2.json
    python scripts/step2_ocr_run.py run --commit --expect-sha <sha>
    python scripts/step2_ocr_run.py verify --plan work/ocr_v3/plan_1.json
    python scripts/step2_ocr_run.py report --plan work/ocr_v3/plan_1.json
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ocr_engine as E                                      # noqa: E402
import score_ocr as S                                       # noqa: E402
import step2_ocr as O                                       # noqa: E402

PROCESSOR_VERSION = "step2_ocr/3.0.0"
CONFIG_VERSION = "step2_2026_09_24_b"
SNAPSHOT_NOTE = ("STEP2_OCR 3.0.0: RapidOCR engine chosen by benchmark, "
                 "crop-and-upscale of small text, coverage rule instead of a "
                 "24-frame cap, number+unit and device-context rules, "
                 "dictionary correction, every-word garbage gate, text-block "
                 "merge, one known-term vocabulary.")
BACKUP_TABLE = "ocr_observations_backup_20260924"
CACHE = PROJECT_ROOT / "work" / "ocr_v3"
REPORT = PROJECT_ROOT / "reports" / "content_v1" / "ocr_results.md"

# --------------------------------------------------------------------------
# The configuration this version runs on. Every entry is applied inside the
# run's transaction and recorded, with its old value, in the report.
# (key, value, value_type, min, max, description) ; value None = delete.
# --------------------------------------------------------------------------
KNOWN_TERMS = {
    "DEVICE_LABEL": ["ULTRAFORMER", "ULTRAFORMER III", "SYLFIRM", "PLINEST",
                     "LASEMD", "HELIOS", "HELIOS 785", "OPTEK", "M-CURE",
                     "NOURKRIN", "PROFHILO"],
    "UI": ["STANDBY", "START", "READY"],
    "CLINIC_NAME": ["KLINIK DR INDER", "DR INDER", "DOCTOR LEW", "DR LEW"],
}
ENGINE_MODELS = {"det_version": "PP-OCRv6", "det_type": "small",
                 "rec_version": "PP-OCRv6", "rec_type": "small",
                 "rec_lang": None}
CHOSEN = "rapid_v6_small"
# Pipeline variant trials on V2/V9/V10 (+ V1/V3/V8/V12 for false positives),
# scored with the full acceptance rules and score_ocr - expected terms found
# with / without normalisation. See reports/ocr_results.md.
VARIANT_TRIALS = {
    "v6 native only": {"found_norm": "16/19", "found_raw": "13/19",
                       "false_positives": 0},
    "v6 + crop-upscale 3x under 40 px (CHOSEN)": {
        "found_norm": "17/19", "found_raw": "13/19", "false_positives": 0},
    "v6 + whole-frame 2x, no crop": {"found_norm": "16/19", "found_raw": "12/19",
                                     "false_positives": 0,
                                     "note": "caption no longer one line; ~3x slower"},
    "v6 + crop + inverted + contrast": {"found_norm": "13/19", "found_raw": "8/19",
                                        "garbage": ["SKIN LASEND"],
                                        "note": "pooling picks confident wrong readings"},
}

CHANGES = [
    ("ocr_engine", "rapidocr", "string", None, None,
     "OCR engine. rapidocr = PaddleOCR's PP-OCR models (PP-OCRv6) on ONNX runtime; "
     "chosen over easyocr on the measured benchmark in ocr_engine_benchmark."),
    ("ocr_engine_models", ENGINE_MODELS, "object", None, None,
     "RapidOCR detector/recogniser model version and size. See "
     "ocr_engine_benchmark for why these."),
    ("ocr_engine_benchmark", "__BENCHMARK__", "object", None, None,
     "The engine comparison the choice rests on (scripts/ocr_engine_bench.py "
     "on the stored keyframes): recall of expected terms, noise on videos "
     "with no text, seconds per frame. Recorded into every run's metadata."),
    ("ocr_known_terms", KNOWN_TERMS, "object", None, None,
     "The clinic's device, product, screen and clinic-name list - the ONE "
     "OCR vocabulary, used for exact matches, known-term normalisation and "
     "text types (groups named after a text type classify). Moved "
     "unchanged from ocr_text_type_lexicon + ocr_vocabulary_extra on "
     "2026-09-24. Curated by the operator; never built from answer_key."),
    ("ocr_text_type_lexicon", None, None, None, None, None),
    ("ocr_vocabulary_extra", None, None, None, None, None),
    ("ocr_crop_upscale_enabled", True, "boolean", None, None,
     "Re-read every box whose letter height is under "
     "ocr_upscale_min_text_height_px from a padded crop enlarged by "
     "ocr_upscale_factor."),
    ("ocr_crop_padding_fraction", 0.25, "number", 0.0, 2.0,
     "Margin around a small box before it is enlarged, as a fraction of the "
     "box's longer side, so letters at the edge are not cut."),
    ("ocr_full_frame_upscale", False, "boolean", None, None,
     "Also read the whole frame enlarged by ocr_upscale_factor. Off: measured "
     "16/19 against 17/19 for crop-and-upscale, about 3x slower, and it broke "
     "the one-line caption (ocr_engine_benchmark.variant_trials)."),
    ("ocr_upscale_factor", 3.0, "number", 1.0, 8.0,
     "Enlargement for small-text crops (and the whole-frame/contrast/"
     "inverted variants when those are on)."),
    ("ocr_upscale_min_text_height_px", 40, "integer", 1, 500,
     "Letter height (box short edge, px on the stored frame) under which a "
     "box is re-read from an enlarged crop."),
    ("ocr_contrast_normalise", False, "boolean", None, None,
     "CLAHE variant of the enlarged frame. Off: measured with inverted, "
     "recall fell 17/19 -> 13/19 and garbage was accepted "
     "(ocr_engine_benchmark.variant_trials)."),
    ("ocr_try_inverted", False, "boolean", None, None,
     "Inverted variant for light-on-dark screens. Off: see "
     "ocr_contrast_normalise - measured worse with the chosen engine."),
    ("ocr_full_coverage_max_seconds", 120, "integer", 1, 3600,
     "Videos up to this long have EVERY keyframe read."),
    ("ocr_long_video_stride_seconds", 2, "integer", 1, 2,
     "Longer videos: read every Nth keyframe (max 2, so text on screen for "
     "2 s or more always lands on a read frame) plus changed frames."),
    ("ocr_change_frame_threshold", 0.08, "number", 0.0, 1.0,
     "Longer videos: also read a keyframe whose mean absolute grey-level "
     "difference from the one before exceeds this (0-1)."),
    ("ocr_unit_tokens", ["J", "MJ", "W", "%", "MM", "CM", "HZ", "KHZ", "MS",
                         "S", "C"], "array", None, None,
     "Units a device shows after a setting. A number and its unit are one "
     "token ('0.4J' -> '0.4 J')."),
    ("ocr_split_letter_digit_boundaries", True, "boolean", None, None,
     "Write a model number run into its name, or a setting run into its unit, "
     "with a space ('HELIOS785' -> 'HELIOS 785'). Letters never change."),
    ("ocr_device_context_max_distance", 0.35, "number", 0.0, 1.0,
     "A reading that is only numbers/units is accepted only if a known "
     "device/screen term was read within this distance (fraction of the "
     "frame diagonal) in the same frame."),
    ("ocr_bare_number_min_digits", None, None, None, None, None),
    ("ocr_dictionary_correction_enabled", True, "boolean", None, None,
     "Correct a non-word to the ONE word-list word within "
     "ocr_dictionary_correct_max_distance at the same length (IRONT -> "
     "FRONT). Ambiguous = not corrected. raw_text keeps the reading."),
    ("ocr_dictionary_correct_max_distance", 1, "integer", 0, 3,
     "Edit distance for dictionary correction."),
    ("ocr_dictionary_correct_min_length", 5, "integer", 3, 20,
     "Shortest token dictionary correction may touch."),
    ("ocr_dictionary_correct_min_line_words", 2, "integer", 1, 20,
     "Dictionary correction only inside a line with at least this many OTHER "
     "real words - a lone token has no context ('UNGER' is YOUNGER half "
     "hidden, not UNDER) and is rejected instead."),
    ("ocr_min_real_word_share", 1.0, "number", 0.0, 1.0,
     "Share of a line's tokens that must be words, known terms or numbers. "
     "1.0: every one - 0.6 let 'THEIR PHONE AT THE IRONT' through."),
    ("ocr_block_merge_height_ratio_max", 1.35, "number", 1.0, 3.0,
     "Stacked lines join into one block only if their letter heights are "
     "within this ratio."),
    ("ocr_block_merge_max_gap_lines", 0.6, "number", 0.0, 3.0,
     "... and the gap between them is at most this many line heights."),
    ("ocr_block_merge_align_tolerance", 0.06, "number", 0.0, 1.0,
     "... and their left edges or centres are within this fraction of the "
     "frame width."),
    ("ocr_line_merge_max_dy_heights", 0.5, "number", 0.0, 3.0,
     "Boxes join one line only if their centres are within this many letter "
     "heights vertically (and their heights within "
     "ocr_block_merge_height_ratio_max) - a small logo on a caption's "
     "baseline is not part of the caption."),
    ("ocr_caption_block_line_gap", None, None, None, None, None),
    ("ocr_variant_box_overlap_min", 0.6, "number", 0.0, 1.0,
     "Two variant readings are one reading when their boxes overlap by this "
     "share of the SMALLER box (IoU missed a box inside a wider one)."),
    ("ocr_variant_box_iou_min", None, None, None, None, None),
]

UNTOUCHED = ("asset_scenes", "asset_keyframes", "asset_derivatives",
             "asset_transcript_chunks", "asset_quality_spans", "answer_key",
             "asset_technical_metadata", "assets", "source_files")


def benchmark_summary(required=True):
    p = PROJECT_ROOT / "work" / "ocr_bench" / "summary.json"
    if not p.exists():
        if not required:
            return None
        raise SystemExit("no engine benchmark at {}".format(p))
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {tag: {k: v for k, v in r.items() if k != "per_term"}
            for tag, r in raw.items()}


def changes(require_benchmark=True):
    out = []
    for key, value, vtype, lo, hi, desc in CHANGES:
        if value == "__BENCHMARK__":
            value = {"chosen": CHOSEN,
                     "engine_results": benchmark_summary(require_benchmark),
                     "variant_trials": VARIANT_TRIALS}
        out.append((key, value, vtype, lo, hi, desc))
    return out


def planned_config(cfg, require_benchmark=False):
    """The live config with this version's changes applied in memory. The
    benchmark record is only required where it is written (run)."""
    c = dict(cfg)
    for key, value, *_ in changes(require_benchmark):
        if value is None:
            c.pop(key, None)
        else:
            c[key] = json.loads(json.dumps(value))
    return c


# --------------------------------------------------------------------------

VIDEOS_SQL = """
select sf.metadata->>'step1_item' as item, a.id::text as asset_id,
       a.content_hash, a.file_name, m.duration_seconds
from content_v1.assets a
join content_v1.asset_technical_metadata m on m.asset_id = a.id
join content_v1.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
where m.media_kind = 'VIDEO'
"""
FRAMES_SQL = """
select k.asset_id::text, k.id::text, k.scene_id::text, k.timestamp_seconds,
       d.storage_path, d.content_hash
from content_v1.asset_keyframes k
join content_v1.asset_derivatives d on d.id = k.derivative_id
order by 1, 4
"""
SPANS_SQL = "select asset_id::text, start_seconds, end_seconds from content_v1.asset_quality_spans"


def item_key(i):
    return int(re.sub(r"\D", "", i))


def load_inputs(conn):
    with conn.cursor() as cur:
        cur.execute(VIDEOS_SQL)
        videos = {r[0]: {"item": r[0], "asset_id": r[1], "hash": r[2],
                         "file_name": r[3], "duration": float(r[4])}
                  for r in cur.fetchall()}
        cur.execute(FRAMES_SQL)
        frames = {}
        for aid, kid, sid, t, path, sha in cur.fetchall():
            frames.setdefault(aid, []).append(
                {"keyframe_id": kid, "scene_id": sid, "t": float(t),
                 "path": path, "sha": sha})
        cur.execute(SPANS_SQL)
        spans = {}
        for aid, a, b in cur.fetchall():
            spans.setdefault(aid, []).append((float(a), float(b)))
        cur.execute("select name from test_v1.treatments where is_active")
        treatments = [r[0] for r in cur.fetchall()]
        cur.execute("select alias from test_v1.treatment_aliases")
        aliases = [r[0] for r in cur.fetchall()]
        cur.execute("select a.id::text, k.expected_onscreen_text "
                    "from content_v1.answer_key k join content_v1.assets a "
                    "on a.id = k.asset_id where k.media_kind = 'VIDEO'")
        key = {aid: exp for aid, exp in cur.fetchall()}
    conn.rollback()
    return videos, frames, spans, treatments, aliases, key


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def stage_capture(conn, args, out, cfg_db):
    import cv2
    import numpy as np
    from storage import KeyframeStore
    cfg = planned_config(cfg_db)
    videos, frames, spans, *_ = load_inputs(conn)
    CACHE.mkdir(parents=True, exist_ok=True)
    engine = E.make_engine(cfg)
    store = KeyframeStore("keyframes")
    items = sorted(videos, key=item_key)
    if args.only:
        items = [i for i in items if i in args.only.split(",")]
    for item in items:
        v = videos[item]
        dest = CACHE / "{}.json".format(item)
        if dest.exists() and not args.force:
            out("{:<4} cached".format(item))
            continue
        frs = frames.get(v["asset_id"], [])
        bad = spans.get(v["asset_id"], [])
        usable = [f for f in frs
                  if not any(a - 1e-9 <= f["t"] < b for a, b in bad)]
        t0 = time.time()
        rec = {"item": item, "asset_id": v["asset_id"],
               "engine": E.engine_version(cfg), "frames_total": len(frs),
               "frames_skipped_unusable": len(frs) - len(usable),
               "frames": [], "error": None}
        try:
            imgs, scores, prev = [], [], None
            for f in usable:
                data = store.get(f["path"])
                if hashlib.sha256(data).hexdigest() != f["sha"]:
                    raise RuntimeError("keyframe {} hash mismatch".format(f["path"]))
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                scores.append(E.change_score(prev, img))
                imgs.append(img)
                prev = img
            idx = O.coverage_frames([f["t"] for f in usable], cfg, scores)
            rec["frames_read"] = len(idx)
            for i in idx:
                f, img = usable[i], imgs[i]
                rec["frames"].append({
                    "keyframe_id": f["keyframe_id"], "scene_id": f["scene_id"],
                    "timestamp": f["t"], "width": img.shape[1],
                    "height": img.shape[0],
                    "variants": E.read_frame(img, engine, cfg)})
            del imgs
        except Exception as exc:
            rec["error"] = "{}: {}".format(type(exc).__name__, str(exc)[:300])
        rec["seconds"] = round(time.time() - t0, 1)
        dest.write_text(json.dumps(rec), encoding="utf-8")
        out("{:<4} {:>3}/{:<3} frames read ({} unusable skipped) {:>6.0f}s{}".format(
            item, rec.get("frames_read", 0), len(frs),
            rec["frames_skipped_unusable"], rec["seconds"],
            "  ERROR " + rec["error"] if rec["error"] else ""))
    return 0


# --------------------------------------------------------------------------
# backup
# --------------------------------------------------------------------------

def stage_backup(conn, args, out):
    with conn.cursor() as cur:
        cur.execute("select to_regclass(%s)", ("content_v1." + BACKUP_TABLE,))
        if cur.fetchone()[0] is not None:
            cur.execute("select count(*) from content_v1." + BACKUP_TABLE)
            out("{} exists with {} rows; left alone".format(
                BACKUP_TABLE, cur.fetchone()[0]))
            conn.rollback()
            return 0
        cur.execute("select count(*) from content_v1.ocr_observations")
        n = cur.fetchone()[0]
        if not args.commit:
            out("would copy {} rows to {}".format(n, BACKUP_TABLE))
            conn.rollback()
            return 0
        cur.execute("create table content_v1.{} as select * from "
                    "content_v1.ocr_observations".format(BACKUP_TABLE))
        cur.execute("alter table content_v1.{} enable row level security"
                    .format(BACKUP_TABLE))
        cur.execute("select count(*) from content_v1." + BACKUP_TABLE)
        got = cur.fetchone()[0]
        if got != n:
            raise SystemExit("backup copied {} of {}".format(got, n))
    conn.commit()
    out("backed up {} rows to content_v1.{} (RLS on)".format(got, BACKUP_TABLE))
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def digests(cur, tables):
    out = {}
    for t in tables:
        cur.execute("select count(*), md5(coalesce(string_agg(x::text, '|' "
                    "order by x::text), '')) from content_v1.{} x".format(t))
        n, md5 = cur.fetchone()
        out[t] = {"rows": n, "md5": md5}
    return out


def build_for(rec, cfg, lexicon, treatments, aliases):
    per_frame = [{"index": i, "timestamp": fr["timestamp"],
                  "variants": fr["variants"]}
                 for i, fr in enumerate(rec["frames"])]
    if not rec["frames"]:
        return [], {"accepted": 0, "lines": 0}
    meta = {"width": rec["frames"][0]["width"],
            "height": rec["frames"][0]["height"]}
    return O.build_observations(per_frame, meta, cfg, lexicon, usable=True,
                                treatments=treatments, aliases=aliases)


def stage_run(conn, args, out, cfg_db):
    import step2_analyse as s2
    from preflight import EXPECTED_CONFIG_ROWS
    videos, frames, spans, treatments, aliases, key = load_inputs(conn)
    missing = [i for i in videos if not (CACHE / "{}.json".format(i)).exists()]
    if missing:
        raise SystemExit("capture first; no cache for {}".format(missing))
    with conn.cursor() as cur:
        cur.execute("select to_regclass(%s)", ("content_v1." + BACKUP_TABLE,))
        if cur.fetchone()[0] is None:
            raise SystemExit("back up first")

        before_cfg = {k: v for k, v in cfg_db.items() if k.startswith("ocr_")}
        before_digest = digests(cur, UNTOUCHED)
        # score what is stored now, under the NEW scorer config, so before
        # and after are measured with the same ruler
        cfg_new = planned_config(cfg_db)
        before_scores = S.score_stored(conn, cfg_new, rollback=False)

        # ---- config ---------------------------------------------------
        config_log = []
        for k, value, vtype, lo, hi, desc in changes():
            old = cfg_db.get(k, "__absent__")
            if value is None:
                cur.execute("delete from content_v1.pipeline_config where key = %s", (k,))
                config_log.append({"key": k, "old": old, "new": "__deleted__"})
                continue
            cur.execute("""
                insert into content_v1.pipeline_config
                  (key, value, value_type, description, min_value, max_value)
                values (%s, %s::jsonb, %s, %s, %s, %s)
                on conflict (key) do update set value = excluded.value,
                  value_type = excluded.value_type,
                  description = excluded.description,
                  min_value = excluded.min_value,
                  max_value = excluded.max_value, updated_at = now()""",
                (k, json.dumps(value), vtype, desc, lo, hi))
            if old != json.loads(json.dumps(value)):
                config_log.append({"key": k, "old": old, "new": value})
        cur.execute("select count(*) from content_v1.pipeline_config")
        cfg_rows = cur.fetchone()[0]
        cfg = s2.load_db_config(conn)
        O.audit_config(cfg)
        fingerprint, how = s2.take_snapshot(conn, CONFIG_VERSION, SNAPSHOT_NOTE)

        # ---- observations, scored twice ------------------------------
        lexicon = s2.build_lexicon(cfg, treatments)
        terms = O.known_terms(cfg, treatments, aliases)
        words = O.load_words(cfg)
        raw_cfg = O.without_normalisation(cfg)
        results = {}
        for item in sorted(videos, key=item_key):
            v = videos[item]
            rec = json.loads((CACHE / "{}.json".format(item)).read_text(encoding="utf-8"))
            exp = key.get(v["asset_id"])
            if rec["error"]:
                results[item] = {"status": "FAILED", "error_code": "OCR_ENGINE_ERROR",
                                 "message": rec["error"], "rows": [], "score": None,
                                 "score_raw": None, "rec": rec}
                continue
            rows, summary = build_for(rec, cfg, lexicon, treatments, aliases)
            rows_raw, _ = build_for(rec, raw_cfg, lexicon, treatments, aliases)
            acc = [r for r in rows if r["accepted_for_search"]]
            acc_raw = [r for r in rows_raw if r["accepted_for_search"]]
            sc = S.score(exp, acc, terms, words, cfg)
            sc_raw = S.score(exp, acc_raw, terms, words, cfg)
            results[item] = {"status": sc["status"], "error_code": sc["error_code"],
                             "message": sc["message"], "rows": rows,
                             "summary": summary, "score": sc, "score_raw": sc_raw,
                             "rec": rec}

        # ---- write ---------------------------------------------------
        now = dt.datetime.now(dt.timezone.utc)
        for item in sorted(videos, key=item_key):
            v, res = videos[item], results[item]
            rec = res["rec"]
            cur.execute("""select coalesce(max(attempt_number), 0) + 1
                           from content_v1.analysis_runs
                           where asset_id = %s and step = 'STEP2_OCR'
                             and config_version = %s""",
                        (v["asset_id"], CONFIG_VERSION))
            attempt = cur.fetchone()[0]
            meta = {
                "item": item, "engine": rec["engine"],
                "engine_benchmark_choice": cfg["ocr_engine_benchmark"]["chosen"],
                "frames_total": rec["frames_total"],
                "frames_read": rec.get("frames_read"),
                "frames_skipped_unusable": rec["frames_skipped_unusable"],
                "capture_seconds": rec["seconds"],
                "score": strip(res["score"]),
                "score_without_normalisation": strip(res["score_raw"]),
                "rejected_by": (res.get("summary") or {}).get("rejected_by"),
                "normalisations": (res.get("summary") or {}).get("normalisations"),
            }
            cur.execute("""
                insert into content_v1.analysis_runs
                  (asset_id, step, status, processor_version, config_version,
                   source_fingerprint, attempt_number, started_at, completed_at,
                   error_code, error_message, metadata)
                values (%s, 'STEP2_OCR', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning id""",
                (v["asset_id"], res["status"], PROCESSOR_VERSION, CONFIG_VERSION,
                 v["hash"], attempt, now, now, res["error_code"],
                 res["message"], json.dumps(meta, default=str)))
            run_id = cur.fetchone()[0]
            if res["status"] == "FAILED":
                continue      # keep the previous rows; nothing new to store
            cur.execute("delete from content_v1.ocr_observations where asset_id = %s",
                        (v["asset_id"],))
            frames_by_index = {i: fr for i, fr in enumerate(rec["frames"])}
            for r in res["rows"]:
                fr = frames_by_index[r["_frame_index"]]
                cur.execute("""
                    insert into content_v1.ocr_observations
                      (asset_id, scene_id, keyframe_id, timestamp_seconds,
                       raw_text, normalized_text, text_type, language,
                       confidence, bounding_box, accepted_for_search,
                       rejection_reason, provider, model, analysis_run_id,
                       config_version)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (v["asset_id"], fr["scene_id"], fr["keyframe_id"],
                     r["timestamp"], r["raw_text"], r["normalized_text"],
                     r["text_type"], None, r["confidence"],
                     json.dumps(dict(r["bounding_box"],
                                     variants=r["variants"],
                                     frames_seen=r["frames_seen"],
                                     block_lines=r["caption_block_lines"],
                                     normalisation=r["normalisation"])),
                     r["accepted_for_search"], r["rejection_reason"],
                     cfg["ocr_engine"], rec["engine"], run_id, CONFIG_VERSION))

        after_digest = digests(cur, UNTOUCHED)
        after_scores = S.score_stored(conn, cfg, rollback=False)

    plan = {
        "config_version": CONFIG_VERSION, "fingerprint": fingerprint,
        "snapshot": how, "config_rows": cfg_rows, "config_log": config_log,
        "before_ocr_config": before_cfg,
        "before": {i: strip(s) for i, s in before_scores.items()},
        "after_stored": {i: strip(s) for i, s in after_scores.items()},
        "results": {i: {"status": r["status"], "error_code": r["error_code"],
                        "message": r["message"], "score": strip(r["score"]),
                        "score_raw": strip(r["score_raw"]),
                        "rows": len(r["rows"]),
                        "accepted": sorted({x["normalized_text"] for x in r["rows"]
                                            if x["accepted_for_search"]}),
                        "frames_read": r["rec"].get("frames_read"),
                        "frames_total": r["rec"]["frames_total"],
                        "normalisations": sorted(
                            {json.dumps({k: n[k] for k in ("kind", "from", "to")},
                                        sort_keys=True)
                             for n in (r.get("summary") or {}).get("normalisations") or []}),
                        "rejected_by": (r.get("summary") or {}).get("rejected_by")}
                    for i, r in results.items()},
        "untouched_before": before_digest, "untouched_after": after_digest,
    }
    plan["problems"] = check(plan, EXPECTED_CONFIG_ROWS)
    sha = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str)
                         .encode()).hexdigest()
    summary_lines(out, plan)
    out("plan sha256 {}".format(sha))
    if args.plan:
        Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
        Path(args.plan).write_bytes(json.dumps(plan, sort_keys=True, indent=1,
                                               default=str).encode())
    if plan["problems"]:
        conn.rollback()
        out("CHECKS FAILED - rolled back")
        for p in plan["problems"]:
            out("  " + p)
        return 1
    if not args.commit:
        conn.rollback()
        out("DRY RUN: every statement ran and every check passed; rolled back")
        return 0
    if sha != args.expect_sha:
        conn.rollback()
        raise SystemExit("plan differs from the dry runs:\n  expected {}\n  now {}"
                         .format(args.expect_sha, sha))
    conn.commit()
    out("COMMITTED")
    return 0


def strip(s):
    if not s:
        return s
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in s.items()}


def check(plan, expected_rows):
    p = []
    if plan["untouched_before"] != plan["untouched_after"]:
        p.append("tables outside STEP2_OCR changed: {}".format(sorted(
            t for t in plan["untouched_after"]
            if plan["untouched_after"][t] != plan["untouched_before"][t])))
    if plan["config_rows"] != expected_rows:
        p.append("pipeline_config has {} rows; preflight expects {}".format(
            plan["config_rows"], expected_rows))
    for item, r in plan["results"].items():
        stored = plan["after_stored"].get(item) or {}
        if r["status"] != "FAILED" and stored.get("status") != r["status"]:
            p.append("{}: run status {} but the stored rows score {}".format(
                item, r["status"], stored.get("status")))
        sc = r["score"] or {}
        if r["status"] == "COMPLETED" and (sc.get("missed") or sc.get("garbage")):
            p.append("{}: COMPLETED with missed/garbage".format(item))
    return p


def summary_lines(out, plan):
    out("config {} ({}) {} rows; {} key(s) changed".format(
        plan["config_version"], plan["snapshot"], plan["config_rows"],
        len(plan["config_log"])))
    f = (lambda v: "  - " if v is None else "{:.2f}".format(v))
    out("{:<4} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} | {:<10} {}".format(
        "item", "recall", "strict", "prec", "conf", "raw_r", "raw_p",
        "status", "missed / garbage / unexpected"))
    for item in sorted(plan["results"], key=item_key):
        r = plan["results"][item]
        s, sr = r["score"] or {}, r["score_raw"] or {}
        out("{:<4} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} | {:<10} {}".format(
            item, f(s.get("recall")), f(s.get("strict_recall")),
            f(s.get("precision")), f(s.get("mean_confidence")),
            f(sr.get("recall")), f(sr.get("precision")), r["status"],
            "; ".join((s.get("missed") or []) + (s.get("garbage") or [])
                      + (s.get("false_positives") or []))[:110]))
    same = plan["untouched_before"] == plan["untouched_after"]
    out("frames/scenes/transcripts/quality/answer key unchanged: {}".format(same))


# --------------------------------------------------------------------------

def stage_verify(conn, args, out, cfg_db):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    problems = []
    if cfg_db.get("ocr_engine") != "rapidocr":
        problems.append("config not migrated")
    with conn.cursor() as cur:
        cur.execute("select fingerprint from content_v1.config_snapshots "
                    "where config_version = %s", (CONFIG_VERSION,))
        row = cur.fetchone()
        if not row or row[0] != plan["fingerprint"]:
            problems.append("snapshot missing or different")
        cur.execute("""
            select sf.metadata->>'step1_item', r.status, r.error_code,
                   r.config_version, r.processor_version
            from content_v1.analysis_runs r
            join content_v1.asset_sources s on s.asset_id = r.asset_id
                 and s.is_first_discovered_source
            join content_v1.source_files sf on sf.id = s.source_file_id
            where r.step = 'STEP2_OCR' and r.config_version = %s""",
                    (CONFIG_VERSION,))
        runs = {i: (st, code, cv, pv) for i, st, code, cv, pv in cur.fetchall()}
        cur.execute("select count(*) from content_v1.ocr_observations "
                    "where config_version <> %s", (CONFIG_VERSION,))
        stale = cur.fetchone()[0]
        cur.execute("select count(*), bool_and(relrowsecurity) from pg_class "
                    "where relname = %s and relnamespace = 'content_v1'::regnamespace",
                    (BACKUP_TABLE,))
        bk = cur.fetchone()
        now = digests(cur, UNTOUCHED)
    conn.rollback()
    for item, r in plan["results"].items():
        got = runs.get(item)
        if not got or got[0] != r["status"] or got[1] != r["error_code"]:
            problems.append("{}: run {} vs plan {}".format(item, got, r["status"]))
    if stale:
        problems.append("{} observation rows not on {}".format(stale, CONFIG_VERSION))
    if tuple(bk) != (1, True):
        problems.append("backup table missing or RLS off")
    if now != plan["untouched_before"]:
        problems.append("tables outside STEP2_OCR changed since the dry run")
    stored = S.score_stored(conn, cfg_db)
    for item, s in stored.items():
        if strip(s) != plan["after_stored"][item]:
            problems.append("{}: stored score differs from the plan".format(item))
    out("runs: {}  statuses: {}".format(len(runs), sorted(
        {v[0] for v in runs.values()})))
    out("VERIFY {}".format("PASSED" if not problems else "FAILED"))
    for p in problems:
        out("  " + p)
    return 0 if not problems else 1


# --------------------------------------------------------------------------

def stage_report(conn, args, out, cfg_db):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    bench = benchmark_summary()
    L = []
    A = L.append
    f = (lambda v: "-" if v is None else "{:.2f}".format(v))
    A("# OCR results - STEP2_OCR 3.0.0")
    A("")
    A("- Config version `{}` (snapshot fingerprint `{}`), processor `{}`.".format(
        plan["config_version"], plan["fingerprint"][:16], PROCESSOR_VERSION))
    A("- Previous rows backed up to `content_v1.{}`.".format(BACKUP_TABLE))
    A("- Scope: the 15 videos. Photos are out of scope - none has an expected "
      "on-screen text in the answer key and their STEP2_OCR runs hold no rows; "
      "they were not re-run.")
    A("- Before = the rows stored by `step2_ocr/2.0.0` (`step2_2026_09_23_e`), "
      "scored with the same scorer as after.")
    A("- *With normalisation* is what is stored. *Without* re-runs the same "
      "readings with known-term normalisation and dictionary correction off - "
      "what the engine really read.")
    A("")
    A("## Per video")
    A("")
    A("| Item | Before: recall / strict / precision / conf / status | After (stored): recall / strict / precision / conf | Status | Without normalisation: recall / strict / precision | Missed (after) |")
    A("|---|---|---|---|---|---|")
    for item in sorted(plan["results"], key=item_key):
        b = plan["before"].get(item) or {}
        r = plan["results"][item]
        s, sr = r["score"] or {}, r["score_raw"] or {}
        A("| {} | {} / {} / {} / {} / {} | {} / {} / {} / {} | **{}** {} | {} / {} / {} | {} |".format(
            item, f(b.get("recall")), f(b.get("strict_recall")),
            f(b.get("precision")), f(b.get("mean_confidence")), b.get("status"),
            f(s.get("recall")), f(s.get("strict_recall")), f(s.get("precision")),
            f(s.get("mean_confidence")), r["status"], r["error_code"] or "",
            f(sr.get("recall")), f(sr.get("strict_recall")), f(sr.get("precision")),
            "; ".join(s.get("missed") or []) or "-"))
    A("")
    A("## What was accepted, per video (after)")
    A("")
    for item in sorted(plan["results"], key=item_key):
        r = plan["results"][item]
        A("- **{}** ({} of {} keyframes read): {}".format(
            item, r["frames_read"], r["frames_total"],
            ", ".join("`{}`".format(a) for a in r["accepted"]) or "nothing"))
        if r["normalisations"]:
            A("  - normalisations: {}".format(", ".join(
                "{from} -> {to} ({kind})".format(**json.loads(n))
                for n in r["normalisations"])))
    A("")
    A("## Engine benchmark (recorded in `ocr_engine_benchmark`)")
    A("")
    A("| Engine @ scale | Terms read (strict) | Read ignoring spacing | Noise on text-free videos | s/frame |")
    A("|---|---:|---:|---:|---:|")
    for tag, r in sorted(bench.items()):
        A("| {} | {}/{} | {} | {} | {} |".format(
            tag, r["read"], r["expected"], r.get("read_ignoring_spacing", "-"),
            r.get("noise", "-"), r["s_per_frame"]))
    A("")
    A("## Config keys changed")
    A("")
    A("| Key | Old | New |")
    A("|---|---|---|")
    for c in plan["config_log"]:
        A("| `{}` | `{}` | `{}` |".format(
            c["key"], short(c["old"]), short(c["new"])))
    A("")
    if args.extra and Path(args.extra).exists():
        A(Path(args.extra).read_text(encoding="utf-8").strip())
        A("")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    out("wrote {}".format(REPORT))
    return 0


def short(v):
    if v == "__absent__":
        return "(new key)"
    if v == "__deleted__":
        return "(deleted)"
    s = json.dumps(v)
    return s if len(s) <= 90 else s[:87] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["capture", "backup", "run", "verify", "report"])
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--plan")
    ap.add_argument("--expect-sha")
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra")
    args = ap.parse_args()
    if args.stage == "run" and args.commit and not args.expect_sha:
        raise SystemExit("--commit needs --expect-sha")
    import step2_analyse as s2
    from db import Reconnecting
    handle = Reconnecting()
    conn = handle.ensure()
    cfg = s2.load_db_config(conn)
    conn.rollback()
    out = print
    try:
        fn = {"capture": stage_capture, "run": stage_run, "verify": stage_verify,
              "report": stage_report}.get(args.stage)
        if args.stage == "backup":
            return stage_backup(conn, args, out)
        return fn(conn, args, out, cfg)
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
