#!/usr/bin/env python3
"""
STEP2_OCR for STILL images (content_v1), the batch-1 method (user decision
12 Oct 2026, for batch 2 and every later still):

  engine   step2_analyse 1.0.0 process_image / easyocr, from the VERBATIM copy
           step2_analyse_v1_0_0_c94e765.py, with the OCR config of snapshot
           step2_2026_09_22_d - exactly what step2_new7_20261005.py ran for
           the batch-1 stills (5 Oct 2026)
  input    the STORED derivative of the still (keyframe bucket, sha256
           checked), never the Drive original
  rows     every ocr_observations row carries the still's one keyframe_id,
           its scene_id and timestamp_seconds 0. Batch 1's driver left these
           NULL (its 2 rows were never accepted for search, so Step 3 never
           read them); step3_vision needs a frame and a second per line.

Needs the still's Step 2 rows first (step2_stills.py: scene, derivative,
keyframe). A still that already has STEP2_OCR rows is refused unless
--replace: then its observations are copied to
content_v1.ocr_observations_backup_20261012 (RLS on) and deleted in the same
transaction as the new rows, and a new STEP2_OCR attempt is written.

    python scripts_content_v1/step2_stills_ocr.example.jpg     # dry run
    python scripts_content_v1/step2_stills_ocr.py --only B2-P02 --commit [--replace]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import target  # noqa: E402

PROCESSOR_VERSION = "step2_stills_ocr/1.0.0"
OCR_CV = "step2_2026_09_22_d"
BACKUP_TABLE = "ocr_observations_backup_20261012"
CODE_SOURCE = ("scripts_content_v1/step2_stills_ocr.py driving process_image of "
               "scripts_content_v1/step2_analyse_v1_0_0_c94e765.py (verbatim git c94e765), "
               "OCR config snapshot step2_2026_09_22_d")

STILL_SQL = """
select a.id::text, a.file_name, a.content_hash, sf.metadata->>'step1_item' as item,
       sf.relative_path, sf.parent_visible,
       k.id::text as keyframe_id, k.scene_id::text as scene_id,
       d.storage_bucket, d.storage_path, d.content_hash as deriv_sha, d.width_px, d.height_px,
       (select count(*) from content_v1.ocr_observations o where o.asset_id = a.id) as ocr_rows,
       (select count(*) from content_v1.analysis_runs r where r.asset_id = a.id
          and r.step = 'STEP2_OCR') as ocr_runs
from content_v1.assets a
join content_v1.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
left join content_v1.asset_keyframes k on k.asset_id = a.id and k.image_status = 'STORED'
left join content_v1.asset_derivatives d on d.id = k.derivative_id
where a.media_kind = 'IMAGE'
  and (a.file_name = any(%(names)s) or upper(sf.metadata->>'step1_item') = any(%(items)s))
order by sf.metadata->>'step1_item'
"""


def snapshot_cfg(cur, version):
    cur.execute("select payload from content_v1.config_snapshots where config_version = %s",
                (version,))
    row = cur.fetchone()
    if not row:
        raise SystemExit("no config snapshot {!r}".format(version))
    return {k: v["value"] for k, v in row[0].items()}


def run_row(cur, a, status, started, completed, meta, error_code=None, error_message=None):
    # attempt counted across every config version, as step2_writer does
    cur.execute("""select coalesce(max(attempt_number), 0) + 1 from content_v1.analysis_runs
                   where asset_id = %s and step = 'STEP2_OCR'""", (a["id"],))
    attempt = cur.fetchone()[0]
    xc, xv, xp = target.run_columns(PROCESSOR_VERSION, status)
    cur.execute("""
        insert into content_v1.analysis_runs
          (asset_id, step, status, processor_version, config_version, source_fingerprint,
           attempt_number, started_at, completed_at, error_code, error_message, metadata{xc})
        values (%s,'STEP2_OCR',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s{xv}) returning id""".format(xc=xc, xv=xv),
        (a["id"], status, PROCESSOR_VERSION, OCR_CV, a["content_hash"], attempt, started,
         completed, error_code, error_message, json.dumps(meta, default=str)) + xp)
    return cur.fetchone()[0]


def backup_and_clear(cur, aid):
    bk = target.backup_schema()
    cur.execute("""create table if not exists {bk}.{t}
                   (like content_v1.ocr_observations including defaults)""".format(bk=bk, t=BACKUP_TABLE))
    cur.execute("alter table {bk}.{t} enable row level security".format(bk=bk, t=BACKUP_TABLE))
    cur.execute("insert into {bk}.{t} select * from content_v1.ocr_observations "
                "where asset_id = %s".format(bk=bk, t=BACKUP_TABLE), (aid,))
    copied = cur.rowcount
    cur.execute("delete from content_v1.ocr_observations where asset_id = %s", (aid,))
    if cur.rowcount != copied:
        raise RuntimeError("backup copied {} rows, delete removed {}".format(copied, cur.rowcount))
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", required=True, help="still file names or step1 items")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--engine", choices=("easyocr", "apple_vision"), default="easyocr",
                    help="apple_vision: Apple's text recognition on the Neural Engine "
                         "(user decision 2 Nov 2026), through the same filters and rows")
    ap.add_argument("--replace", action="store_true",
                    help="replace existing OCR rows (backed up first)")
    args = ap.parse_args()

    import step2_analyse_v1_0_0_c94e765 as v1
    from db import Reconnecting
    from storage import KeyframeStore

    handle = Reconnecting()
    conn = handle.ensure()
    with conn.cursor() as cur:
        ocr_cfg = snapshot_cfg(cur, OCR_CV)
        cur.execute(STILL_SQL, {"names": args.only, "items": [x.upper() for x in args.only]})
        cols = [d.name for d in cur.description]
        stills = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.rollback()
    found = {s["file_name"] for s in stills} | {(s["item"] or "").upper() for s in stills}
    missing = [x for x in args.only if x not in found and x.upper() not in found]
    if missing:
        raise SystemExit("not a content_v1 still: {}".format(missing))
    if args.engine == "apple_vision":
        from apple_vision_ocr import VisionReader, ENGINE
        v1._OCR["r"] = VisionReader(ocr_cfg.get("ocr_languages") or ["en"])
        ocr_cfg = dict(ocr_cfg, ocr_engine=ENGINE)
    elif ocr_cfg.get("ocr_engine") != "easyocr":
        raise SystemExit("snapshot {} ocr_engine is {!r}, not easyocr".format(
            OCR_CV, ocr_cfg.get("ocr_engine")))
    # one folder per process: process_image always writes "ocr_input.jpg", so runs
    # side by side sharing a folder read each other's photos (found 2 Nov 2026)
    tmp = PROJECT_ROOT / "work" / "tmp" / "stills_ocr" / str(os.getpid())   # work/ is gitignored
    tmp.mkdir(parents=True, exist_ok=True)
    print("{} - {} still(s); OCR config {} ({}, step2_analyse 1.0.0 code)".format(
        "COMMIT" if args.commit else "DRY RUN (no rows)", len(stills), OCR_CV, ocr_cfg["ocr_engine"]))
    failures = 0
    for a in stills:
        label = "{:<8} {}".format(a["item"] or "-", a["file_name"])
        if not a["keyframe_id"] or not a["storage_path"]:
            print("  {}  FAILED: no stored keyframe - run step2_stills.py first".format(label))
            failures += 1
            continue
        if (a["ocr_rows"] or a["ocr_runs"]) and args.commit and not args.replace:
            print("  {}  REFUSED: already has STEP2_OCR ({} rows) - pass --replace".format(
                label, a["ocr_rows"]))
            failures += 1
            continue
        started = dt.datetime.now(dt.timezone.utc)
        store = KeyframeStore(a["storage_bucket"], verify_readback=False)
        data = store.get(a["storage_path"])
        if hashlib.sha256(data).hexdigest() != a["deriv_sha"]:
            print("  {}  FAILED: stored derivative does not match its sha256".format(label))
            failures += 1
            continue
        path = tmp / "{}_deriv.jpg".format(a["id"])
        path.write_bytes(data)
        try:
            out = v1.process_image({"display_width_px": a["width_px"],
                                    "display_height_px": a["height_px"]},
                                   path, tmp, ocr_cfg, do_ocr=True)
        except Exception as exc:                                   # recorded, not hidden
            out, err = None, "{}: {}".format(type(exc).__name__, str(exc)[:300])
        finally:
            path.unlink(missing_ok=True)
        completed = dt.datetime.now(dt.timezone.utc)
        path_parse = v1.parse_relative_path(a["relative_path"], a["parent_visible"])
        ocr_input = {"source": "stored_derivative", "keyframe_id": a["keyframe_id"],
                     "storage_bucket": a["storage_bucket"], "storage_path": a["storage_path"],
                     "sha256": a["deriv_sha"], "derivative_px": [a["width_px"], a["height_px"]]}
        if out is None:
            print("  {}  OCR FAILED: {}".format(label, err))
            failures += 1
            if args.commit:
                with conn.cursor() as cur:
                    run_row(cur, a, "FAILED", started, completed,
                            {"item": a["item"], "ocr_input": ocr_input,
                             "code_source": CODE_SOURCE, "ocr_coverage": "NONE"},
                            error_code="OCR_FAILED", error_message=err)
                conn.commit()
            continue
        obs = out["observations"]
        acc = [o for o in obs if o["accepted_for_search"]]
        print("  {}  {} obs, {} accepted{}: {}".format(
            label, len(obs), len(acc), " (replacing {} rows)".format(a["ocr_rows"])
            if a["ocr_rows"] else "", " | ".join(o["normalized_text"] or o["raw_text"]
                                                for o in acc)[:200]), flush=True)
        if not args.commit:
            continue
        full = out["ocr_coverage"] == "FULL"
        lang = (ocr_cfg.get("ocr_languages") or ["en"])[0]
        engine = ocr_cfg["ocr_engine"]
        meta = {"item": a["item"], "observations": len(obs), "engine": engine,
                "path_parse": path_parse, "ocr_frames_total": out["ocr_frames_total"],
                "ocr_frames_processed": out["ocr_frames_processed"],
                "ocr_coverage": out["ocr_coverage"],
                "ocr_degraded_reason": out["ocr_degraded_reason"],
                "ocr_input": dict(ocr_input, ocr_px=list(out["ocr_size"])),
                "code_source": CODE_SOURCE}
        try:
            with conn.cursor() as cur:
                if a["ocr_rows"]:
                    meta["replaced_rows_backed_up_to"] = "content_v1.{} ({} rows)".format(
                        BACKUP_TABLE, backup_and_clear(cur, a["id"]))
                run_id = run_row(cur, a, "COMPLETED", started, completed, meta,
                                 error_code=None if full else "OCR_DEGRADED",
                                 error_message=None if full else out["ocr_degraded_reason"])
                for ob in obs:
                    cur.execute("""
                        insert into content_v1.ocr_observations
                          (asset_id, scene_id, keyframe_id, timestamp_seconds, raw_text,
                           normalized_text, text_type, language, confidence, bounding_box,
                           accepted_for_search, rejection_reason, provider, model,
                           analysis_run_id, config_version)
                        values (%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (a["id"], a["scene_id"], a["keyframe_id"], ob["raw_text"],
                         ob["normalized_text"], ob["text_type"], lang, ob["confidence"],
                         json.dumps(ob["bounding_box"]), ob["accepted_for_search"],
                         ob["rejection_reason"], engine, engine, run_id, OCR_CV))
            conn.commit()
        except Exception as exc:
            conn.rollback()
            print("  {}  WRITE FAILED: {}: {}".format(label, type(exc).__name__, str(exc)[:300]))
            failures += 1
    handle.close()
    print("failures: {}".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
