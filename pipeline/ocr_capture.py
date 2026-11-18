#!/usr/bin/env python3
"""
Read the stored keyframes with easyocr and cache every reading.

READ ONLY: selects from content_v1, GETs nothing from storage that it does not
already have locally, and writes only a JSON cache. Nothing goes to the
database.

The OCR itself is by far the slowest thing in this pipeline - about 5 seconds
for a 1280x720 frame and about 20 for the same frame enlarged - so it is done
once and cached. Everything downstream (pooling, line grouping, persistence,
vocabulary matching, acceptance) is fast, deterministic and re-runnable
against this cache, which is what makes two identical dry runs and then a
commit affordable.

Frames inside an unusable span (content_v1.asset_quality_spans) are not read at
all: there is no text to find where the camera was pointed at clothing.

    python scripts/ocr_capture.py --cache <frames dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


SQL = """
select sf.metadata->>'step1_item' as item, a.id as asset_id, a.file_name,
       k.id as keyframe_id, k.scene_id, k.timestamp_seconds,
       k.technical_quality_score
from content_v1.asset_keyframes k
join content_v1.assets a on a.id = k.asset_id
join content_v1.asset_sources s on s.asset_id = a.id
     and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
order by (regexp_replace(sf.metadata->>'step1_item', '\\D', '', 'g'))::int,
         k.timestamp_seconds
"""

SPANS = """
select sf.metadata->>'step1_item' as item, q.start_seconds, q.end_seconds
from content_v1.asset_quality_spans q
join content_v1.assets a on a.id = q.asset_id
join content_v1.asset_sources s on s.asset_id = a.id
     and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
"""


def item_num(i):
    return int(re.sub(r"\D", "", i or "") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--upscale", type=float, default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    import cv2
    import easyocr
    import step2_analyse as s2
    import step2_ocr as O
    from db import Reconnecting

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    handle = Reconnecting()
    conn = handle.ensure()
    cfg = s2.load_db_config(conn)
    with conn.cursor() as cur:
        cur.execute(SQL)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute(SPANS)
        spans = {}
        for it, a, b in cur.fetchall():
            spans.setdefault(it, []).append((float(a), float(b)))
    conn.rollback()
    handle.close()

    by_item = {}
    for r in rows:
        by_item.setdefault(r["item"], []).append(r)

    reader = easyocr.Reader(list(cfg["ocr_languages"]), gpu=False,
                            verbose=False)

    def read(img):
        out = []
        for bbox, txt, conf in reader.readtext(img):
            t = (txt or "").strip()
            if t:
                out.append({"raw_text": t, "confidence": round(float(conf), 4),
                            "points": [[float(p[0]), float(p[1])]
                                       for p in bbox]})
        return out

    def scale_points(readings, f):
        for r in readings:
            r["points"] = [[p[0] / f, p[1] / f] for p in r["points"]]
        return readings

    items = sorted(by_item, key=item_num)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        items = [i for i in items if i in keep]

    for item in items:
        dest = out_dir / "{}.json".format(item)
        if dest.exists():
            print("{:<5} cached".format(item), flush=True)
            continue
        frs = sorted(by_item[item], key=lambda r: float(r["timestamp_seconds"]))
        bad = spans.get(item, [])

        def unusable(t):
            return any(a - 1e-9 <= t < b for a, b in bad)

        keep_frames = [r for r in frs if not unusable(float(r["timestamp_seconds"]))]
        skipped = len(frs) - len(keep_frames)
        quality = [{"sharpness": float(r["technical_quality_score"] or 0.0)}
                   for r in keep_frames]
        idx = O.choose_frames(keep_frames, quality, cfg, args.max_frames)
        chosen = [keep_frames[i] for i in idx]

        upscale = (args.upscale if args.upscale
                   else float(cfg["ocr_upscale_factor"]))
        rec = {"item": item, "file_name": frs[0]["file_name"],
               "asset_id": str(frs[0]["asset_id"]),
               "frames_total": len(frs), "frames_skipped_unusable": skipped,
               "frames_read": len(chosen), "upscale": upscale,
               "frames": []}
        t0 = time.time()
        for r in chosen:
            t = float(r["timestamp_seconds"])
            p = cache / item / "{:06d}.jpg".format(int(round(t)))
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit("missing cached keyframe {}".format(p))
            h, w = img.shape[:2]
            variants = {"native": read(img)}
            # The enlarged pass costs four times what the native pass costs,
            # so it is spent where it can help: on a frame that actually
            # contains SMALL text. Enlarging a frame whose only text is a
            # 60-pixel caption adds nothing, while the device settings that
            # need it are 12 to 18 pixels tall. Measured on this library, the
            # frames carrying the missed reads all contain boxes well under
            # the threshold, so none of them is skipped.
            min_h = float(cfg["ocr_upscale_min_text_height_px"])
            heights = [max(p[1] for p in r["points"]) - min(p[1] for p in r["points"])
                       for r in variants["native"]]
            small = any(h < min_h for h in heights)
            if variants["native"] and small:
                up = cv2.resize(img, None, fx=upscale, fy=upscale,
                                interpolation=cv2.INTER_CUBIC)
                lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)
                cl = cv2.createCLAHE(clipLimit=2.0,
                                     tileGridSize=(8, 8)).apply(L)
                contrast = cv2.cvtColor(cv2.merge([cl, A, B]),
                                        cv2.COLOR_LAB2BGR)
                if bool(cfg.get("ocr_contrast_normalise", True)):
                    variants["contrast"] = scale_points(read(contrast), upscale)
                if bool(cfg.get("ocr_try_inverted", True)):
                    variants["inverted"] = scale_points(
                        read(cv2.bitwise_not(up)), upscale)
            rec["frames"].append({
                "keyframe_id": str(r["keyframe_id"]),
                "scene_id": str(r["scene_id"]) if r["scene_id"] else None,
                "timestamp": t, "width": w, "height": h,
                "variants": variants})
        rec["seconds"] = round(time.time() - t0, 1)
        dest.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print("{:<5} {:>3} frames read ({} skipped unusable)  {:>5.0f}s"
              .format(item, len(chosen), skipped, rec["seconds"]), flush=True)

    print("captured to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
