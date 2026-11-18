#!/usr/bin/env python3
"""
Step 2 for STILL images (content_v1): one scene, one keyframe, one stored
JPEG, one quality score per photo - so Step 3 can treat a photo exactly as a
one-frame video.

Before this, the 9 content_v1 photos had no scene, no keyframe and no stored
image: Step 2 only ever ran OCR on stills. Per photo, from the ORIGINAL bytes
(downloaded from Drive, sha256 checked against assets.content_hash):

  image    decoded (HEIC via pillow_heif), EXIF orientation applied, RGB,
           long side capped at frame_width_px (the video keyframe box),
           JPEG at frame_jpeg_quality -> keyframes/{asset_id}/000000_0.000.jpg
           (read back and sha256-checked by KeyframeStore)
  quality  step2_quality.measure_frame / score_frame on the stored JPEG,
           exactly as video keyframes are scored (no previous frame)
  rows     one transaction per photo:
           STEP2_SCENES run      + asset_scenes (index 0, 0-1 s,
                                   detection_method 'still_image',
                                   boundary_source 'start')
           STEP2_DERIVATIVES run + asset_derivatives (KEYFRAME)
           STEP2_KEYFRAMES run   + asset_keyframes (t=0, is_representative,
                                   technical_quality_score)
           STEP2_QUALITY run     (score in metadata)

One row set per ASSET: DSC03426.JPG example.jpg are the same bytes and
the same asset, so they get one scene. A photo that already has a scene is
refused, never duplicated.

    python scripts_content_v1/step2_stills.py              # dry run: decode, score, no writes
    python scripts_content_v1/step2_stills.py --commit
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import target  # noqa: E402

PROCESSOR_VERSION = "step2_stills/1.0.0"
DEFAULT_CONFIG_VERSION = "content_v1_step2_2026_09_30_a"
STORAGE_NAME = "000000_0.000.jpg"

STILLS_SQL = """
select a.id::text, a.file_name, a.content_hash, a.mime_type,
       sf.google_file_id, sf.relative_path, sf.metadata->>'step1_item',
       (select count(*) from content_v1.asset_scenes x where x.asset_id = a.id),
       f.source_kind, f.root_path
from content_v1.assets a
join content_v1.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
join content_v1.source_files sf on sf.id = s.source_file_id
join content_v1.source_folders f on f.id = sf.source_folder_id
where a.media_kind = 'IMAGE'
order by a.file_name
"""


def render(path, cfg):
    """Original file -> (jpeg bytes, width, height, orientation note)."""
    from PIL import Image, ImageOps
    import pillow_heif
    pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        orientation = (im.getexif() or {}).get(0x0112)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        cap = int(cfg["frame_width_px"])
        w, h = im.size
        scale = min(1.0, float(cap) / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(cfg["frame_jpeg_quality"]),
                optimize=True)
        return buf.getvalue(), im.size[0], im.size[1], orientation


def quality(jpeg, cfg):
    import cv2
    import numpy as np
    import step2_quality as Q
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    orb = cv2.ORB_create(nfeatures=800, nlevels=8, fastThreshold=7)
    matcher = {"orb": orb, "bf": cv2.BFMatcher(cv2.NORM_HAMMING)}
    m, _grey = Q.measure_frame(img, cfg, None, matcher)
    score, sub, reason = Q.score_frame(m, cfg)
    return score, reason


def run_row(cur, asset, step, status, cv, started, completed, meta):
    cur.execute("""select coalesce(max(attempt_number), 0) + 1
                   from content_v1.analysis_runs
                   where asset_id = %s and step = %s""", (asset["id"], step))
    attempt = cur.fetchone()[0]
    xc, xv, xp = target.run_columns(PROCESSOR_VERSION, status)
    cur.execute("""
        insert into content_v1.analysis_runs
          (asset_id, step, status, processor_version, config_version,
           source_fingerprint, attempt_number, started_at, completed_at,
           metadata{xc})
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s{xv}) returning id""".format(xc=xc, xv=xv),
        (asset["id"], step, status, PROCESSOR_VERSION, cv,
         asset["content_hash"], attempt, started, completed,
         json.dumps(meta, default=str)) + xp)
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--config-version", default=DEFAULT_CONFIG_VERSION)
    ap.add_argument("--part", type=int, default=0,
                    help="with --parts: take only share `part` of the stills, so "
                         "several runs side by side never take the same one")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--last", action="append", default=[],
                    help="relative_path prefix whose stills go last (repeatable)")
    ap.add_argument("--skip-done", action="store_true",
                    help="leave stills that already have a scene out of the list "
                         "instead of printing REFUSED for each")
    args = ap.parse_args()

    import step1_ingest as s1
    import step2_analyse as s2
    from db import Reconnecting
    from storage import KeyframeStore

    handle = Reconnecting()
    conn = handle.ensure()
    cfg = s2.load_db_config(conn)
    with conn.cursor() as cur:
        cur.execute(STILLS_SQL)
        stills = [dict(zip(("id", "file_name", "content_hash", "mime",
                            "google_file_id", "relative_path", "item",
                            "scenes", "source_kind", "root_path"), r))
                  for r in cur.fetchall()]
        cur.execute("select count(*) from content_v1.config_snapshots "
                    "where config_version = %s", (args.config_version,))
        if cur.fetchone()[0] != 1:
            raise SystemExit("no config snapshot {!r}".format(args.config_version))
    conn.rollback()
    stills = [a for a in stills
              if int(hashlib.sha256(a["id"].encode()).hexdigest(), 16) % args.parts == args.part
              and not (args.skip_done and a["scenes"])]
    stills.sort(key=lambda a: any((a["relative_path"] or "").startswith(p) for p in args.last))
    if args.commit:
        fp, how = s2.take_snapshot(conn, args.config_version,
                                   "Step 2 stills: scene/keyframe/derivative/quality")
        conn.commit()
        print("config snapshot {} {} ({})".format(args.config_version, how, fp[:16]))

    drive = {}

    def get_service():
        # Only Drive originals need the service; NAS stills are copied.
        if "s" not in drive:
            drive["s"], _sa = s1.build_drive_service()
        return drive["s"]
    store = KeyframeStore(s2.need(cfg, "keyframe_storage_bucket"),
                          verify_readback=True) if args.commit else None
    tmp = PROJECT_ROOT / "work" / "tmp" / "stills"   # work/ is gitignored
    tmp.mkdir(parents=True, exist_ok=True)

    print("{} - {} still image(s)".format(
        "COMMIT" if args.commit else "DRY RUN (no upload, no rows)", len(stills)))
    failures = 0
    for a in stills:
        if a["scenes"]:
            print("  {:<14} REFUSED: already has {} scene(s)".format(
                a["file_name"], a["scenes"]))
            continue
        started = dt.datetime.now(dt.timezone.utc)
        src = tmp / "{}{}".format(a["id"], Path(a["file_name"]).suffix)
        try:
            s2.fetch_original(s1, get_service, a, src)
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            if sha != a["content_hash"]:
                raise RuntimeError("downloaded bytes do not match content_hash")
            jpeg, w, h, orientation = render(src, cfg)
            score, reason = quality(jpeg, cfg)
            jsha = hashlib.sha256(jpeg).hexdigest()
            path = "{}/{}".format(a["id"], STORAGE_NAME)
            print("  {:<14} {:>4}x{:<4} {:>7} B  exif_orientation={}  quality={:.3f}{}"
                  .format(a["file_name"], w, h, len(jpeg), orientation, score,
                          " ({})".format(reason) if reason else ""))
            if not args.commit:
                continue
            back = store.put(path, jpeg)
            if back != jsha:
                raise RuntimeError("stored object hash differs")
            completed = dt.datetime.now(dt.timezone.utc)
            cv = args.config_version
            meta = {"item": a["item"], "source": a["relative_path"],
                    "exif_orientation": orientation, "width_px": w,
                    "height_px": h, "long_side_cap_px": int(cfg["frame_width_px"])}
            try:
                with conn.cursor() as cur:
                    r_sc = run_row(cur, a, "STEP2_SCENES", "COMPLETED", cv,
                                   started, completed,
                                   dict(meta, scenes=1, rule="still_image"))
                    cur.execute("""
                        insert into content_v1.asset_scenes
                          (asset_id, scene_index, start_seconds, end_seconds,
                           detection_method, boundary_source,
                           analysis_run_id, config_version)
                        values (%s,0,0,1,'still_image','start',%s,%s)
                        returning id""", (a["id"], r_sc, cv))
                    scene_id = cur.fetchone()[0]
                    r_dv = run_row(cur, a, "STEP2_DERIVATIVES", "COMPLETED", cv,
                                   started, completed,
                                   dict(meta, storage_path=path, sha256=jsha))
                    cur.execute("""
                        insert into content_v1.asset_derivatives
                          (asset_id, derivative_type, storage_bucket,
                           storage_path, mime_type, file_size_bytes, width_px,
                           height_px, content_hash, generation_status,
                           analysis_run_id, config_version)
                        values (%s,'KEYFRAME',%s,%s,'image/jpeg',%s,%s,%s,%s,
                                'READY',%s,%s) returning id""",
                        (a["id"], store.bucket, path, len(jpeg), w, h, jsha,
                         r_dv, cv))
                    deriv_id = cur.fetchone()[0]
                    r_kf = run_row(cur, a, "STEP2_KEYFRAMES", "COMPLETED", cv,
                                   started, completed, dict(meta, keyframes=1))
                    cur.execute("""
                        insert into content_v1.asset_keyframes
                          (asset_id, scene_id, timestamp_seconds, frame_index,
                           image_status, derivative_id, selection_reason,
                           technical_quality_score, is_representative,
                           analysis_run_id, config_version)
                        values (%s,%s,0,0,'STORED',%s,'still_image:whole_photo',
                                %s,true,%s,%s)""",
                        (a["id"], scene_id, deriv_id, score, r_kf, cv))
                    run_row(cur, a, "STEP2_QUALITY", "COMPLETED", cv, started,
                            completed, dict(meta, frames=1, min_score=score,
                                            median_score=score, mean_score=score,
                                            spans=0, unusable_share=0.0,
                                            reason=reason))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        except Exception as exc:
            failures += 1
            print("  {:<14} FAILED: {}: {}".format(a["file_name"],
                                                    type(exc).__name__, str(exc)[:200]))
        finally:
            if src.exists():
                src.unlink()
    handle.close()
    print("failures: {}".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
