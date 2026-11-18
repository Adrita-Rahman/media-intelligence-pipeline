#!/usr/bin/env python3
"""
desk_thumbs (10 Nov 2026): one small JPEG per scene for the Content Desk page, in ml_test.desk_thumbs.

The page is a claude.ai artifact: it can only show pictures that come out of the database (its
connector runs SQL; it cannot load images from Supabase Storage). So for every scene in
ml_test.desk_index without a thumbnail, this takes the scene's best keyframe that is not black or
blank (12 Nov) from the
ml-test-media bucket (through ml-test-media-upload with the upload token, as Step 2 does),
shrinks it to SIZE px on the long side and stores the JPEG bytes.

    venv/bin/python scripts_content_v1/desk_thumbs.py                 # every scene still without one
    venv/bin/python scripts_content_v1/desk_thumbs.py --limit 50      # a test
    venv/bin/python scripts_content_v1/desk_thumbs.py --redo-blank    # fix black / blank ones already made
    nohup caffeinate -i venv/bin/python scripts_content_v1/desk_thumbs.py > work/logs/desk_thumbs.log 2>&1 &

Resuming = running it again (only scenes without a thumbnail are taken). The nightly run calls it
after Step 4. Logs hold counts only - no file names, paths or keys.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PIPELINE_TARGET", "ml_test")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import Reconnecting  # noqa: E402
import storage  # noqa: E402
import target  # noqa: E402

SIZE = 240        # px, long side; 160 until 12 Nov 2026 (blurry when zoomed or on sharp screens)
QUALITY = 74       # 62 until 12 Nov 2026
CHUNK = 400       # scenes per database round
WORKERS = 12

# 12 Nov 2026: up to CANDIDATES keyframes per scene, best first, so a black or blank first
# choice (a fade-in, a dark cut) can be skipped for the next one - see make()
CANDIDATES = 8
DARK_MEAN = 20    # average brightness 0-255 below this = a black frame
FLAT_STD = 8      # brightness spread below this = a blank one-colour frame

CHOICES = """
cross join lateral (
  select array_agg(k.id::text order by k.n) ids, array_agg(k.storage_path order by k.n) paths from (
    select k.id, d.storage_path,
           row_number() over (order by k.vision_selected desc nulls last, k.is_representative desc nulls last,
                              k.technical_quality_score desc nulls last, k.timestamp_seconds) n
    from ml_test.asset_keyframes k
    join ml_test.asset_derivatives d on d.id = k.derivative_id
    where k.scene_id = i.scene_id and k.derivative_id is not null
    order by n limit {}) k) c
""".format(CANDIDATES)

PICK = """
select i.scene_id::text, c.ids, c.paths
from ml_test.desk_index i""" + CHOICES + """
where c.ids is not null
  and not exists (select 1 from ml_test.desk_thumbs t where t.scene_id = i.scene_id)
  and not (i.scene_id = any(%s::uuid[]))
limit %s"""

REPICK = """
select i.scene_id::text, c.ids, c.paths
from ml_test.desk_index i""" + CHOICES + """
where c.ids is not null and i.scene_id = any(%s::uuid[])"""


def blank(im):
    """True for a black or one-colour picture (needs Pillow image)."""
    from PIL import ImageStat
    st = ImageStat.Stat(im.convert("L"))
    return st.mean[0] < DARK_MEAN or st.stddev[0] < FLAT_STD


def log(msg):
    print("{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


_local = threading.local()


def store():
    if not hasattr(_local, "s"):
        _local.s = storage.KeyframeStore(target.profile()["keyframe_bucket"], verify_readback=False)
    return _local.s


def make(row):
    """(scene_id, [keyframe ids], [paths]) -> (scene_id, keyframe_id, jpeg bytes) or (scene_id, None, error).
    Takes the first candidate that is not black or blank; if all are, the one with the most detail."""
    from PIL import Image, ImageStat
    scene_id, ids, paths = row
    best = None   # (spread, keyframe id, image) - fallback when every candidate is blank
    err = "no keyframe"
    for kf, path in zip(ids, paths):
        im = None
        for attempt in range(3):
            try:
                data = store().get(path)
                im = Image.open(io.BytesIO(data))
                im.draft("RGB", (SIZE * 2, SIZE * 2))
                im = im.convert("RGB")
                break
            except Exception as exc:   # storage errors never carry the token (storage._safe)
                err = type(exc).__name__
                im = None
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        if im is None:
            continue
        if not blank(im):
            return scene_id, kf, _jpeg(im)
        spread = ImageStat.Stat(im.convert("L")).stddev[0]
        if best is None or spread > best[0]:
            best = (spread, kf, im)
    if best is not None:
        return scene_id, best[1], _jpeg(best[2])
    return scene_id, None, err


def _jpeg(im):
    from PIL import Image
    im = im.copy()
    im.thumbnail((SIZE, SIZE), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    return out.getvalue()


def set_size(size, quality):
    global SIZE, QUALITY
    SIZE, QUALITY = int(size), int(quality)


def remake(h, path, workers):
    """Remake (overwrite) the thumbnails of the listed scenes at the current SIZE / QUALITY."""
    ids = [x.strip() for x in Path(path).read_text().split() if x.strip()]
    con = h.ensure()
    with con.cursor() as cur:
        cur.execute(REPICK, (ids,))
        rows = cur.fetchall()
    con.commit()
    with cf.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(make, rows))
    good = [(kf, out, sid) for sid, kf, out in results if kf is not None]
    con = h.ensure()
    with con.cursor() as cur:
        cur.executemany("update ml_test.desk_thumbs set keyframe_id = %s, jpg = %s, created_at = now() "
                        "where scene_id = %s", good)
    con.commit()
    from PIL import Image
    borrowed = borrow(h, [sid for kf, out, sid in good if blank(Image.open(io.BytesIO(out)))])
    log("desk_thumbs --remake: {} of {} scenes remade at {} px, quality {} (avg {:.1f} KB); "
        "{} dark ones borrowed a neighbour".format(len(good), len(ids), SIZE, QUALITY,
                                                   sum(len(o) for _, o, _ in good) / max(len(good), 1) / 1024,
                                                   borrowed))
    return 0 if len(good) == len(rows) else 1


def find_blank(h):
    """scene ids whose stored thumbnail is black or blank (reads all thumbnails, about 250 MB)."""
    from PIL import Image
    found, last = [], None
    while True:
        con = h.ensure()
        with con.cursor() as cur:
            cur.execute("select scene_id::text, jpg from ml_test.desk_thumbs "
                        "where (%s::uuid is null or scene_id > %s::uuid) order by scene_id limit 5000", (last, last))
            rows = cur.fetchall()
        con.commit()
        if not rows:
            return found
        found += [sid for sid, jpg in rows if blank(Image.open(io.BytesIO(bytes(jpg))))]
        last = rows[-1][0]


def redo_blank(h, workers):
    ids = find_blank(h)
    log("desk_thumbs --redo-blank: {} thumbnails are black or blank".format(len(ids)))
    con = h.ensure()
    with con.cursor() as cur:
        cur.execute(REPICK, (ids,))
        rows = cur.fetchall()
    con.commit()
    fixed = same = failed = 0
    with cf.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(make, rows))
    from PIL import Image
    good = []
    for scene_id, kf, out in results:
        if kf is None:
            failed += 1
        elif blank(Image.open(io.BytesIO(out))):
            same += 1          # every keyframe of this scene is dark: kept the most detailed one
            good.append((kf, out, scene_id))
        else:
            fixed += 1
            good.append((kf, out, scene_id))
    con = h.ensure()
    with con.cursor() as cur:
        cur.executemany("update ml_test.desk_thumbs set keyframe_id = %s, jpg = %s where scene_id = %s", good)
    con.commit()
    log("desk_thumbs --redo-blank: {} replaced with a clear frame, {} scenes are dark throughout, "
        "{} failed".format(fixed, same, failed))
    borrowed = borrow(h, [sid for sid, kf, out in results
                          if kf is not None and blank(Image.open(io.BytesIO(out)))])
    log("desk_thumbs --redo-blank: {} dark scenes now show the nearest clear scene of the same "
        "video".format(borrowed))
    return 0 if not failed else 1


def borrow(h, ids):
    """A scene that is dark all the way through (a fade, a black title card) shows the picture of
    the nearest clear scene in the same video, so a search card is never just black."""
    from PIL import Image
    n = 0
    for sid in ids:
        con = h.ensure()
        with con.cursor() as cur:
            cur.execute("""select t.keyframe_id, t.jpg from ml_test.desk_index me
                           join ml_test.desk_index o on o.asset_sha256 = me.asset_sha256 and o.scene_id <> me.scene_id
                           join ml_test.desk_thumbs t on t.scene_id = o.scene_id
                           where me.scene_id = %s order by abs(o.scene_no - me.scene_no), o.scene_no""", (sid,))
            rows = cur.fetchall()
        con.commit()
        pick = next(((kf, jpg) for kf, jpg in rows if not blank(Image.open(io.BytesIO(bytes(jpg))))), None)
        if pick:
            con = h.ensure()
            with con.cursor() as cur:
                cur.execute("update ml_test.desk_thumbs set keyframe_id = %s, jpg = %s where scene_id = %s",
                            (pick[0], pick[1], sid))
            con.commit()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="at most this many scenes (0 = all)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--redo-blank", action="store_true",
                    help="replace existing thumbnails that are black or blank with a clearer keyframe")
    ap.add_argument("--size", type=int, default=SIZE, help="px, long side (default {})".format(SIZE))
    ap.add_argument("--quality", type=int, default=QUALITY, help="JPEG quality (default {})".format(QUALITY))
    ap.add_argument("--remake", metavar="FILE",
                    help="remake the thumbnails of the scene ids listed in FILE (one per line) at --size/--quality")
    args = ap.parse_args()
    set_size(args.size, args.quality)

    h = Reconnecting()
    if args.remake:
        return remake(h, args.remake, args.workers)
    if args.redo_blank:
        return redo_blank(h, args.workers)
    with h.ensure().cursor() as cur:
        cur.execute("select count(*) from ml_test.desk_index i where not exists "
                    "(select 1 from ml_test.desk_thumbs t where t.scene_id = i.scene_id)")
        todo = cur.fetchone()[0]
    h.conn.commit()
    if args.limit:
        todo = min(todo, args.limit)
    log("desk_thumbs: {} scenes without a thumbnail{}".format(todo, " (limit)" if args.limit else ""))

    done = failed = nbytes = 0
    failed_ids = []          # skipped for the rest of this run
    dark_ids = []            # scenes dark all the way through: borrow a neighbour's picture at the end
    from PIL import Image
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        while not args.limit or done + failed < args.limit:
            n = CHUNK if not args.limit else min(CHUNK, args.limit - done - failed)
            con = h.ensure()
            with con.cursor() as cur:
                cur.execute(PICK, (failed_ids, n))
                rows = cur.fetchall()
            con.commit()
            if not rows:
                break
            good = []
            for scene_id, kf, out in pool.map(make, rows):
                if kf is None:
                    failed += 1
                    failed_ids.append(scene_id)
                else:
                    good.append((scene_id, kf, out))
                    nbytes += len(out)
                    if blank(Image.open(io.BytesIO(out))):
                        dark_ids.append(scene_id)
            con = h.ensure()
            with con.cursor() as cur:
                cur.executemany("insert into ml_test.desk_thumbs (scene_id, keyframe_id, jpg) values (%s, %s, %s) "
                                "on conflict (scene_id) do nothing", good)
            con.commit()
            done += len(good)
            rate = done / max(time.time() - t0, 1)
            log("  {} made, {} failed, avg {:.1f} KB, {:.1f}/s, about {} min left".format(
                done, failed, nbytes / max(done, 1) / 1024, rate,
                int(max(todo - done - failed, 0) / max(rate, 0.01) / 60)))
    if dark_ids:
        log("desk_thumbs: {} scenes are dark throughout; {} now show the nearest clear scene".format(
            len(dark_ids), borrow(h, dark_ids)))
    log("desk_thumbs: finished - {} made, {} failed ({:.0f} MB)".format(done, failed, nbytes / 1e6))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
