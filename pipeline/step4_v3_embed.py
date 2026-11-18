#!/usr/bin/env python3
"""
step4_v3_embed (8 Nov 2026): Step 4 for the whole ml_test library after the Step 3 v3lib run.

    nohup caffeinate -i venv/bin/python scripts_content_v1/step4_v3_embed.py --after-step3 > work/logs/step4_v3_embed.log 2>&1 &

--after-step3: first waits for the Step 3 supervisor (pid in work/logs/step3_v3_supervise.pid) to exit, then
starts only if Step 3 really finished (no file left without a v3lib reading). If Step 3 stopped for any
other reason (money, errors) it logs that and does nothing.

Then calls ml-test-embed (gte-small, Supabase's built-in model, no cost) with EMBED_PARTS calls side by side,
EMBED_ROWS rows each, until no scene or file description is left to embed in the current text version.
Supabase hiccups (HTTP 5xx, timeouts) pause it (1, 2, then 5 minutes) instead of stopping it; it stops after
MAX_STALLS rounds in a row that embed nothing. Resuming = running it again: only rows still to do are taken.
The runner token is only sent as a header; never printed or logged.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PIPELINE_TARGET", "ml_test")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import step3_v3_run as s  # noqa: E402
from db import Reconnecting  # noqa: E402

EMBED_PARTS = 8
EMBED_ROWS = 2    # 9 Nov: 4 still hit the CPU limit with 8 side by side; 9 Nov: 10 rows (longer v2 texts) went over the edge function CPU limit
VERSION = "ml_test_search_text_v2_20261108"   # must match ml-test-embed
MAX_STALLS = 12
PAUSES = [60, 120, 300]


def log(msg):
    print("{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def left(h):
    return tuple(s.q(h, """select
      (select count(*) from ml_test.scene_analysis where summary_en is not null
         and (embedding is null or embedding_version is distinct from %(v)s))::int,
      (select count(*) from ml_test.asset_analysis where summary_en is not null
         and (embedding is null or embedding_version is distinct from %(v)s))::int""", {"v": VERSION})[0])


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_step3(h):
    pidf = ROOT / "work" / "logs" / "step3_v3_supervise.pid"
    pid = int(pidf.read_text().strip()) if pidf.exists() else 0
    if pid and alive(pid):
        log("waiting for the Step 3 supervisor (pid {}) to finish".format(pid))
        while alive(pid):
            time.sleep(60)
    main_left = len(s.pick_files(h, "main", 0))
    pat_left = len(s.pick_files(h, "patients", 0))
    if main_left or pat_left:
        log("Step 3 stopped with {} main and {} patient-record files not analysed - NOT starting Step 4; "
            "see work/logs/step3_v3_supervise.log".format(main_left, pat_left))
        return False
    log("Step 3 finished: every file has a v3lib reading")
    return True


def todo_ids(h):
    """Every scene and file description still to embed in VERSION (one scan each, done by the Mac, not per call)."""
    sc = [str(r[0]) for r in s.q(h, """select id from ml_test.scene_analysis where summary_en is not null
          and (embedding is null or embedding_model is distinct from 'gte-small' or embedding_version is distinct from %(v)s)
          order by created_at""", {"v": VERSION})]
    fi = [str(r[0]) for r in s.q(h, """select id from ml_test.asset_analysis where summary_en is not null
          and (embedding is null or embedding_model is distinct from 'gte-small' or embedding_version is distinct from %(v)s)
          order by created_at""", {"v": VERSION})]
    return sc, fi


def main():
    """9 Nov 2026: the Mac picks the rows once and sends exact ids ({"ids": ...}); EMBED_PARTS calls are kept in
    flight all the time. A call that fails (CPU limit, timeout) is retried later; a whole new pass picks up anything
    still left at the end. Stops after MAX_STALLS minutes in a row with nothing embedded."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-step3", action="store_true")
    args = ap.parse_args()
    h = Reconnecting()
    if args.after_step3 and not wait_for_step3(h):
        return
    cfg = dict(s.q(h, "select key, value from ml_test.pipeline_config where key in ('runner_url','runner_token')"))
    url = cfg["runner_url"].rsplit("/", 1)[0] + "/ml-test-embed"
    t0, done, failed_calls = time.time(), 0, 0
    for pass_no in range(1, 4):
        sc, fi = todo_ids(h)
        if not sc and not fi:
            break
        log("STEP 4 pass {} (gte-small): {} scene and {} file description(s) to embed, {} per call, {} calls side by side".format(
            pass_no, len(sc), len(fi), EMBED_ROWS, EMBED_PARTS))
        work = [{"scenes": sc[i:i + EMBED_ROWS], "files": []} for i in range(0, len(sc), EMBED_ROWS)]
        work += [{"scenes": [], "files": fi[i:i + EMBED_ROWS]} for i in range(0, len(fi), EMBED_ROWS)]
        total, last_log, last_progress, pause_i = len(sc) + len(fi), time.time(), time.time(), 0
        with cf.ThreadPoolExecutor(EMBED_PARTS) as ex:
            busy = {}
            while work or busy:
                while work and len(busy) < EMBED_PARTS:
                    job = work.pop(0)
                    busy[ex.submit(s.post, url, cfg["runner_token"], {"ids": job})] = job
                fin, _ = cf.wait(list(busy), timeout=30, return_when=cf.FIRST_COMPLETED)
                for f in fin:
                    job = busy.pop(f)
                    st, b = f.result()
                    n = (b.get("scenes_embedded", 0) + b.get("files_embedded", 0)) if st == 200 and isinstance(b, dict) else 0
                    if n:
                        done += n
                        last_progress = time.time()
                        pause_i = 0
                    if st != 200:
                        failed_calls += 1      # a later pass re-reads what is left, so rows this call missed are not lost
                if time.time() - last_progress > 60 * (pause_i + 1) and not fin:
                    pause_i += 1
                    if pause_i > MAX_STALLS:
                        log("nothing embedded for {} minutes - stopping (run again to resume)".format(MAX_STALLS))
                        return
                if time.time() - last_log > 120:
                    el = time.time() - t0
                    rate = done / el * 3600 if el else 0
                    log("{} embedded ({:.0f}/h) | about {} calls queued in this pass | {} failed call(s) so far | about {:.1f} h to go".format(
                        done, rate, len(work), failed_calls, (len(work) * EMBED_ROWS) / rate if rate else 0))
                    last_log = time.time()
    sc, fi = todo_ids(h)
    log("STEP 4 finished in {:.1f} h: {} embedded, {} failed call(s); still left: {} scene + {} file".format(
        (time.time() - t0) / 3600, done, failed_calls, len(sc), len(fi)))


if __name__ == "__main__":
    main()
