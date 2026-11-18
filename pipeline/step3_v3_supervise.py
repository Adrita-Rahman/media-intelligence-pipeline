#!/usr/bin/env python3
"""
step3_v3_supervise (7 Nov 2026): keeps the Step 3 v3lib run going through DeepSeek outages.

    nohup caffeinate -i venv/bin/python scripts_content_v1/step3_v3_supervise.py > work/logs/step3_v3_supervise.log 2>&1 &

Runs step3_v3_run.py (main files in batches of 400, then patient-record photos in batches of
1,000). When a run stops early it looks at the last 100 DeepSeek calls:
  - mostly DeepSeek server errors (HTTP 5xx, empty answers, timeouts, network): an outage on
    DeepSeek's side - wait (10, 20, then 30 minutes), test one call, reset the error brake, go on
  - anything else (bad answers, our own errors), a spending cap, the balance guard, or more than
    MAX_RESUMES outage resumes: stop and leave it for a person to look at.
Every decision is logged. Stopping it: kill the pid in work/logs/step3_v3_supervise.pid.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PIPELINE_TARGET", "ml_test")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import step3_v3_run as s  # noqa: E402
from db import Reconnecting  # noqa: E402

MAX_RESUMES = 30
WAITS = [600, 1200, 1800]
OUTAGE = ("DeepSeek 5", "reading 'usage'", "timed out", "TimeoutError", "error sending request", "aborted", "502", "503", "504", "connection slots", "cannot read settings")


def log(msg):
    print("{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def left(h):
    main = len(s.pick_files(h, "main", 0))
    return main, (len(s.pick_files(h, "patients", 0)) if main == 0 else None)


def outage_share(h):
    rows = s.q(h, "select ok, coalesce(error, '') from ml_test.analysis_runs where provider = 'deepseek' "
                  "order by created_at desc limit 100")
    bad = [e for ok, e in rows if not ok]
    if not bad:
        return 0, 0.0
    return len(bad), sum(any(k in e for k in OUTAGE) for e in bad) / len(bad)


def reset_brake(h):
    now = json.dumps(dt.datetime.now(dt.timezone.utc).isoformat())
    s.q(h, "update ml_test.pipeline_config set value = %s::jsonb, updated_at = now() where key = 'brake_since'", (now,))
    s.q(h, "update ml_test.pipeline_config set value = 'true'::jsonb, updated_at = now() where key = 'run_enabled'")
    s.q(h, "update ml_test.pipeline_config set value = '\"\"'::jsonb, updated_at = now() where key = 'stop_reason'")


def probe(h, cfg):
    """One real call on a waiting single-scene job; True when DeepSeek answered."""
    r = s.q(h, "select id from ml_test.ai_jobs where kind = 'analyse' and status = 'waiting' "
               "order by cardinality(scene_ids), created_at limit 1")
    if not r:
        return True
    st, body = s.post(cfg["runner_url"], cfg["runner_token"], {"run_job": str(r[0][0])})
    log("probe: HTTP {} -> {}".format(st, json.dumps(body)[:160]))
    return body.get("status") == "done"


def main():
    (ROOT / "work" / "logs" / "step3_v3_supervise.pid").write_text(str(os.getpid()))
    h = Reconnecting()
    cfg = dict(s.q(h, "select key, value from ml_test.pipeline_config where key in ('runner_url','runner_token')"))
    resumes, wait_i, stuck, last_left = 0, 0, 0, None
    while True:
        main_left, pat_left = left(h)
        if main_left == 0 and pat_left == 0:
            log("everything analysed; done")
            return
        # 9 Nov 2026 (nightly run): files that keep failing must not be retried for ever
        stuck = stuck + 1 if (main_left, pat_left) == last_left else 0
        last_left = (main_left, pat_left)
        if stuck >= 3:
            log("3 runs in a row without one more file analysed ({} main, {} patient-record left) - "
                "stopping for a person to check".format(main_left, pat_left))
            return
        scope, size = ("main", 400) if main_left else ("patients", 1000)
        log("starting {} ({} files left there)".format(scope, main_left or pat_left))
        t0 = time.time()
        subprocess.call([sys.executable, str(HERE / "step3_v3_run.py"), "--scope", scope, "--limit", str(size),
                         "--min-balance", "1"], cwd=str(ROOT))
        now_main, now_pat = left(h)
        if now_main == 0 and now_pat == 0:
            log("everything analysed; done")
            return
        if (scope == "main" and now_main == 0) or (scope == "patients" and now_pat == 0):
            continue                      # that scope is finished; the next loop starts the other
        if time.time() - t0 > 3600:
            wait_i = 0
        bal, info = s.balance(cfg)
        if bal is not None and bal < 1:
            log("DeepSeek balance ${:.2f}: stopping for a top-up".format(bal))
            return
        stop = s.q(h, "select value from ml_test.pipeline_config where key = 'stop_reason'")[0][0] or ""
        if "limit reached" in stop or "cap reached" in stop:
            log("spending cap: {} - stopping".format(stop))
            return
        n_bad, share = outage_share(h)
        if n_bad and share < 0.7:
            log("stopped with {} failed calls, only {:.0%} of them DeepSeek outages - stopping for a person to check".format(n_bad, share))
            return
        resumes += 1
        if resumes > MAX_RESUMES:
            log("more than {} outage resumes - stopping".format(MAX_RESUMES))
            return
        probes = 0
        while True:
            probes += 1
            if probes > 48:
                log("DeepSeek still failing after about a day of waiting - stopping")
                return
            w = WAITS[min(wait_i, len(WAITS) - 1)]
            if n_bad == 0:                # 9 Nov: no DeepSeek call failed - a Supabase hiccup (connection slots); short pause
                w = 180
            log("{} ({} failed calls, {:.0%} server errors); waiting {} min".format(
                "Supabase hiccup" if n_bad == 0 else "DeepSeek outage", n_bad, share, w // 60))
            time.sleep(w)
            wait_i += 1
            reset_brake(h)
            if probe(h, cfg):
                log("DeepSeek answers again; resuming")
                break


if __name__ == "__main__":
    main()
