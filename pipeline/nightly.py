#!/usr/bin/env python3
"""
nightly (9 Nov 2026): the 9 pm run. Finds the photos and videos added to the NAS since the last
scan and takes them through Steps 1-4 with no one starting each step, exactly as the 91,000-file
run of 2-9 Nov (run_ml_test.py 2.0.0); at the end they can be searched.

    venv/bin/python scripts_content_v1/nightly.py              # the real run (launchd starts it at 21:00)
    venv/bin/python scripts_content_v1/nightly.py --dry-run    # checks + count the new files; changes nothing
    venv/bin/python scripts_content_v1/nightly.py --limit 5    # at most 5 new files (a test)

Order:
  1. checks: the NAS share and the work drive are there, at least MIN_FREE_GB free, the
     database answers, ml_test.pipeline_config run_enabled, the DeepSeek balance (Step 3 waits
     for a top-up below MIN_BALANCE; Steps 1, 2 and 4 still run). One run at a time (lock).
  2. batches an earlier night did not finish (power cut, NAS gone) are run again first, at most
     MAX_ATTEMPTS times each, then given up and listed in the report
  3. plan --all --settle-minutes 60: every new file on the whole share, except files changed in
     the last hour (maybe still copying); manifests of BATCH_SIZE files in /Volumes/WorkDrive/batches
  4. each batch: Steps 1 to 4 (run_ml_test.py run --to-step step4), then the Content Desk page
     (search index ml_test.desk_index + thumbnails), so each batch can be searched as soon as
     it is done instead of after the last one (14 Nov 2026: 17 batches took days to reach Step 3)
  5. Step 3 and Step 4 once more, for anything still needing them (leftovers of earlier nights)
  6. the Content Desk page again: refresh its search index and make thumbnails
  7. a report: one line per night in work/logs/nightly/summary.md and a Mac notification

Logs: work/logs/nightly/<date>.log (counts and item codes only; work/ is not in git). The batch
manifests and the per-batch step logs hold file names (patient names), so they stay in
/Volumes/WorkDrive/batches as before. The runner token is never printed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PIPELINE_TARGET"] = "ml_test"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_ml_test as C  # noqa: E402

LOG_DIR = ROOT / "work" / "logs" / "nightly"
STATE = LOG_DIR / "state.json"
SUMMARY = LOG_DIR / "summary.md"
BATCH_SIZE = 500
SETTLE_MINUTES = 60
MAX_ATTEMPTS = 3
MIN_FREE_GB = 30          # cached originals of the new files are kept on the work drive
MIN_BALANCE = 1.0         # USD; the same guard as step3_v3_supervise
LOG = None


def say(msg=""):
    line = "{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


C.say = say   # the controller's progress lines go to the nightly log too


def notify(text):
    """A Mac notification (counts only, never file names)."""
    subprocess.run(["osascript", "-e", 'display notification "{}" with title "KDI nightly run"'.format(
        text.replace('"', "'"))], capture_output=True)


def record(handle, started, outcome, **fields):
    """18 Nov 2026: one row per run in ml_test.nightly_runs, so a colleague can see from any computer
    (the Nightly Runs page) whether each night processed the new files and finished properly.
    Written as 'running' at the start and filled in at the end; a run that dies leaves 'running'.
    Counts and batch codes only. A failure here is logged and never stops the run."""
    try:
        local = started.astimezone()
        cols = dict(fields, outcome=outcome)
        cols.setdefault("trigger", "scheduled" if local.hour == 21 and local.minute < 10 else "manual")
        names = sorted(cols)
        C.write(handle, "insert into ml_test.nightly_runs (started_at, {}, updated_at) values (%s, {}, now()) "
                        "on conflict (started_at) do update set {}, updated_at = now()".format(
                            ", ".join(names), ", ".join(["%s"] * len(names)),
                            ", ".join("{0} = excluded.{0}".format(n) for n in names)),
                [started] + [cols[n] for n in names])
    except Exception as e:
        say("note: nightly record not saved ({}: {})".format(type(e).__name__, str(e)[:120]))


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"batches": {}}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=1))


def nas_ok(root):
    """The share answers a listing within 2 minutes (an SMB share over Tailscale can hang)."""
    try:
        r = subprocess.run(["ls", root], capture_output=True, timeout=120)
        return r.returncode == 0 and bool(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return False


def checks(handle):
    """Returns (problems that stop the night, notes)."""
    stop, notes = [], []
    if not nas_ok(C.NAS_ROOT):
        stop.append("the NAS share is not connected ({} does not answer)".format(C.NAS_ROOT))
    free = shutil.disk_usage(str(C.BATCH_DIR.parent)).free / 1e9
    if free < MIN_FREE_GB:
        stop.append("only {:.0f} GB free on the work drive (need {})".format(free, MIN_FREE_GB))
    cfg = dict(C.rows(handle, "select key, value from ml_test.pipeline_config where key in "
                              "('run_enabled','deepseek_spend_cap_usd','runner_url','runner_token')"))
    if not cfg.get("run_enabled"):
        notes.append("pipeline_config run_enabled is false: Step 3 will not run")
    import step3_v3_run as s3
    bal, info = s3.balance(cfg)
    spent = C.rows(handle, "select coalesce(sum(cost_usd), 0)::float from ml_test.analysis_runs "
                           "where provider = 'deepseek'")[0][0]
    cap = float(cfg.get("deepseek_spend_cap_usd") or 0)
    if bal is None:
        notes.append("could not read the DeepSeek balance")
    elif bal < MIN_BALANCE:
        notes.append("DeepSeek balance ${:.2f}: Step 3 waits for a top-up".format(bal))
    if cap and cap - spent < 2:
        notes.append("DeepSeek all-time cap ${:.2f}, ${:.2f} used: Step 3 will stop at the cap".format(cap, spent))
    say("checks: NAS {}, {:.0f} GB free, DeepSeek balance {}, cap ${:.2f} with ${:.2f} used".format(
        "ok" if not any("NAS" in s for s in stop) else "MISSING", free,
        "${:.2f}".format(bal) if bal is not None else "unknown", cap, spent))
    return stop, notes, bal


def run_batch(manifest, dry):
    """Steps 1 to 4 for one manifest, in a separate process (its own lock and report)."""
    cmd = [str(C.PYTHON), "-u", str(HERE / "run_ml_test.py"), "run", "--manifest", str(manifest),
           "--to-step", "step4"] + (["--dry-run"] if dry else [])
    with open(LOG, "a") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, **C.STEP2_ENV})
    return rc


def counts(handle, manifests):
    """(files in the manifests, ingested, searchable) for tonight's batches."""
    items = []
    for m in manifests:
        try:
            items += [e["item"] for e in json.loads(Path(m).read_text())["files"]]
        except (OSError, ValueError, KeyError):
            pass
    if not items:
        return 0, 0, 0
    ing, srch = C.rows(handle, """
        select count(distinct a.id),
               count(distinct a.id) filter (where exists (
                 select 1 from ml_test.scene_analysis sa where sa.asset_sha256 = a.file_sha256
                   and sa.embedding is not null and sa.prompt_version = 'v3lib'))
        from ml_test.source_files sf
        join ml_test.asset_sources s on s.source_file_id = sf.id
        join ml_test.media_assets a on a.id = s.asset_id
        where sf.metadata->>'step1_item' = any(%s)""", (items,))[0]
    return len(items), ing, srch


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="checks and counts only; changes nothing")
    ap.add_argument("--limit", type=int, default=None, help="at most this many new files (a test)")
    args = ap.parse_args()
    # keep the Mac awake until this process ends (launchd starts Python directly)
    subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    LOG = LOG_DIR / "{}{}.log".format(stamp, "_dry" if args.dry_run else "")
    lock_path = LOG_DIR / ".lock"
    if not args.dry_run:
        try:
            C.lock(lock_path)
        except SystemExit:
            # another run is still going: leave a row, so the night does not look like "no run"
            say("not started: another run is still going")
            try:
                when = C.now()
                record(C.connect(), when, "blocked", result="not started: another run was still going",
                       finished_at=when, hours=0)
            except Exception:
                pass
            raise
    t0, started = time.time(), C.now()
    say("nightly run{} ({})".format(" - DRY RUN, nothing is changed" if args.dry_run else "", C.VERSION))
    handle = C.connect()
    if not args.dry_run:
        record(handle, started, "running", result="running")
    state = load_state()
    notes, status, tonight = [], "finished", []
    try:
        stop, more, bal = checks(handle)
        notes += more
        if stop:
            status = "not run: " + "; ".join(stop)
            say("STOP: " + status)
            return 2
        # 2. batches an earlier night did not finish
        todo = []
        for name, b in sorted(state["batches"].items()):
            if b.get("done"):
                continue
            if b.get("attempts", 0) >= MAX_ATTEMPTS:
                notes.append("{} gave up after {} attempts (see its logs)".format(name, b["attempts"]))
                continue
            if (C.BATCH_DIR / name).exists():
                todo.append(C.BATCH_DIR / name)
        if todo:
            say("{} unfinished batch(es) from earlier nights first".format(len(todo)))
        # 3. new files on the whole share
        say("scanning the whole NAS share for new files (read only) ...")
        pargs = argparse.Namespace(root=C.NAS_ROOT, folder=[], all=True, settle_minutes=SETTLE_MINUTES,
                                   limit=args.limit, out_dir=C.BATCH_DIR, batch_size=BATCH_SIZE,
                                   dry_run=args.dry_run)
        new = C.cmd_plan(pargs)
        if args.dry_run:
            say("dry run: Steps 1-4 not started")
            if todo:
                for m in todo:
                    run_batch(m, True)
            return 0
        for m in new:
            state["batches"][Path(m).name] = {"created": stamp, "attempts": 0, "done": False}
        save_state(state)
        tonight = todo + [Path(m) for m in new]
        # 4. Steps 1 and 2, batch by batch
        for m in tonight:
            b = state["batches"].setdefault(m.name, {"created": stamp, "attempts": 0, "done": False})
            b["attempts"] = b.get("attempts", 0) + 1
            save_state(state)
            say("{}: Steps 1-4 (attempt {})".format(m.name, b["attempts"]))
            rc = run_batch(m, False)
            b["done"] = rc == 0
            b["last_exit"] = rc
            save_state(state)
            if rc:
                status = "finished with problems"
                notes.append("{} stopped in Steps 1-4 (exit {}); tried again tomorrow".format(m.name, rc))
            else:   # searchable now, not after the last batch
                say("CONTENT DESK  search index and thumbnails for {}".format(m.name))
                warn = C.content_desk(handle, LOG_DIR / stamp)
                if warn:
                    notes.append("{}: {}".format(m.name, warn))
            if not nas_ok(C.NAS_ROOT):
                notes.append("the NAS went away during the run")
                break
        # 5. Steps 3 and 4 for everything still needing them (also leftovers of earlier nights)
        step_dir = LOG_DIR / stamp
        say("STEP 3  DeepSeek analysis")
        left3 = C.run_step3(handle, step_dir, notes)
        say("STEP 4  embeddings")
        left4 = C.run_step4(handle, step_dir, notes)
        if sum(left3) or sum(left4):
            status = "finished with problems"
        # 6. the Content Desk page: search index + thumbnails for the new scenes
        say("CONTENT DESK  search index and thumbnails")
        warn = C.content_desk(handle, step_dir)
        if warn:
            notes.append(warn)
            status = "finished with problems"
    except C.Stop as e:
        status = "stopped: {}".format(e)
        say("STOPPED: {}".format(e))
    except Exception as e:   # anything unexpected: report it, the next night tries again
        status = "error: {}".format(str(e)[:300])
        say("ERROR: {!r}".format(e))
        import traceback
        with open(LOG, "a") as fh:
            traceback.print_exc(file=fh)
    finally:
        if not args.dry_run:
            report(handle, started, t0, status, notes, tonight)
            if lock_path.exists():
                lock_path.unlink()
        handle.close()
    return 0 if status == "finished" else 1


def report(handle, started, t0, status, notes, tonight):
    try:
        n, ing, srch = counts(handle, tonight)
        usd = C.rows(handle, "select coalesce(sum(cost_usd), 0)::float from ml_test.analysis_runs "
                             "where provider = 'deepseek' and created_at >= %s", (started,))[0][0]
    except Exception as e:  # the database may be the problem
        n = ing = srch = "?"
        usd = 0.0
        notes = notes + ["report counts failed: {}".format(str(e)[:120])]
    hrs = (time.time() - t0) / 3600
    say("")
    say("RESULT: {}".format(status))
    say("files: {} new, {} ingested, {} searchable | DeepSeek ${:.3f} | {:.1f} h".format(n, ing, srch, usd, hrs))
    for x in notes:
        say("note: " + x)
    if not SUMMARY.exists():
        SUMMARY.write_text("# Nightly runs\n\n| Started | Result | New files | Ingested | Searchable | DeepSeek | Hours | Notes |\n"
                           "|---|---|---|---|---|---|---|---|\n")
    with open(SUMMARY, "a") as fh:
        fh.write("| {} | {} | {} | {} | {} | ${:.3f} | {:.1f} | {} |\n".format(
            started.astimezone().strftime("%Y-%m-%d %H:%M"),
            status.replace("|", "/")[:120], n, ing, srch, usd, hrs,
            "; ".join(notes).replace("|", "/")[:400]))
    num = lambda x: x if isinstance(x, int) else None
    record(handle, started,
           "finished" if status == "finished" else "not_run" if status.startswith("not run") else "problems",
           result=status[:200], finished_at=C.now(), new_files=num(n), ingested=num(ing),
           searchable=num(srch), deepseek_usd=round(float(usd), 3), hours=round(hrs, 1),
           notes="; ".join(notes)[:2000] or None)
    notify("{}: {} new file(s), {} searchable".format(status if len(status) < 60 else status[:57] + "...", n, srch))


if __name__ == "__main__":
    sys.exit(main())
