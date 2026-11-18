#!/usr/bin/env python3
"""
step3_v3_run (7 Nov 2026): Step 3 visual analysis with prompt v3lib (reference library: look,
age range, expression, named doctors / influencers) for a chosen set of Step 2 files, through
the DeepSeek analyser ml-test-media-analyse. Only the chosen files are queued - never
ml_test.ai_enqueue(), which would queue the whole library.

    venv/bin/python scripts_content_v1/step3_v3_run.py --scope test            # 10 videos + 10 photos
    venv/bin/python scripts_content_v1/step3_v3_run.py --scope main            # videos + non-patient photos
    venv/bin/python scripts_content_v1/step3_v3_run.py --scope patients        # patient-record photos
    ... --limit N   (at most N files)   --dry-run   (counts only, no queue, no AI call)

What it does, resumable (run the same command again after a stop):
  1. picks files that finished Step 2 (they have scenes) and have no v3lib reading yet
  2. sets each scene's picture / on-screen text / transcript inputs (as run_ml_test STEP3)
  3. queues one analyse job per file chunk (up to max_scenes_per_call scenes); a file analysed
     with an older prompt is queued again and its file summary redone
  4. calls the analyser until these files have no waiting or working job; the spend caps in
     ml_test.pipeline_config (ai_gate) stop it, and it says so
Progress: printed and in work/logs/step3_v3_<scope>_<time>.log. Counts only, never file names.
The runner token is read from ml_test.pipeline_config and never printed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("PIPELINE_TARGET", "ml_test")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import Reconnecting  # noqa: E402
from run_ml_test import Run as Controller  # noqa: E402

PROMPT = "v3lib"
PATIENT = "Patient Records"
POLL, FIRE, STALL_MIN = 5, 6, 15   # 9 Nov 01:50: 3 halved speed, back to 6 (hiccups now pause only 3 min); 8 Nov 21:50: 12 -> 6, fewer DB connections at once (20:45 stop: connection slots full)
LOG = None


def say(msg):
    line = "{} {}".format(dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def q(h, sql, p=()):
    con = h.ensure()
    with con.cursor() as cur:
        cur.execute(sql, p)
        out = cur.fetchall() if cur.description else []
    con.commit()
    return out


def pick_files(h, scope, limit):
    base = """
      select a.file_sha256, a.id, a.kind, (sf.relative_path like %(pat)s) as patient,
             exists (select 1 from ml_test.ref_people_files f where f.asset_sha256 = a.file_sha256) as linked
      from ml_test.media_assets a
      join ml_test.asset_sources s on s.asset_id = a.id and s.is_first_discovered_source
      join ml_test.source_files sf on sf.id = s.source_file_id
      where exists (select 1 from ml_test.media_scenes m where m.asset_sha256 = a.file_sha256)
        and exists (select 1 from ml_test.media_scenes m where m.asset_sha256 = a.file_sha256
                    and exists (select 1 from ml_test.asset_keyframes kf where kf.scene_id = m.id)  -- no keyframe: cannot be read
                    and not exists (select 1 from ml_test.scene_analysis sa
                                    where sa.scene_id = m.id and sa.prompt_version = %(pv)s))"""
    rows = q(h, base, {"pat": "%" + PATIENT + "%", "pv": PROMPT})
    if scope == "test":
        import random
        rnd = random.Random(20261107)
        vids_l = [r for r in rows if r[2] == "video" and r[4]]
        vids_o = [r for r in rows if r[2] == "video" and not r[4]]
        ph_o = [r for r in rows if r[2] == "photo" and not r[3]]
        ph_p = [r for r in rows if r[2] == "photo" and r[3]]
        out = []
        for pool, n in ((vids_l, 6), (vids_o, 4), (ph_o, 6), (ph_p, 4)):
            rnd.shuffle(pool)
            out += pool[:n]
        rows = out
    elif scope == "main":
        rows = [r for r in rows if r[2] == "video" or not r[3]]
        rows.sort(key=lambda r: (r[2] != "video", not r[4], r[0]))  # videos first, linked first, fixed order
    elif scope == "patients":
        rows = sorted((r for r in rows if r[2] == "photo" and r[3]), key=lambda r: r[0])
    return rows[:limit] if limit else rows


def queue(h, shas):
    """Queue the scenes of these files that have no v3lib reading yet, max_scenes_per_call per job.
    Unfinished older jobs of these files are replaced; finished ones are kept (and counted in
    chunk_count, so the file summary waits for every job of the file)."""
    mx = int((q(h, "select value from ml_test.pipeline_config where key = 'max_scenes_per_call'") or [[6]])[0][0])
    q(h, "delete from ml_test.ai_jobs where kind = 'analyse' and asset_sha256 = any(%s) and status in ('waiting','quarantined','failed')", (shas,))
    q(h, "delete from ml_test.ai_jobs where kind = 'file_summary' and asset_sha256 = any(%s)", (shas,))
    return q(h, """
      with missing as (
        select s.id, s.asset_sha256, s.scene_no from ml_test.media_scenes s
        where s.asset_sha256 = any(%(shas)s) and s.thumb_path is not null
          and not exists (select 1 from ml_test.scene_analysis sa where sa.scene_id = s.id and sa.prompt_version = %(pv)s)),
      numbered as (
        select m.*, (row_number() over (partition by m.asset_sha256 order by m.scene_no) - 1) / %(mx)s as grp from missing m),
      chunks as (
        select asset_sha256, grp, array_agg(id order by scene_no) as ids from numbered group by asset_sha256, grp),
      prev as (
        select asset_sha256, coalesce(max(chunk_no) + 1, 0) as base,
               count(*) filter (where status = 'done') as done_n
        from ml_test.ai_jobs where kind = 'analyse' and asset_sha256 = any(%(shas)s) group by asset_sha256),
      ins as (
        insert into ml_test.ai_jobs (kind, asset_sha256, chunk_no, chunk_count, scene_ids, is_test)
        select 'analyse', c.asset_sha256, coalesce(p.base, 0) + c.grp,
               coalesce(p.done_n, 0) + count(*) over (partition by c.asset_sha256), c.ids, false
        from chunks c left join prev p using (asset_sha256)
        on conflict (kind, asset_sha256, chunk_no) do nothing
        returning 1)
      select count(*) from ins""", {"mx": mx, "shas": shas, "pv": PROMPT})[0][0]


def counts(h, shas):
    return dict(q(h, "select status, count(*)::int from ml_test.ai_jobs where kind in ('analyse','file_summary') "
                     "and asset_sha256 = any(%s) group by status", (shas,)))


def post(url, token, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-runner-token": token})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:300].decode(errors="replace")}
    except Exception as e:  # network blip: try again next round
        return 0, {"error": str(e)[:200]}


def spent(h, since):
    return q(h, "select coalesce(sum(cost_usd),0)::float, count(*), count(*) filter (where not ok) from ml_test.analysis_runs "
                "where provider = 'deepseek' and created_at >= %s", (since,))[0]


def balance(cfg):
    """DeepSeek balance in USD of the key the analyser uses (check_keys: no AI call, the key is never shown)."""
    import re
    st, body = post(cfg["runner_url"], cfg["runner_token"], {"check_keys": True})
    m = re.search(r"balance available \w+: ([0-9.]+) USD", str(body.get("deepseek", "")))
    return (float(m.group(1)) if m else None), str(body.get("deepseek", ""))[:120]


def run_batch(h, cfg, files, label):
    """Prepare, queue and drive one batch. Returns False when the analyser stopped (cap, money, errors)."""
    shas, ids = [r[0] for r in files], [r[1] for r in files]
    n_v = sum(r[2] == "video" for r in files)
    say("{}: {} files ({} videos, {} photos, {} in patient records, {} linked to a doctor/influencer)".format(
        label, len(files), n_v, len(files) - n_v, sum(bool(r[3]) for r in files), sum(bool(r[4]) for r in files)))
    t0 = dt.datetime.now(dt.timezone.utc)
    for sql in Controller.PREPARE_SQL:
        q(h, sql, {"shas": shas, "ids": ids})
    say("  queued {} job(s)".format(queue(h, shas)))
    last, last_change, idle, tick = None, time.time(), 0, 0
    while True:
        c = counts(h, shas)
        waiting, working = c.get("waiting", 0), c.get("working", 0)
        state = (c.get("done", 0), c.get("quarantined", 0), waiting, working)
        tick += 1
        if state != last and (tick % 12 == 0 or not waiting):
            usd, calls, bad = spent(h, t0)
            say("  done {}, working {}, waiting {}, failed {} | {} calls, {} errors, ${:.3f}".format(
                c.get("done", 0), working, waiting, c.get("quarantined", 0), calls, bad, usd))
        if state != last:
            last, last_change = state, time.time()
        if not waiting and not working:
            idle += 1
            if idle >= 3:          # a multi-chunk file's summary job appears after its last chunk
                break
        else:
            idle = 0
        if waiting:
            # 8 Nov: start the jobs in parallel. Sent one after another, each start (gate + claim,
            # ~1 s as the tables grew) kept only 2-3 jobs in flight; DeepSeek itself stayed ~4.5 s.
            from concurrent.futures import ThreadPoolExecutor
            n = min(FIRE, waiting)
            with ThreadPoolExecutor(max_workers=n) as ex:
                replies = list(ex.map(lambda _: post(cfg["runner_url"], cfg["runner_token"], {}), range(n)))
            for st, body in replies:
                if body.get("ok") is False or (st not in (200, 202) and st != 0):
                    say("  analyser stopped: HTTP {} {}".format(st, json.dumps(body)[:300]))
                    say("  stop reason: {}".format(q(h, "select value from ml_test.pipeline_config where key = 'stop_reason'")[0][0]))
                    return False
                if not body.get("claimed"):
                    break
        if time.time() - last_change > STALL_MIN * 60:
            say("  no progress for {} minutes; stopping (run again to resume)".format(STALL_MIN))
            return False
        time.sleep(POLL)
    usd, calls, bad = spent(h, t0)
    c = counts(h, shas)
    say("  batch finished: {} calls, {} errors, ${:.3f}; {} job(s) failed".format(calls, bad, usd, c.get("quarantined", 0)))
    say("  influencers named as patients, linked: {}".format(link_influencer_patients(h)))
    return True


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["test", "main", "patients"], action="append", required=True,
                    help="repeat to run scopes one after another, e.g. --scope main --scope patients")
    ap.add_argument("--limit", type=int, default=0, help="files per batch (0 = all in one batch)")
    ap.add_argument("--max-files", type=int, default=0, help="stop after this many files in total")
    ap.add_argument("--min-balance", type=float, default=0.40, help="stop when the DeepSeek balance falls below this (USD)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = HERE.parent
    (root / "work" / "logs").mkdir(parents=True, exist_ok=True)
    LOG = root / "work" / "logs" / "step3_v3_{}_{}.log".format("_".join(a.scope), dt.datetime.now().strftime("%Y%m%d_%H%M"))
    h = Reconnecting()
    cfg = dict(q(h, "select key, value from ml_test.pipeline_config where key in "
                    "('prompt_version','runner_url','runner_token','run_enabled','deepseek_spend_cap_usd','spend_cap_usd')"))
    if cfg.get("prompt_version") != PROMPT:
        raise SystemExit("pipeline_config prompt_version is {!r}, not {!r}".format(cfg.get("prompt_version"), PROMPT))
    say("caps: all-time ${}, per 16 h ${}; run_enabled {}; stop below ${:.2f} balance".format(
        cfg.get("deepseek_spend_cap_usd"), cfg.get("spend_cap_usd"), cfg.get("run_enabled"), a.min_balance))
    total, t_start = 0, time.time()
    for scope in a.scope:
        n_left = len(pick_files(h, scope, 0))
        say("scope {}: {} files without a v3lib reading".format(scope, n_left))
        if a.dry_run:
            continue
        b, prev = 0, None
        while True:
            if a.max_files and total >= a.max_files:
                say("reached --max-files {}; stopping".format(a.max_files))
                return
            bal, info = balance(cfg)
            if bal is None:
                say("could not read the DeepSeek balance ({}); stopping to be safe".format(info))
                return
            if bal < a.min_balance:
                say("DeepSeek balance ${:.2f} is below ${:.2f}; stopping (run again after a top-up)".format(bal, a.min_balance))
                return
            n = a.limit or 0
            if a.max_files:
                n = min(n or a.max_files, a.max_files - total)
            files = pick_files(h, scope, n)
            if not files:
                say("scope {}: nothing left".format(scope))
                break
            # 9 Nov 2026 (nightly run): the same files picked twice in a row means none of them got
            # a reading (a scene that keeps failing, or no picture to send): stop instead of looping
            shas = sorted(r[0] for r in files)
            if shas == prev:
                say("scope {}: {} file(s) still without a reading after a full batch; stopping "
                    "(they need a look)".format(scope, len(files)))
                return
            prev = shas
            b += 1
            say("balance ${:.2f}".format(bal))
            if not run_batch(h, cfg, files, "{} batch {}".format(scope, b)):
                return
            total += len(files)
            hrs = (time.time() - t_start) / 3600
            say("progress: {} files in {:.1f} h ({:.0f} files/h)".format(total, hrs, total / hrs if hrs else 0))
    bal, _ = balance(cfg)
    say("all done: {} files; DeepSeek balance now ${}".format(total, bal))


def link_influencer_patients(h):
    """The analyser names nobody whose role is 'patient'. An influencer having a treatment is
    both: link them when the model gave their name and the reference library links them to the
    file. Only influencers - real patients are never in the people list."""
    return q(h, """
      with hit as (
        select p.id as sp_id, pe.id as person_id, d.original_codes->>'name_given' as given
        from ml_test.scene_persons p
        join ml_test.scene_analysis sa on sa.id = p.analysis_id and sa.prompt_version = %s
        join ml_test.scene_person_detail d on d.scene_person_id = p.id
        join ml_test.ref_people_files f on f.asset_sha256 = sa.asset_sha256
        join ml_test.people pe on pe.id = f.person_id and pe.kind = 'influencer' and pe.active
        where p.person_id is null and p.role = 'patient' and d.original_codes->>'name_given' is not null
          and (lower(d.original_codes->>'name_given') = lower(pe.display_name)
               or lower(d.original_codes->>'name_given') = any (select lower(x) from unnest(pe.aliases) x))),
      upd as (
        update ml_test.scene_persons p set person_id = hit.person_id, person_link_source = 'FACE'
        from hit where p.id = hit.sp_id returning p.id)
      update ml_test.scene_person_detail d set person_link_evidence =
             'influencer named "' || hit.given || '" as the patient; linked to the file by the reference library'
      from hit where d.scene_person_id = hit.sp_id and hit.sp_id in (select id from upd)
      returning 1""", (PROMPT,)).__len__()


if __name__ == "__main__":
    main()
