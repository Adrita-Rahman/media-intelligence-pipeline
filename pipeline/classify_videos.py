#!/usr/bin/env python3
"""
RAW or EDITED for every video whose Step 2 is done (user's rule, 2 Nov 2026;
supabase/ml_test/007_ml_test_video_edit_class.sql):

  RAW only when all hold: one scene, no music, no clear speech, no text on
  screen. Anything else is EDITED, and `reasons` says which signs were found.

  scenes        Step 2 scenes for the video
  music         music_check.music_profile on the cached original (Apple sound
                classifier)
  clear speech  a COMPLETE transcript chunk with at least SPEECH_MIN_WORDS words;
                a video Step 2 did not transcribe (older runs skipped one-scene
                videos) falls back to the sound classifier's speech share
  text          any accepted OCR reading (names on scrubs included: user, 2 Nov)

Writes ml_test.video_edit_class; videos already labelled with this
RULE_VERSION are skipped. --loop keeps going while Step 2 video runs are
still working, so new videos are labelled as they finish.

    PIPELINE_TARGET=ml_test venv/bin/python scripts_content_v1/classify_videos.py [--loop]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import target  # noqa: E402
from db import Reconnecting  # noqa: E402
from music_check import music_profile  # noqa: E402

RULE_VERSION = "edit_class_user_rule_2026_11_02"
SPEECH_MIN_WORDS = 3
CLASSIFIER_SPEECH_SHARE = 0.30

TODO_SQL = """
select a.id, a.content_hash, a.file_name,
  (select count(*) from content_v1.asset_scenes s where s.asset_id = a.id) as scenes,
  (select count(*) from content_v1.asset_transcript_chunks c where c.asset_id = a.id) as chunks,
  (select coalesce(max(array_length(regexp_split_to_array(trim(c.transcript_text), '\\s+'), 1)), 0)
     from content_v1.asset_transcript_chunks c
    where c.asset_id = a.id and c.transcription_status = 'COMPLETE'
      and coalesce(trim(c.transcript_text), '') <> '') as words,
  (select count(*) from content_v1.ocr_observations o
    where o.asset_id = a.id and o.accepted_for_search) as ocr_lines
from content_v1.assets a
where a.media_kind = 'VIDEO'
  and exists (select 1 from content_v1.asset_scenes s where s.asset_id = a.id)
  and not exists (select 1 from ml_test.video_edit_class v
                  where v.asset_sha256 = a.content_hash and v.rule_version = %s)
limit 200
"""

UPSERT_SQL = """
insert into ml_test.video_edit_class
  (asset_sha256, edit_class, scenes, music, music_share, music_seconds, music_labels,
   clear_speech, speech_words, speech_source, ocr_lines, reasons, rule_version, decided_at)
values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
on conflict (asset_sha256) do update set
  edit_class = excluded.edit_class, scenes = excluded.scenes, music = excluded.music,
  music_share = excluded.music_share, music_seconds = excluded.music_seconds,
  music_labels = excluded.music_labels, clear_speech = excluded.clear_speech,
  speech_words = excluded.speech_words, speech_source = excluded.speech_source,
  ocr_lines = excluded.ocr_lines, reasons = excluded.reasons,
  rule_version = excluded.rule_version, decided_at = now()
"""


def decide(scenes, music, words, chunks, speech_share, ocr_lines):
    if chunks:
        speech, source = words >= SPEECH_MIN_WORDS, "transcript"
    else:
        speech, source = speech_share >= CLASSIFIER_SPEECH_SHARE, "sound classifier"
    reasons = []
    if scenes > 1:
        reasons.append("{} scenes".format(scenes))
    if music["music"]:
        reasons.append("music ({:.0%} of the soundtrack, {}s clear)".format(
            music["music_share"], music["music_seconds"]))
    if speech:
        reasons.append("clear speech" + (" ({} words)".format(words) if chunks else " (sound classifier)"))
    if ocr_lines:
        reasons.append("text on screen ({} readings)".format(ocr_lines))
    return ("EDITED" if reasons else "RAW"), speech, source, reasons


def video_runs_going():
    return subprocess.run(["pgrep", "-f", "step2_all_logs/videos.sh"],
                          capture_output=True).returncode == 0


def one_pass(handle):
    cur = handle.ensure().cursor()
    cur.execute(TODO_SQL, (RULE_VERSION,))
    rows = cur.fetchall()
    handle.ensure().rollback()
    handle.close()      # hold no connection while measuring music (pooler limit)
    done = 0
    for aid, sha, name, scenes, chunks, words, ocr_lines in rows:
        path = target.cached_original_path(sha, name)
        if not path.exists():
            print("  {}  skipped: no cached original".format(sha[:12]), flush=True)
            continue
        try:
            music = music_profile(path)
        except Exception as exc:
            print("  {}  music check failed: {}".format(sha[:12], str(exc)[:200]), flush=True)
            continue
        cls, speech, source, reasons = decide(scenes, music, words, chunks,
                                              music.get("speech_share", 0.0), ocr_lines)
        c = handle.ensure()
        with c.cursor() as cur:
            cur.execute(UPSERT_SQL, (sha, cls, scenes, music["music"], music["music_share"],
                                     music["music_seconds"], json.dumps(music["top_music_labels"]),
                                     speech, words, source, ocr_lines, reasons, RULE_VERSION))
        c.commit()
        done += 1
        print("  {}  {:6s} {}".format(sha[:12], cls, "; ".join(reasons) or "one take, no music, "
                                      "no clear speech, no text"), flush=True)
    handle.close()
    return len(rows), done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true", help="keep labelling while video runs are going")
    args = ap.parse_args()
    for _ in range(60):          # wait for a free slot in the connection pooler
        try:
            handle = Reconnecting()
            break
        except Exception:
            time.sleep(10)
    else:
        raise SystemExit("no database connection after 10 minutes")
    while True:
        found, done = one_pass(handle)
        print("{}  labelled {} of {} found".format(time.strftime("%F %T"), done, found), flush=True)
        if done:
            continue
        if args.loop and video_runs_going():
            time.sleep(300)
            continue
        break


if __name__ == "__main__":
    raise SystemExit(main())
