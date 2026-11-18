#!/usr/bin/env python3
"""
Score an OCR pass, and decide the run's status from the score.

For a video with expected_onscreen_text in content_v1.answer_key:

  recall         expected terms found in accepted text - case-insensitive,
                 whitespace-normalised, the whole term inside ONE accepted
                 observation (a caption split over rows is not a caption)
  strict_recall  the same, but case-sensitive against raw_text: what the
                 engine literally returned, before any tidying or
                 normalisation
  precision      share of accepted observations that are real text: a known
                 term, or made entirely of en/ms words, known-term words,
                 numbers and units. Everything else is listed as garbage.
  missed         the terms not found
  mean_confidence of accepted observations

For a video whose expected_onscreen_text is empty, every accepted
observation is a false positive.

Status, from the score and nothing else:
  COMPLETED  all expected terms found and no garbage (and, where nothing is
             expected, nothing accepted)
  PARTIAL    OCR_EXPECTED_TEXT_MISSED, else OCR_GARBAGE_ACCEPTED, else
             OCR_UNEXPECTED_TEXT_ACCEPTED - the message lists exactly what
  FAILED     the engine itself failed (set by the runner)

    python scripts/score_ocr.py [--report file]   # score what is stored now
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_ocr as O                                       # noqa: E402


def ws(s):
    return " ".join((s or "").split())


def clean_row(text, terms, term_words, words, cfg):
    """Is an accepted observation real text?"""
    key = O.normalise_text(text)
    if not key:
        return False
    if key in terms:
        return True
    toks = [t for t in key.split() if re.search(r"[A-Za-z0-9]", t)]
    if not toks:
        return False
    # A partial reading of the logo lockup is not real text however real its
    # words are ('THINK DRINDER MNG BEAUTY& HEALTH': mng is in the word list).
    if O.logo_fragment(key, terms, cfg) is not None:
        return False
    # Tokens judged in their line, as step2_ocr 3.1.0 judges them:
    # contractions, hyphenated compounds, domain terms, quantity numbers.
    flags = O.line_token_flags(toks, terms, words, cfg)
    for t, ok in zip(toks, flags):
        if ok:
            continue
        bare = re.sub(r"[^A-Za-z0-9.%-]", "", t)
        alpha = re.sub(r"[^A-Za-z]", "", bare).lower()
        if O.is_number(O.clean_number(t)) or O.is_unit(t, cfg):
            continue
        # Judged exactly as step2_ocr.token_ok judges it: punctuation around a
        # token makes no claim, a known term's own word is real text.
        if alpha and (alpha in words or alpha.upper() in term_words
                      or bare.upper() in term_words or bare.upper() in terms):
            continue
        return False
    return True


def score(expected, accepted, terms, words, cfg):
    """expected: list of terms, or None when the answer key has no opinion.
    accepted: [{"raw_text","normalized_text","confidence"}] accepted rows."""
    term_words = {w for t in terms for w in t.split()}
    exp = [str(t) for t in (expected or [])]
    norm_rows = [ws(r["normalized_text"]).upper() for r in accepted]
    raw_rows = [ws(r["raw_text"]) for r in accepted]
    found = [t for t in exp if any(ws(t).upper() in n for n in norm_rows)]
    strict = [t for t in exp if any(ws(t) in r for r in raw_rows)]
    missed = [t for t in exp if t not in found]
    garbage = [r["normalized_text"] for r in accepted
               if not clean_row(r["normalized_text"], terms, term_words,
                                words, cfg)]
    conf = [float(r["confidence"]) for r in accepted]
    out = {
        "expected": len(exp),
        "found": found, "strict_found": strict, "missed": missed,
        "recall": (len(found) / len(exp)) if exp else None,
        "strict_recall": (len(strict) / len(exp)) if exp else None,
        "accepted": len(accepted),
        "precision": ((len(accepted) - len(garbage)) / len(accepted))
        if accepted else None,
        "garbage": sorted(set(garbage)),
        "mean_confidence": (sum(conf) / len(conf)) if conf else None,
        "false_positives": (sorted({r["normalized_text"] for r in accepted})
                            if expected is not None and not exp else []),
    }
    out["status"], out["error_code"], out["message"] = status(out)
    return out


def status(s):
    parts = []
    if s["missed"]:
        parts.append("expected on-screen text not read: {}".format(
            "; ".join(s["missed"])))
    if s["garbage"]:
        parts.append("accepted text that is not real text: {}".format(
            "; ".join(s["garbage"])))
    if s["false_positives"] and not s["garbage"]:
        parts.append("accepted text where none is expected: {}".format(
            "; ".join(s["false_positives"])))
    if s["missed"]:
        return "PARTIAL", "OCR_EXPECTED_TEXT_MISSED", " | ".join(parts)
    if s["garbage"]:
        return "PARTIAL", "OCR_GARBAGE_ACCEPTED", " | ".join(parts)
    if s["false_positives"]:
        return "PARTIAL", "OCR_UNEXPECTED_TEXT_ACCEPTED", " | ".join(parts)
    if s["expected"]:
        msg = "all {} expected term(s) read; {} accepted, none garbage".format(
            s["expected"], s["accepted"])
    else:
        msg = "no on-screen text expected and none accepted"
    return "COMPLETED", None, msg


# --------------------------------------------------------------------------

def load_reference(conn, cfg):
    with conn.cursor() as cur:
        cur.execute("select name from test_v1.treatments where is_active")
        treatments = [r[0] for r in cur.fetchall()]
        cur.execute("select alias from test_v1.treatment_aliases")
        aliases = [r[0] for r in cur.fetchall()]
    return treatments, aliases


def score_stored(conn, cfg, rollback=True):
    """Score what ocr_observations holds right now, per video.

    rollback=False when called inside a transaction that is still writing:
    ending it here would throw those writes away."""
    treatments, aliases = load_reference(conn, cfg)
    terms = O.known_terms(cfg, treatments, aliases)
    words = O.load_words(cfg)
    with conn.cursor() as cur:
        cur.execute("""
            select sf.metadata->>'step1_item', k.expected_onscreen_text,
                   a.id
            from content_v1.answer_key k
            join content_v1.assets a on a.id = k.asset_id
            join content_v1.asset_sources s on s.asset_id = a.id
                 and s.is_first_discovered_source
            join content_v1.source_files sf on sf.id = s.source_file_id
            where k.media_kind = 'VIDEO'""")
        assets = cur.fetchall()
        cur.execute("""select asset_id, raw_text, normalized_text, confidence
                       from content_v1.ocr_observations
                       where accepted_for_search""")
        acc = {}
        for aid, raw, norm, conf in cur.fetchall():
            acc.setdefault(aid, []).append(
                {"raw_text": raw, "normalized_text": norm, "confidence": conf})
    if rollback:
        conn.rollback()
    return {item: score(exp, acc.get(aid, []), terms, words, cfg)
            for item, exp, aid in assets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    import step2_analyse as s2
    from db import Reconnecting
    handle = Reconnecting()
    conn = handle.ensure()
    cfg = s2.load_db_config(conn)
    res = score_stored(conn, cfg)
    handle.close()
    lines = ["| Item | Recall | Strict | Precision | Mean conf | Status | Missed / garbage |",
             "|---|---|---|---|---|---|---|"]
    for item in sorted(res, key=lambda i: int(i[1:])):
        r = res[item]
        f = (lambda v: "-" if v is None else "{:.2f}".format(v))
        lines.append("| {} | {} | {} | {} | {} | {} {} | {} |".format(
            item, f(r["recall"]), f(r["strict_recall"]), f(r["precision"]),
            f(r["mean_confidence"]), r["status"], r["error_code"] or "",
            "; ".join(r["missed"] + r["garbage"] + r["false_positives"])))
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
