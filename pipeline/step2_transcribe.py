#!/usr/bin/env python3
"""
Speech, and the refusal to report speech that is not there.

WHAT WAS ACTUALLY WRONG

Every configured safeguard in this area existed and none of it ran. The live
path in step2_analyse.transcribe() read six config keys - model, compute type,
beam size, VAD, enabled, forced language - and ignored the other eleven. So:

    IMG_3413  3.7s of a camera pointed at clothing returned fluent French,
              stored COMPLETE, although transcription_allowed_languages is
              ["en","ms"]. The allowlist was never consulted.
    IMG_6383  "Ia tidak baik." four times over, then the transcript stops at
              40.6s of an 83.7s clip. whisper_condition_on_previous_text is
              configured false and was never passed, so each window was primed
              with the last one and the loop fed itself.
    export_.. "Zilfum, Aftar former, Diners, Dr. Liu, Cranit Dr. Handa" - real
              speech, badly misheard, because the model was never told the
              clinic's own vocabulary.
    IMG_3424  English speech stored as 'ms'.
    IMG_3424  chunks ending at 6.0s on a 5.567s video, and IMG_1160 at 24.4s
              on 23.667s: transcripts overrunning the media they describe.

None of these is Whisper being bad at Malay. Every one is detectable from the
numbers Whisper already returns - mean token log-probability, gzip compression
ratio of its own output, and the probability the audio was not speech - and
the 1.0.0 code asked for none of them.

THE RULES HERE, ALL GENERAL

  * Language is CHOSEN, not accepted. detect_language() gives a probability
    for every language; the choice is the best one among the allowed set, and
    that language is then forced. A clip can therefore never be stored as
    French. Confidence is judged after renormalising over the allowed
    languages only, which is what makes the threshold meaningful: IMG_1160 is
    real Malay at a raw 0.45, which a blunt cutoff would throw away, but it is
    0.9 once the languages this clinic does not speak are removed.
  * condition_on_previous_text=False, which is what stops a loop feeding
    itself.
  * Every segment is scored, and a failing one is KEPT as a row with
    transcription_status='EXCLUDED' and a failure_reason. Excluded text is
    never deleted - a reviewer has to be able to see what was thrown away and
    why - but it is not search text either.
  * Repetition is judged ACROSS segments as well as within one. "Ia tidak
    baik." is unremarkable on its own; four of them in a row is the failure.
  * The clinic's vocabulary is given to the model up front.
  * If most of the speech time is excluded, the asset does not get to be
    COMPLETE; it gets a stated no-usable-speech outcome.
  * Every chunk is clamped to [0, duration], and the overrun is recorded.

'EXCLUDED' rather than 'REJECTED' is not a wording preference: the table's
transcription_status check constraint permits COMPLETE, EMPTY_NO_SPEECH,
FAILED and EXCLUDED, so a 'REJECTED' row could not be written at all.
"""
from __future__ import annotations

import difflib
import gzip
import math
import re


def _need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: " + key)
    return cfg[key]


STATUS_COMPLETE = "COMPLETE"
STATUS_EMPTY_NO_SPEECH = "EMPTY_NO_SPEECH"
STATUS_FAILED = "FAILED"
STATUS_EXCLUDED = "EXCLUDED"

# why a segment was excluded. All three mean "this was never speech", which is
# what lets a video full of them still count as fully processed.
CAT_HALLUCINATION = "hallucination"
CAT_NO_VOICE = "no_voice"
CAT_LANGUAGE = "language"
NOT_REAL_SPEECH = (CAT_HALLUCINATION, CAT_NO_VOICE, CAT_LANGUAGE)
# A decoding LOOP (compression, repeated n-grams, one word repeated, the same
# phrase segment after segment) is deliberately NOT in NOT_REAL_SPEECH. The
# voice-activity filter only hands Whisper audio it judged to be voice, so a
# loop there is speech the decoder lost - V17's "4-4-4-4" covers 13.5 s of a
# patient talking. If a loop survives the no-hotword retry, the run is PARTIAL.
# A loop that is ALSO flagged no_voice or language is still not speech.
CAT_LOOP = "loop"

# asset-level outcomes, which are not chunk statuses
OUTCOME_COMPLETE = "COMPLETE"
OUTCOME_NO_SPEECH = "EMPTY_NO_SPEECH"
OUTCOME_MUSIC_ONLY = "EMPTY_MUSIC_ONLY"
OUTCOME_NO_USABLE = "NO_USABLE_SPEECH"
OUTCOME_FAILED = "FAILED"
OUTCOME_EXCLUDED = "EXCLUDED"


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

def vocabulary(cfg):
    """The terms the clinic says out loud, as a list.

    content_v1 (29 Sep 2026): the list is transcript_vocabulary_hint - the
    clinic name, the doctors in content_v1.persons, the device labels in
    content_v1.devices and the treatment words - replacing the older
    transcription_vocabulary, which lacked the treatment words ('Fractional'
    came out as 'Fashioner')."""
    hint = _need(cfg, "transcript_vocabulary_hint")
    if isinstance(hint, str):
        hint = [t.strip() for t in hint.split(",")]
    return [str(t).strip() for t in hint if str(t).strip()]


VOCAB_NONE = "none"
VOCAB_PROMPT = "initial_prompt"
VOCAB_HOTWORDS = "hotwords"


def vocabulary_arguments(cfg):
    """How the clinic's terms reach the model: as a prompt, as hotwords, or
    not at all. initial_prompt prefixes the decode with the word list, which
    also drags the decode toward that list's style; hotwords biases towards
    the terms without becoming context. Which of the three is used is a
    measured choice, recorded in transcription_vocabulary_mode."""
    mode = _need(cfg, "transcription_vocabulary_mode")
    terms = vocabulary(cfg)
    if not terms or mode == VOCAB_NONE:
        return {"initial_prompt": None, "hotwords": None}
    joined = ", ".join(terms) + "."
    if mode == VOCAB_PROMPT:
        return {"initial_prompt": joined, "hotwords": None}
    if mode == VOCAB_HOTWORDS:
        return {"initial_prompt": None, "hotwords": joined}
    raise SystemExit("unknown transcription_vocabulary_mode: {!r}".format(mode))


# ---------------------------------------------------------------------------
# per-segment measures
# ---------------------------------------------------------------------------

def compression_ratio(text):
    """Whisper's own repetition measure: fluent prose compresses about 1.5-2x,
    a loop compresses far harder."""
    data = (text or "").encode("utf-8")
    if not data:
        return 0.0
    return len(data) / float(len(gzip.compress(data)))


def repeated_ngram_ratio(text, n):
    """Share of n-grams that are not the first occurrence, within one
    segment."""
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    seen, repeats = set(), 0
    for g in grams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / float(len(grams))


def word_stats(text):
    """The single-word loop measures: the longest run of one word repeated
    back to back, and the most frequent word's count and share."""
    words = re.findall(r"\w+", (text or "").lower())
    longest, run, prev, run_word = 0, 0, None, None
    for w in words:
        run = run + 1 if w == prev else 1
        prev = w
        if run > longest:
            longest, run_word = run, w
    top_word, top_count = None, 0
    for w in set(words):
        c = words.count(w)
        if c > top_count or (c == top_count and top_word is not None
                             and w < top_word):
            top_word, top_count = w, c
    return {"words": len(words), "longest_run": longest, "run_word": run_word,
            "top_word": top_word, "top_count": top_count,
            "top_share": (top_count / float(len(words))) if words else 0.0}


def normalise(text):
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def score_segment(seg, cfg, ctx=None):
    """(ok, reason, measures) for one raw Whisper segment.

    avg_logprob is deliberately NOT a single cutoff. Measured across these
    fifteen videos it does not separate what it would need to: IMG_3424's real
    speech scores -1.04 and IMG_3413's noise scores -1.02, so any line drawn
    between them is a line drawn around two files rather than a rule. What
    distinguishes them is not the log-probability at all - IMG_3413 is audio
    that is barely in an allowed language, which transcription_allowed_mass_min
    already catches.

    So there are two levels. Below the hard floor the model is incoherent and
    that is enough on its own. Between the hard floor and the soft minimum the
    reading is weak but not damning, and it only excludes the segment when
    something else agrees with it - the segment is probably not speech
    (no_speech_prob), or the video barely contains speech at all. A weak score
    with nothing corroborating it is kept, because weak is not the same as
    wrong.
    """
    n = int(_need(cfg, "transcript_repeat_ngram_size"))
    text = (seg.get("text") or "").strip()
    cr = seg.get("compression_ratio")
    if cr is None:
        cr = compression_ratio(text)
    ws = word_stats(text)
    measures = {
        "compression_ratio": round(float(cr), 4),
        "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4),
        "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4),
        "repeat_ngram_ratio": round(float(repeated_ngram_ratio(text, n)), 4),
        "word_run": ws["longest_run"],
        "top_word_share": round(ws["top_share"], 4),
    }
    # Each reason carries the KIND of failure it is, because the run status
    # depends on it: audio that was never speech is a video fully and
    # correctly processed, whereas real speech thrown away is a partial result.
    reasons, cats = [], []

    def fail(kind, text):
        cats.append(kind)
        reasons.append(text)

    if measures["compression_ratio"] > float(
            _need(cfg, "transcript_compression_ratio_max")):
        fail(CAT_LOOP, "compression_ratio {:.2f} > {}".format(
            measures["compression_ratio"],
            _need(cfg, "transcript_compression_ratio_max")))
    soft = float(_need(cfg, "transcript_avg_logprob_min"))
    hard = float(_need(cfg, "transcript_avg_logprob_hard_min"))
    mode = _need(cfg, "transcript_avg_logprob_mode")
    lp = measures["avg_logprob"]
    if mode == "sole":
        if lp < soft:
            fail(CAT_NO_VOICE, "avg_logprob {:.2f} < {}".format(lp, soft))
    elif lp < hard:
        fail(CAT_NO_VOICE,
             "avg_logprob {:.2f} < hard floor {}".format(lp, hard))
    elif lp < soft:
        nsp_corr = float(_need(cfg, "transcript_no_speech_prob_corroborating"))
        ratio = None if ctx is None else ctx.get("speech_ratio")
        min_ratio = float(_need(cfg, "min_speech_ratio"))
        corroborations = []
        if measures["no_speech_prob"] > nsp_corr:
            corroborations.append("no_speech_prob {:.2f} > {}".format(
                measures["no_speech_prob"], nsp_corr))
        if ratio is not None and ratio < min_ratio:
            corroborations.append(
                "voice activity covers only {:.1%} of the video, below "
                "min_speech_ratio={}".format(ratio, min_ratio))
        if corroborations:
            fail(CAT_NO_VOICE, "avg_logprob {:.2f} < {} and {}".format(
                lp, soft, "; ".join(corroborations)))
    if measures["no_speech_prob"] > float(
            _need(cfg, "transcript_no_speech_prob_max")):
        fail(CAT_NO_VOICE, "no_speech_prob {:.2f} > {}".format(
            measures["no_speech_prob"],
            _need(cfg, "transcript_no_speech_prob_max")))
    if measures["repeat_ngram_ratio"] > float(
            _need(cfg, "transcript_repeat_ngram_ratio_max")):
        fail(CAT_LOOP, "repeated {}-gram ratio {:.2f} > {}".format(
            n, measures["repeat_ngram_ratio"],
            _need(cfg, "transcript_repeat_ngram_ratio_max")))
    # One word, not a phrase: "klinik klinik klinik" has no repeated 3-gram at
    # all, which is how V18's Malay decode passed every guard above.
    # (A one-word-share check was tried and dropped on 25 Sep: "the" is 22%
    # of a real English sentence in IMG_1160. The run check catches every
    # real loop seen so far.)
    run_min = int(_need(cfg, "transcript_word_run_min"))
    if ws["longest_run"] >= run_min:
        fail(CAT_LOOP, "word {!r} {} times in a row >= transcript_word_run_min={}"
             .format(ws["run_word"], ws["longest_run"], run_min))
    if not text:
        fail(CAT_NO_VOICE, "empty text")
    return (not reasons), "; ".join(reasons) or None, measures, cats


def mark_repetitions(scored, cfg):
    """Exclude consecutive near-identical segments beyond the repeat limit.

    A single "Ia tidak baik." is ordinary speech and no per-segment measure
    should condemn it - it is short, it compresses poorly, it is confidently
    decoded. Four of them in a row is the failure, and it is only visible by
    looking along the sequence. The first few are kept, because a person may
    genuinely repeat themselves; what cannot be true is a phrase continuing
    indefinitely.
    """
    limit = int(_need(cfg, "transcript_repeat_max_consecutive"))
    sim_min = float(_need(cfg, "transcript_repeat_similarity_min"))
    run_text, run_len = None, 0
    for s in scored:
        text = normalise(s["text"])
        if not text:
            run_text, run_len = None, 0
            continue
        if run_text is not None and difflib.SequenceMatcher(
                None, run_text, text).ratio() >= sim_min:
            run_len += 1
        else:
            run_text, run_len = text, 1
        if run_len > limit:
            s["ok"] = False
            reason = "repetition: segment {} of a consecutive run of the same "\
                     "phrase, limit is transcript_repeat_max_consecutive="\
                     "{}".format(run_len, limit)
            s["reason"] = "; ".join(x for x in (s.get("reason"), reason) if x)
            s.setdefault("categories", []).append(CAT_LOOP)
    return scored


def degenerate_flags(segs, cfg):
    """Which raw segments a loop guard fires on - per segment (compression,
    repeated n-grams, one word repeated) and along the sequence (the same
    phrase segment after segment). Used to decide which spans to re-decode
    without hotwords, and to judge whether the re-decode is better."""
    scored = []
    for seg in segs:
        _ok, _why, _m, cats = score_segment(seg, cfg, None)
        scored.append({"text": (seg.get("text") or "").strip(),
                       "ok": CAT_LOOP not in cats, "categories": list(cats)})
    mark_repetitions(scored, cfg)
    return [CAT_LOOP in s["categories"] for s in scored]


def degenerate_seconds(segs, flags):
    return sum(max(0.0, float(s["end"]) - float(s["start"]))
               for s, f in zip(segs, flags) if f)


def decode_score(segs, cfg):
    """Duration-weighted avg_logprob over the segments no loop guard fires
    on. A loop decodes with a deceptively GOOD log-probability (V17's
    '4-4-4' scores -0.11), so it must not count towards its language."""
    flags = degenerate_flags(segs, cfg)
    total = weighted = 0.0
    for s, f in zip(segs, flags):
        d = max(0.0, float(s["end"]) - float(s["start"]))
        lp = float(s.get("avg_logprob", 0.0))
        # 6 Nov 2026: mlx-whisper can return avg_logprob NaN for a segment;
        # one NaN made the whole language score NaN, which no comparison
        # handles and Postgres JSON rejects (4 videos failed). Unscored.
        if f or d <= 0 or not math.isfinite(lp):
            continue
        total += d
        weighted += d * lp
    return (round(weighted / total, 4) if total > 0 else None), round(total, 3)


def merge_languages(by_language, chosen, cfg):
    """Choose the language PER REGION OF TIME, not per video.

    Clinic speech is mixed: IMG_1160 runs "Jadi seperti yang ini, frosting.
    The white dots here." in one breath, so one language forced over a whole
    video has to be wrong for part of it.

    The obvious implementation - take the chosen language's segments and swap
    in better text from the other language - is wrong, and IMG_3424 shows why:
    forced to English it returns ONE segment spanning the clip, forced to Malay
    it returns four. Matching them up copies that single English sentence into
    all four Malay slots, and the transcript then contains the same words four
    times over. The two languages do not merely disagree about the words, they
    disagree about where the segments are.

    So the timeline is cut at every boundary either language proposes, each
    elementary interval goes to whichever language read it with the better
    avg_logprob, neighbouring intervals with the same winner are joined, and
    each region then emits THAT language's own segments - its own text, its own
    boundaries. A segment is placed by its midpoint, so it is emitted exactly
    once and no text is ever duplicated. The video's own language keeps a
    margin, so a tie does not cause a switch.
    """
    if not bool(_need(cfg, "transcription_per_segment_language")):
        spine = [dict(s, language=chosen) for s in (by_language.get(chosen) or [])]
        return spine, 0

    margin = float(_need(cfg, "transcript_language_switch_margin"))
    runs = {k: list(v or []) for k, v in by_language.items()}
    if not any(runs.values()):
        return [], 0

    edges = sorted({round(float(x), 3) for segs in runs.values() for s in segs
                    for x in (s["start"], s["end"])})
    if len(edges) < 2:
        return [dict(s, language=chosen) for s in (runs.get(chosen) or [])], 0

    def covering(segs, t):
        for s in segs:
            if float(s["start"]) <= t < float(s["end"]):
                return s
        return None

    regions = []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        best, best_score = None, None
        for lang, segs in runs.items():
            cov = covering(segs, mid)
            if cov is None:
                continue
            # the video's own language is only displaced by a clear margin
            score = float(cov["avg_logprob"]) + (margin if lang == chosen else 0.0)
            if best_score is None or score > best_score:
                best, best_score = lang, score
        if best is None:
            continue
        if regions and regions[-1][2] == best and abs(regions[-1][1] - a) < 1e-6:
            regions[-1] = (regions[-1][0], b, best)
        else:
            regions.append((a, b, best))

    out, switched = [], 0
    for a, b, lang in regions:
        for s in runs.get(lang) or []:
            mid = (float(s["start"]) + float(s["end"])) / 2.0
            if not (a <= mid < b):
                continue
            t = dict(s)
            t["start"] = round(max(float(s["start"]), a), 3)
            t["end"] = round(min(float(s["end"]), b), 3)
            if t["end"] <= t["start"]:
                continue
            t["language"] = lang
            if lang != chosen:
                t["language_switched_from"] = chosen
                switched += 1
            out.append(t)
    out.sort(key=lambda x: x["start"])
    return out, switched


def speech_ratio(segments, duration):
    """Share of the duration covered by segments Whisper returned at all."""
    if not duration:
        return 0.0
    spans = sorted((max(0.0, float(s["start"])),
                    min(float(duration), float(s["end"])))
                   for s in segments)
    total, cur_start, cur_end = 0.0, None, None
    for a, b in spans:
        if b <= a:
            continue
        if cur_start is None:
            cur_start, cur_end = a, b
        elif a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = a, b
    if cur_start is not None:
        total += cur_end - cur_start
    return min(1.0, total / float(duration))


# ---------------------------------------------------------------------------
# language
# ---------------------------------------------------------------------------

def choose_language(probabilities, cfg):
    """The best language among the allowed ones, and how sure that is.

    Whisper's raw confidence answers "which of a hundred languages is this?",
    which is the wrong question here: this clinic speaks two. Renormalising
    over the allowed set answers the question actually being asked, and it is
    the difference between keeping IMG_1160's real Malay (raw 0.45, which a
    blunt cutoff deletes) and keeping IMG_3413's noise.
    """
    allowed = list(_need(cfg, "transcription_allowed_languages"))
    min_prob = float(_need(cfg, "transcription_language_min_probability"))
    mass_min = float(_need(cfg, "transcription_allowed_mass_min"))
    probs = {k: float(v) for k, v in dict(probabilities).items()}
    among = {k: probs.get(k, 0.0) for k in allowed}
    chosen = max(among, key=lambda k: among[k]) if among else None
    mass = sum(among.values())
    renorm = (among[chosen] / mass) if (chosen and mass > 0) else 0.0
    top = sorted(probs.items(), key=lambda kv: -kv[1])[0] if probs else (None, 0.0)
    return {
        "detected": top[0],
        "detected_probability": round(float(top[1]), 4),
        "allowed_probabilities": {k: round(v, 6) for k, v in among.items()},
        "chosen": chosen,
        "chosen_probability_raw": round(among.get(chosen, 0.0), 6),
        "chosen_probability_renormalised": round(renorm, 4),
        # How much of the model's belief lands inside the allowed set at all.
        # Renormalising answers "which of en/ms is this?" but says nothing
        # about whether it is either: IMG_3413 is 93% 'en' among the allowed
        # while only 8% of the model's belief is allowed at all. Real Malay
        # (IMG_1160) carries 47%. So the mass is reported separately and is
        # what tells a confident choice from an arithmetic artefact.
        "allowed_mass": round(mass, 6),
        "confident": bool(renorm >= min_prob),
        "mass_ok": bool(mass >= mass_min),
        "trustworthy": bool(renorm >= min_prob and mass >= mass_min),
        "outside_allowed": bool(top[0] not in allowed),
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def assemble(raw, duration, cfg):
    """Turn raw Whisper output into the chunks that go in the table."""
    duration = round(float(duration), 3)
    out = {
        "outcome": None, "language": None, "language_probability": None,
        "chunks": [], "reason": None, "language_choice": raw.get("language_choice"),
        "excluded_segments": 0, "kept_segments": 0,
        "excluded_speech_share": 0.0, "speech_ratio": 0.0,
        "clamped_chunks": 0, "out_of_range_chunks": [],
        "vocabulary_applied": raw.get("vocabulary_applied"),
        "span_retries": raw.get("span_retries") or [],
        "language_decode_scores": raw.get("language_decode_scores"),
        "language_overridden": bool(raw.get("language_overridden")),
        "loop_lost_segments": 0, "loop_lost_seconds": 0.0,
    }

    if raw.get("status") == "EXCLUDED":
        out.update({"outcome": OUTCOME_EXCLUDED,
                    "reason": raw.get("reason") or "no audio stream",
                    "chunks": [_empty_chunk(duration, None, STATUS_EXCLUDED,
                                            raw.get("reason")
                                            or "no audio stream")]})
        return out
    if raw.get("status") == "ERROR":
        out.update({"outcome": OUTCOME_FAILED, "reason": raw.get("reason"),
                    "chunks": [_empty_chunk(duration, None, STATUS_FAILED,
                                            raw.get("reason"))]})
        return out

    segments = list(raw.get("segments") or [])
    choice = raw.get("language_choice") or {}
    lang = choice.get("chosen") or raw.get("language")
    out["language"] = lang
    out["language_probability"] = choice.get("chosen_probability_renormalised")
    out["speech_ratio"] = round(speech_ratio(segments, duration), 4)

    ctx = {"speech_ratio": out["speech_ratio"]}
    scored = []
    for seg in segments:
        ok, reason, measures, cats = score_segment(seg, cfg, ctx)
        start, end = float(seg["start"]), float(seg["end"])
        # A chunk running past the end of the video is clamped to it, and the
        # overrun recorded. It is NOT a reason to exclude the segment or to
        # mark the run incomplete: IMG_3424's last segment overran by 433ms
        # and the words in it are still the words that were said.
        c_start = min(max(0.0, start), duration)
        c_end = min(max(0.0, end), duration)
        if abs(c_start - start) > 1e-6 or abs(c_end - end) > 1e-6:
            out["clamped_chunks"] += 1
            out["out_of_range_chunks"].append(
                {"start": round(start, 3), "end": round(end, 3),
                 "clamped_to": [round(c_start, 3), round(c_end, 3)]})
        if c_end <= c_start:
            c_end = min(duration, c_start + 0.001)
        scored.append({
            "start": round(c_start, 3), "end": round(c_end, 3),
            "text": (seg.get("text") or "").strip(),
            "words": seg.get("words") or [],
            "language": seg.get("language") or lang,
            "ok": ok, "reason": reason, "measures": measures,
            "categories": list(cats),
            "confidence": _confidence(measures["avg_logprob"]),
        })

    mark_repetitions(scored, cfg)

    # Audio that is barely in an allowed language at all is not this clinic's
    # speech, whatever words come back. IMG_3413 puts 8% of the detector's
    # belief inside en/ms; every video here with real speech puts at least
    # 69% there. Nothing is deleted - the rows stay, with the reason on them.
    if not choice.get("mass_ok", True):
        for s in scored:
            note = ("language outside the allowed set: only {:.0%} of the "
                    "detector's belief falls inside {}".format(
                        choice.get("allowed_mass") or 0.0,
                        list(_need(cfg, "transcription_allowed_languages"))))
            s["ok"] = False
            s["reason"] = "; ".join(x for x in (s.get("reason"), note) if x)
            s.setdefault("categories", []).append(CAT_LANGUAGE)

    kept = [s for s in scored if s["ok"]]
    excluded = [s for s in scored if not s["ok"]]
    out["kept_segments"] = len(kept)
    out["excluded_segments"] = len(excluded)
    real_lost = [s for s in excluded
                 if not set(s.get("categories") or []) & set(NOT_REAL_SPEECH)]
    out["excluded_real_speech"] = len(real_lost)
    loops = [s for s in real_lost if CAT_LOOP in (s.get("categories") or [])]
    out["loop_lost_segments"] = len(loops)
    out["loop_lost_seconds"] = round(sum(s["end"] - s["start"] for s in loops), 3)
    spoken = sum(s["end"] - s["start"] for s in scored) or 0.0
    exc = sum(s["end"] - s["start"] for s in excluded) or 0.0
    out["excluded_speech_share"] = round(exc / spoken, 4) if spoken else 0.0

    if not segments:
        out.update({"outcome": OUTCOME_NO_SPEECH,
                    "reason": "voice activity filter found no speech",
                    "chunks": [_empty_chunk(duration, lang,
                                            STATUS_EMPTY_NO_SPEECH, None)]})
        return out

    min_ratio = float(_need(cfg, "min_speech_ratio"))
    if out["speech_ratio"] < min_ratio and not kept:
        out.update({
            "outcome": OUTCOME_MUSIC_ONLY,
            "reason": ("voice activity covers {:.1%} of the duration, below "
                       "min_speech_ratio={}, and no segment passed the quality"
                       " gates - treated as music or ambience, not speech"
                       .format(out["speech_ratio"], min_ratio)),
            "chunks": _chunks(scored)})
        return out

    cap = float(_need(cfg, "transcript_reject_share_max"))
    if not kept:
        out.update({
            "outcome": OUTCOME_NO_USABLE,
            "reason": (("audio is not in an allowed language: only {:.0%} of "
                        "the detector's belief falls inside {}".format(
                            choice.get("allowed_mass") or 0.0,
                            list(_need(cfg,
                                       "transcription_allowed_languages"))))
                       if not choice.get("mass_ok", True) else
                       ("every segment failed the quality gates; nothing is "
                        "accepted as speech")),
            "chunks": _chunks(scored)})
        return out
    if out["excluded_speech_share"] > cap:
        out.update({
            "outcome": OUTCOME_NO_USABLE,
            "reason": ("{:.0%} of speech time was excluded, above "
                       "transcript_reject_share_max={} - the transcript is "
                       "not trustworthy enough to stand as this asset's speech"
                       .format(out["excluded_speech_share"], cap)),
            "chunks": _chunks(scored)})
        return out

    out["outcome"] = OUTCOME_COMPLETE
    if not choice.get("trustworthy", True):
        out["reason"] = (
            "language {!r} chosen from {} but only {:.2f} confident after "
            "renormalising over the allowed languages, below "
            "transcription_language_min_probability={}".format(
                lang, list(_need(cfg, "transcription_allowed_languages")),
                choice.get("chosen_probability_renormalised") or 0.0,
                _need(cfg, "transcription_language_min_probability")))
    out["chunks"] = _chunks(scored)
    return out


def _confidence(avg_logprob):
    return round(min(1.0, max(0.0, math.exp(float(avg_logprob)))), 4)


def _empty_chunk(duration, lang, status, reason):
    return {"start": 0.0, "end": round(float(duration), 3), "text": "",
            "language": lang, "confidence": None,
            "transcription_status": status, "failure_reason": reason}


def _chunks(segs):
    """Kept and excluded rows together, in time order. The table is unique on
    (asset_id, start_seconds), so nothing may be emitted twice."""
    out = []
    for s in sorted(segs, key=lambda x: x["start"]):
        out.append({
            "start": s["start"], "end": s["end"], "text": s["text"],
            "words": s.get("words") or [],
            "language": s["language"], "confidence": s["confidence"],
            "transcription_status": (STATUS_COMPLETE if s["ok"]
                                     else STATUS_EXCLUDED),
            "failure_reason": s["reason"],
            "measures": s["measures"],
            "categories": list(s.get("categories") or []),
        })
    return out


def run_status(result, cfg):
    """(status, error_code, message) for the STEP2_TRANSCRIPT sub-run.

    A video is COMPLETED when Step 2 did its whole job on it, which includes
    the case where the honest answer is "there is no usable speech here".
    IMG_1147 is 189 seconds of room noise and IMG_3413 is a camera pointed at
    clothing; both are fully processed, and recording them as partial failures
    would mean the run table could never be clean however well the pipeline
    worked. PARTIAL is reserved for the one thing it should mean - real speech
    was found and then dropped - and FAILED for an actual error.

    analysis_runs requires an error_code on PARTIAL and forbids one on
    COMPLETED, so a COMPLETED run carries its explanation in error_message.
    """
    outcome = result.get("outcome")
    if outcome == OUTCOME_FAILED:
        return "FAILED", "TRANSCRIBE_FAILED", result.get("reason")

    # speech lost to a decoding loop that the no-hotword retry did not rescue
    if result.get("loop_lost_segments"):
        return "PARTIAL", "TRANSCRIPT_SPEECH_LOST_TO_LOOP", (
            "{} segment(s), {:.1f}s of voice-active audio, still decode as a "
            "loop after the no-hotword retry; that speech is not in the "
            "transcript".format(result["loop_lost_segments"],
                                result.get("loop_lost_seconds") or 0.0))

    # speech that was real and was nonetheless excluded - the only PARTIAL
    if result.get("excluded_real_speech"):
        return "PARTIAL", "TRANSCRIPT_SEGMENTS_EXCLUDED", (
            "{} of {} segments held speech and were still excluded".format(
                result["excluded_real_speech"],
                result["excluded_segments"] + result["kept_segments"]))

    if outcome in (OUTCOME_NO_USABLE, OUTCOME_MUSIC_ONLY, OUTCOME_NO_SPEECH,
                   OUTCOME_EXCLUDED):
        return "COMPLETED", None, result.get("reason")
    return "COMPLETED", None, result.get("reason")


# ---------------------------------------------------------------------------
# running the model
# ---------------------------------------------------------------------------

def transcribe_asset(s2, media, tmp_dir, cfg, duration, has_audio):
    """Raw Whisper output: language chosen from the allowed set, then forced."""
    if not has_audio:
        return {"status": "EXCLUDED", "reason": "no audio stream",
                "segments": []}
    wav = tmp_dir / "audio.wav"
    try:
        s2.extract_audio(media, wav)
    except Exception as exc:
        return {"status": "ERROR", "segments": [],
                "reason": "audio extraction failed: {}:{}".format(
                    type(exc).__name__, str(exc)[:160])}

    try:
        from faster_whisper.audio import decode_audio
        model = s2.whisper_model(cfg)
        audio = decode_audio(str(wav), sampling_rate=16000)

        forced = _need(cfg, "transcription_language")
        if forced:
            choice = {"detected": forced, "detected_probability": None,
                      "allowed_probabilities": {}, "chosen": forced,
                      "chosen_probability_raw": None,
                      "chosen_probability_renormalised": 1.0,
                      "confident": True, "outside_allowed": False,
                      "note": "transcription_language forces this language"}
        else:
            _lang, _prob, all_probs = model.detect_language(audio=audio)
            choice = choose_language(all_probs, cfg)

        vocab_args = vocabulary_arguments(cfg)
        # A single temperature and no fallback. faster-whisper otherwise
        # retries a window at rising temperatures when it dislikes its own
        # output, and those retries sample - which is why two identical runs
        # produced different transcripts for IMG_3424 and IMG_0476.
        temperatures = tuple(float(t) for t in
                             _need(cfg, "whisper_temperatures"))

        def run(language, vargs, clip=None, offset=0.0):
            """clip: a slice of the decoded audio (16 kHz) to decode instead
            of the whole file; its timestamps are shifted by offset."""
            segments, _info = model.transcribe(
                str(wav) if clip is None else clip,
                beam_size=int(_need(cfg, "whisper_beam_size")),
                vad_filter=bool(_need(cfg, "whisper_vad_filter")),
                condition_on_previous_text=bool(
                    _need(cfg, "whisper_condition_on_previous_text")),
                temperature=temperatures,
                language=language,
                word_timestamps=bool(_need(cfg, "transcription_word_timestamps")),
                **vargs)
            return [{
                "start": round(float(x.start) + offset, 3),
                "end": round(float(x.end) + offset, 3),
                "text": (x.text or "").strip(),
                # Word timings (29 Sep 2026) let the writer split a chunk that
                # crosses a scene boundary at the nearest gap between words.
                "words": [[round(float(w.start) + offset, 3),
                           round(float(w.end) + offset, 3), w.word]
                          for w in (getattr(x, "words", None) or [])],
                "avg_logprob": round(float(getattr(x, "avg_logprob", 0.0)), 4),
                "compression_ratio": round(
                    float(getattr(x, "compression_ratio", 0.0)), 4),
                "no_speech_prob": round(
                    float(getattr(x, "no_speech_prob", 0.0)), 4),
            } for x in segments]

        def degenerate(segs):
            """Whisper's own repetition measure over the whole transcript."""
            return compression_ratio(
                " ".join(x["text"] for x in segs)) > float(
                    _need(cfg, "transcript_compression_ratio_max"))

        def run_with_retry(language):
            """The vocabulary prompt, unless it makes the model degenerate.

            Whisper breaks its own repetition loops by retrying the window at
            a HIGHER TEMPERATURE, which samples - and sampling is exactly why
            two identical runs produced different transcripts. With the
            temperature pinned there is no such escape, and on this material
            the prompt itself is what tips three of the videos into a loop:
            IMG_3424 returns "Bagaimana dengan Klinik Dr Inder?" fourteen
            times over, at a compression ratio of 7.4 and, revealingly, a
            BETTER avg_logprob than the correct transcript.

            So the fallback is kept but made deterministic. The temperature
            and the beam never move; the one thing that varies is whether the
            vocabulary is offered, and it is withdrawn only from a transcript
            that has visibly collapsed. One rule, applied identically to every
            file, decided by that file's own compression ratio.
            """
            segs = run(language, vocab_args)
            if not vocab_args.get("initial_prompt") and not vocab_args.get(
                    "hotwords"):
                return segs, "vocabulary"
            if not degenerate(segs):
                return segs, "vocabulary"
            plain = run(language, {"initial_prompt": None, "hotwords": None})
            if degenerate(plain) and not degenerate(segs):
                return segs, "vocabulary"
            return plain, "vocabulary_withdrawn"

        vocab_on = bool(vocab_args.get("initial_prompt")
                        or vocab_args.get("hotwords"))
        sr = 16000

        def run_with_span_retry(language):
            """run_with_retry, then: every span a loop guard fires on is
            decoded ONCE more without hotwords, and the re-decode replaces it
            only if it is strictly less degenerate. Hotwords stay on for the
            rest of the file - V9's product names need them - and are taken
            away only where they visibly broke the decode (V17's '4-4-4')."""
            segs, used = run_with_retry(language)
            retries = []
            if not vocab_on or used != "vocabulary" or not segs:
                return segs, used, retries
            flags = degenerate_flags(segs, cfg)
            spans, i = [], 0
            while i < len(segs):
                if flags[i]:
                    j = i
                    while j + 1 < len(segs) and flags[j + 1]:
                        j += 1
                    spans.append((i, j))
                    i = j + 1
                else:
                    i += 1
            out = list(segs)
            for i0, i1 in reversed(spans):
                a = max(0.0, float(segs[i0]["start"]))
                b = min(float(duration), float(segs[i1]["end"]))
                clip = audio[int(a * sr):int(math.ceil(b * sr))]
                sub = []
                if len(clip) > 0 and b > a:
                    sub = run(language, {"initial_prompt": None,
                                         "hotwords": None}, clip=clip, offset=a)
                    for s in sub:
                        s["start"] = round(min(max(s["start"], a), b), 3)
                        s["end"] = round(min(max(s["end"], a), b), 3)
                    sub = [s for s in sub if s["end"] > s["start"]]
                before = degenerate_seconds(segs[i0:i1 + 1], flags[i0:i1 + 1])
                after = degenerate_seconds(sub, degenerate_flags(sub, cfg))
                kept = bool(sub) and after < before
                retries.append({
                    "language": language, "span": [round(a, 3), round(b, 3)],
                    "segments_before": i1 - i0 + 1, "segments_after": len(sub),
                    "loop_seconds_before": round(before, 3),
                    "loop_seconds_after": round(after, 3),
                    "kept": "retry" if kept else "original",
                    "text_before": " ".join(s["text"] for s in segs[i0:i1 + 1])[:200],
                    "text_after": " ".join(s["text"] for s in sub)[:200]})
                if kept:
                    for s in sub:
                        s["span_retry"] = "no_hotwords"
                    out[i0:i1 + 1] = sub
            if any(r["kept"] == "retry" for r in retries):
                used = "vocabulary+span_retry"
            return out, used, list(reversed(retries))

        allowed = list(_need(cfg, "transcription_allowed_languages"))
        overridden, decode_scores = False, None
        if bool(_need(cfg, "transcription_per_segment_language")) and not forced:
            by_language, used, span_retries = {}, {}, []
            for c in allowed:
                by_language[c], used[c], r = run_with_span_retry(c)
                span_retries += r
            vocab_used = used.get(choice["chosen"])
        else:
            chosen = choice["chosen"]
            decodes = {chosen: run_with_span_retry(chosen)}
            if not forced and len(allowed) > 1:
                # The detector listens to the first 30 s and can be wrong with
                # confidence: V18 is English and it said Malay at 0.96. So the
                # other allowed language is decoded too, and replaces the
                # detector's choice only by a clear margin - Whisper forced to
                # English translates Malay fluently (V17), which a plain
                # best-score rule would mistake for a better reading.
                margin = float(_need(cfg, "transcript_language_override_margin"))
                for c in allowed:
                    if c != chosen:
                        decodes[c] = run_with_span_retry(c)
                decode_scores = {}
                for c, (s, _u, _r) in decodes.items():
                    sc, secs = decode_score(s, cfg)
                    decode_scores[c] = {"avg_logprob": sc, "scored_seconds": secs,
                                        "segments": len(s)}
                base = decode_scores[chosen]["avg_logprob"]
                others = [(c, v["avg_logprob"]) for c, v in decode_scores.items()
                          if c != chosen and v["avg_logprob"] is not None]
                if base is not None and others:
                    best_c, best_sc = max(others, key=lambda kv: kv[1])
                    if best_sc - base >= margin:
                        probs = choice.get("allowed_probabilities") or {}
                        mass = sum(probs.values()) or 0.0
                        choice = dict(
                            choice, detector_chosen=chosen, chosen=best_c,
                            chosen_probability_raw=probs.get(best_c),
                            chosen_probability_renormalised=round(
                                probs.get(best_c, 0.0) / mass, 4) if mass else 0.0,
                            confident=True,
                            trustworthy=bool(choice.get("mass_ok", True)),
                            note="detector chose {!r}; decoding in {!r} scored "
                                 "{:.3f} vs {:.3f} avg_logprob, better by >= "
                                 "transcript_language_override_margin={}".format(
                                     chosen, best_c, best_sc, base, margin))
                        overridden = True
                choice = dict(choice, decode_scores=decode_scores)
            segs0, vocab_used, span_retries = decodes[choice["chosen"]]
            by_language = {choice["chosen"]: segs0}
        segs, switched = merge_languages(by_language, choice["chosen"], cfg)
        return {"status": "OK", "language": choice["chosen"],
                "language_probability":
                    choice["chosen_probability_renormalised"],
                "language_choice": choice, "segments": segs,
                "vocabulary_applied": vocab_used,
                "language_switched_segments": switched,
                "span_retries": span_retries,
                "language_decode_scores": decode_scores,
                "language_overridden": overridden}
    except Exception as exc:
        return {"status": "ERROR", "segments": [], "reason": "{}:{}".format(
            type(exc).__name__, str(exc)[:200])}
    finally:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass
