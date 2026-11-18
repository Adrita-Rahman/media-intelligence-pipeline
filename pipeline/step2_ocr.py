#!/usr/bin/env python3
"""
On-screen text: read more of it, believe less of it. (step2_ocr 3.1.0)

3.1.0 (30 Sep 2026): a token is judged in its line - edge punctuation
stripped, contractions and hyphenated compounds read as words, a number
beside a quantity word is real, ocr_domain_terms adds the clinic's medical
vocabulary - and a logo guard (ocr_logo_lockup) rejects partial readings of
the clinic's logo lockup as LOGO_FRAGMENT and short standalone fragments as
WORD_FRAGMENT. See line_token_flags and logo_fragment.

2.0.0 missed text that was there and accepted text that was not. The
diagnosis (reports/ocr_diagnosis.md) traced every missed term: mostly
engine misreads, then pipeline rules - a unit dropped in pooling, a setting
under the confidence floor, a logo glued onto a caption, a poster headline
never joined, a legible term lost inside unreadable small print. 3.0.0:

  ENGINE. RapidOCR (PP-OCRv6), chosen on a measured benchmark
  (ocr_engine_benchmark); small text is re-read from an enlarged crop
  (ocr_engine.read_frame). No AI vision or language model anywhere.

  COVERAGE. coverage_frames replaces the 24-frame cap: every keyframe up to
  ocr_full_coverage_max_seconds, then every <=2 s plus changed frames.

  POOLING. Variant readings are the same reading when one box sits inside
  the other (intersection over the smaller box), not only on IoU.

  LINES AND BLOCKS. Boxes join a line only at a similar letter height and
  within a fraction of a letter height vertically; stacked lines join a
  block by height, gap and left/centre alignment, anywhere in the frame. A
  merged line that fails the word test is split back into its parts.

  TIDYING. NFKC (the numeral U+2162 becomes III), a model number split from
  its name ('HELIOS785' -> 'HELIOS 785'), a unit split from its number
  ('0.4J' -> '0.4 J'). Letters are never changed by tidying.

  NUMBERS. A reading that is only numbers and units is accepted only next
  to a known device or screen term in the same frame - '785', '0.4 J' - and
  rejected BARE_NUMBER_NO_CONTEXT otherwise. There is no digit-count rule.

  ACCEPTANCE. Confidence floor AND every token a word, a known term, or a
  number/unit (ocr_min_real_word_share 1.0) AND persistence (or a confident
  known-term match). A non-word inside an otherwise real line may be
  corrected to the ONE word-list word within one edit (IRONT -> FRONT);
  a lone token never is.

VOCABULARY NORMALISATION, with the safeguards that make it acceptable:
raw_text is NEVER altered; only normalized_text carries a known term or a
dictionary correction, only on a tight match, and every substitution is
recorded on the row (bounding_box.normalisation) and in the run metadata.
Every result is also scored with normalisation switched off
(without_normalisation), so what the engine really read stays visible. The
vocabulary is ocr_known_terms plus the treatments reference tables - never
the answer key.
"""
from __future__ import annotations

import math
import re


def _need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: " + key)
    return cfg[key]


def normalise_text(raw):
    return " ".join((raw or "").upper().split())


def box_metrics(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
            "cx": (min(xs) + max(xs)) / 2.0,
            "cy": (min(ys) + max(ys)) / 2.0,
            "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


# ---------------------------------------------------------------------------
# pooling the pre-processing variants
# ---------------------------------------------------------------------------

def _overlap(a, b):
    """Intersection over the SMALLER box.

    IoU was the wrong measure. The native pass boxed V9's second caption line
    as 'their phone at the iront deska)' 0.80 of the frame wide, the contrast
    pass boxed 'phone at the iront desk' 0.61 wide: the same patch of screen,
    an IoU well under 0.5, so pooling kept both and the line grouper glued
    them into one line carrying the caption twice. One box sitting inside
    another is the same reading."""
    x0 = max(a["x0"], b["x0"]); x1 = min(a["x1"], b["x1"])
    y0 = max(a["y0"], b["y0"]); y1 = min(a["y1"], b["y1"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    ub = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    den = min(ua, ub)
    return (inter / den) if den > 0 else 0.0


def pool_variants(per_variant, frame_w, frame_h, cfg):
    """One list of readings per frame, from however many variants were run.

    Two readings are the same reading when their boxes are the same box. That
    is a change from matching on the TEXT, and it had to change: the whole
    point of a second pass is that it reads the same patch of screen
    DIFFERENTLY. V9's caption line comes back as "their phone at the iront
    deska)" at 0.38 from the native pass and "phone at the iront desk" at 0.74
    from the contrast pass. Pooling on text leaves both, and the line grouper
    then concatenates them into one line containing the caption twice.

    So overlapping boxes are one reading and the most confident wins, with the
    variants that found it recorded - "only the contrast pass could read this"
    is worth knowing when the device screens are reviewed.
    """
    tol = float(_need(cfg, "ocr_persistence_position_tolerance"))
    overlap_min = float(_need(cfg, "ocr_variant_box_overlap_min"))
    diag = math.hypot(frame_w, frame_h)
    pooled = []
    for variant, readings in (per_variant or {}).items():
        for r in readings or []:
            m = box_metrics(r["points"])
            text = normalise_text(r["raw_text"])
            if not text:
                continue
            hit = None
            for q in pooled:
                same_box = _overlap(q["m"], m) >= overlap_min
                same_text = (q["normalized"] == text and math.hypot(
                    q["m"]["cx"] - m["cx"], q["m"]["cy"] - m["cy"])
                    <= tol * diag)
                if same_box or same_text:
                    hit = q
                    break
            if hit is None:
                pooled.append({"raw_text": r["raw_text"], "normalized": text,
                               "confidence": float(r["confidence"]),
                               "points": r["points"], "m": m,
                               "variants": [variant]})
            else:
                if variant not in hit["variants"]:
                    hit["variants"].append(variant)
                if float(r["confidence"]) > hit["confidence"]:
                    hit.update({"raw_text": r["raw_text"],
                                "normalized": text,
                                "confidence": float(r["confidence"]),
                                "points": r["points"], "m": m})
    return pooled


# ---------------------------------------------------------------------------
# vocabulary, words and numbers
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    """Plain edit distance. Small strings, so the simple table is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def known_term_groups(cfg):
    """{group: [term, ...]} from ocr_known_terms - the clinic's device,
    product, screen and clinic-name list, the one place the vocabulary lives.
    It is reference data curated by the operator and must never be built
    from the answer key."""
    groups = _need(cfg, "ocr_known_terms") or {}
    if isinstance(groups, str):
        import json as _json
        groups = _json.loads(groups)
    return groups


def known_terms(cfg, treatments=None, aliases=None):
    """Every term this clinic is known to put on screen, upper case.

    Sources: ocr_known_terms, plus the treatments and treatment_aliases
    reference tables. Nothing here is derived from the answer key - a
    vocabulary built from the expected answers would make the scoring
    meaningless.
    """
    terms = set()
    for group in known_term_groups(cfg).values():
        for t in group:
            k = normalise_text(str(t))
            if k:
                terms.add(k)
    for t in list(treatments or []) + list(aliases or []):
        k = normalise_text(str(t))
        if k:
            terms.add(k)
    return terms


def load_words(cfg):
    """The en/ms word list, lower case. See the file for what it is not."""
    from pathlib import Path as _P
    path = _P(_need(cfg, "ocr_wordlist_path"))
    if not path.is_absolute():
        path = _P(__file__).resolve().parent.parent / path
    words = set()
    if not path.exists():
        raise SystemExit("ocr_wordlist_path does not exist: {}".format(path))
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for w in line.split():
            w = re.sub(r"[^A-Za-z-]", "", w).lower()
            if w:
                words.add(w)
    return words


def clean_number(token):
    """A reading of a number with the marks the camera put beside it removed.

    A device screen reading "0.4,-" is the setting 0.4 with a stray tick, and
    "0.0, ^" the same. raw_text keeps the tick; normalized_text should not,
    because normalized_text is what gets searched.
    """
    t = re.sub(r"^[^0-9+-]+|[^0-9%A-Za-z]+$", "", (token or "").strip())
    return t


NUMBER_RE = re.compile(
    r"^[+-]?\d{1,4}(?:[.,]\d{1,3})?\s*"
    r"(?:%|J|MJ|W|MM|CM|S|MS|HZ|KHZ|MHZ|C|F|X)?$", re.I)


def is_number(token):
    """A setting a device would actually show: 0.4, 0.4J, 80%, 785, 12mm.

    A lone digit is NOT one. '4' on its own carried no meaning in any frame it
    appeared in, and accepting it as a reading of the screen would be
    accepting noise that happens to look like a number.
    """
    t = (token or "").strip().strip(".,:;-_~^*'\"")
    if not t:
        return False
    digits = sum(c.isdigit() for c in t)
    if digits == 0:
        return False
    if len(t) < 2:
        return False
    return bool(NUMBER_RE.match(t))


def match_known_term(text, terms, cfg):
    """(term, distance) for a tight match, or None.

    Exact first. Then the closest term within both limits - close enough that
    the reading is the term misread, not a different string that happens to
    be nearby.
    """
    key = normalise_text(text)
    if not key:
        return None
    if key in terms:
        return (key, 0)
    if not bool(_need(cfg, "ocr_normalize_to_known_terms")):
        return None
    max_d = int(_need(cfg, "ocr_normalize_max_edit_distance"))
    max_len = int(_need(cfg, "ocr_normalize_max_length_delta"))
    min_term = int(_need(cfg, "ocr_normalize_min_term_length"))
    hits = []
    for t in sorted(terms):
        # A short term cannot afford a fuzzy match. FUE is three characters, so
        # at a flat distance of 2 it swallows CU, CUF, CUN, CUT and FR - five
        # pieces of noise promoted into a treatment code. Below min_term a
        # reading has to match exactly, and above it the tolerance grows with
        # the term, never exceeding the configured ceiling.
        if len(t) < min_term:
            continue
        if abs(len(t) - len(key)) > max_len:
            continue
        allowed = min(max_d, max(1, len(t) // 4))
        d = levenshtein(key, t)
        if d <= allowed:
            hits.append((d, t))
    if not hits:
        return None
    best_d = min(d for d, _t in hits)
    nearest = [t for d, t in hits if d == best_d]
    # A tie is ambiguous, not a choice to make. 'ULTRAFORMER I' is two edits
    # from ULTRAFORMER and two from ULTRAFORMER III; picking one depended on
    # set iteration order and made two identical dry runs disagree. Ambiguous
    # readings are not normalised - the same rule as dictionary correction.
    if len(nearest) != 1:
        return None
    return (nearest[0], best_d)


def unit_tokens(cfg):
    return {str(u).upper() for u in (_need(cfg, "ocr_unit_tokens") or [])}


LETTER_DIGIT = re.compile(r"(?<=[A-Z]{2})(?=\d)|(?<=\d)(?=[A-Z]{2})")


def tidy_text(text, cfg):
    """Settle how a reading is WRITTEN, without changing what it says.

    Two things a recogniser routinely gets wrong about spacing, and nothing
    else: a model number run into its name ('HELIOS785' is HELIOS 785) and a
    setting run into its unit ('0.4J' is 0.4 J). Letters are never changed
    here - that is what raw_text / normalized_text distinguish - and only
    the configured unit tokens are split off a number."""
    import unicodedata
    # Unicode compatibility forms first: the recogniser returns the device's
    # stylised numeral as the single character U+2162 (ROMAN NUMERAL THREE);
    # NFKC writes it as the three letters III that a search will contain.
    key = normalise_text(unicodedata.normalize("NFKC", text or ""))
    # Trademark marks are not text a search contains: 'M-CURE®' is M-CURE.
    key = normalise_text(re.sub("[®™©]", " ", key))
    if not bool(_need(cfg, "ocr_split_letter_digit_boundaries")):
        return key
    units = unit_tokens(cfg)
    out = []
    for tok in key.split():
        m = re.fullmatch(r"([+-]?\d+(?:[.,]\d+)?)([A-Z%]+)", tok)
        if m and m.group(2) in units:
            out.extend([m.group(1), m.group(2)])
            continue
        out.extend(LETTER_DIGIT.sub(" ", tok).split())
    return " ".join(out)


def is_unit(token, cfg):
    return (re.sub(r"[^A-Z%]", "", (token or "").upper()) in unit_tokens(cfg)
            and bool(re.fullmatch(r"[A-Za-z%]+[.,:;]?", token or "")))


def numeric_line(text, cfg):
    """True when a reading is nothing but numbers and their units - a
    setting, which only means something next to the device showing it."""
    toks = [t for t in normalise_text(text).split()
            if re.search(r"[A-Za-z0-9]", t)]
    if not toks:
        return False
    return (all(is_number(clean_number(t)) or is_unit(t, cfg) for t in toks)
            and any(is_number(clean_number(t)) for t in toks))


def dictionary_correct(token, words, cfg):
    """(word, distance) when exactly one word-list word is within the
    configured edit distance of `token`, at the same length; else None.

    IRONT has one such neighbour, FRONT, and becomes it. A token with two
    neighbours is ambiguous and is not corrected - it stays unknown and the
    line fails the real-word test, which is the safe outcome. The raw
    reading is never touched; only normalized_text carries the correction,
    and the correction is recorded on the row."""
    if not bool(_need(cfg, "ocr_dictionary_correction_enabled")):
        return None
    alpha = re.sub(r"[^A-Za-z]", "", token or "").lower()
    if len(alpha) < int(_need(cfg, "ocr_dictionary_correct_min_length")):
        return None
    if alpha != (token or "").lower().strip(".,:;!?'\"()"):
        return None
    max_d = int(_need(cfg, "ocr_dictionary_correct_max_distance"))
    hits = [w for w in words if len(w) == len(alpha)
            and levenshtein(alpha, w) <= max_d]
    if len(hits) != 1:
        return None
    return hits[0], levenshtein(alpha, hits[0])


def token_ok(token, terms, words, cfg):
    """Does this one token say anything?"""
    bare = re.sub(r"[^A-Za-z0-9.%-]", "", token or "")
    if not bare:
        return False
    if is_number(bare) or is_unit(bare, cfg):
        return True
    alpha = re.sub(r"[^A-Za-z]", "", bare).lower()
    if len(alpha) < int(_need(cfg, "ocr_min_token_length")):
        # Real one-letter words ('a', 'i') inside a line: 'WITH A MUSTACHE?'.
        # A line that is only such a token is still refused by
        # ocr_min_standalone_token_length in evaluate_text.
        return alpha in {w.lower() for w in _need(cfg, "ocr_short_words")}
    if alpha in words:
        return True
    return match_known_term(bare, terms, cfg) is not None


# ---------------------------------------------------------------------------
# 3.1.0: judging a token IN ITS LINE
# ---------------------------------------------------------------------------
# C1 (the fractional CO2 laser video) had real captions read at 0.99-1.00
# rejected because one token failed: DON'TS lost its apostrophe and became
# DONTS, NEWLY-PRODUCED lost its hyphen and became NEWLYPRODUCED, the 5 of
# 'AFTER 5 DAYS' is a single digit, CO2 is tidied to 'CO 2', and
# MICROABLATION / EPIDERMIS / RETINOID are not in a general word list. The
# every-word rule (ocr_min_real_word_share 1.0) is unchanged; what changed is
# how a token is read before it is judged. raw_text is never altered.

APOSTROPHES = "’‘ʼ`"


def _key(token):
    """A token as the vocabulary compares it: letters and digits, upper case.
    'BEAUTY&' is BEAUTY, 'DR.' is DR, 'I'M' is IM."""
    return re.sub(r"[^A-Z0-9]", "", (token or "").upper())


def _edge_strip(token, cfg):
    """The token without the punctuation a caption puts around a word
    (ocr_token_edge_punctuation), curly apostrophes made straight."""
    t = token or ""
    for a in APOSTROPHES:
        t = t.replace(a, "'")
    return t.strip(str(_need(cfg, "ocr_token_edge_punctuation")))


def _phrase(text, cfg):
    """A vocabulary entry as a tuple of keys, tidied the way a reading is
    ('CO2' is written 'CO 2' by tidy_text, so it must be compared so)."""
    return tuple(k for k in (_key(x) for x in tidy_text(text, cfg).split()) if k)


def domain_terms(cfg):
    """(words, phrases) from ocr_domain_terms.

    words   single keys that are real text anywhere (MICROABLATION).
    phrases tuples of keys that are real only TOGETHER: 'CO 2', and the
            doctor's name - LEW, KAR and MIN count as a name only inside
            'DR LEW KAR MIN' / 'DOCTOR LEW', never on their own.
    An entry that tidies to more than one token is a phrase."""
    d = _need(cfg, "ocr_domain_terms") or {}
    if isinstance(d, str):
        import json as _json
        d = _json.loads(d)
    words, phrases = set(), set()
    entries = list(d.get("words") or []) + list(d.get("phrases") or [])
    entries += list((d.get("names") or {}).get("contexts") or [])
    for e in entries:
        p = _phrase(str(e), cfg)
        if len(p) == 1:
            words.add(p[0])
        elif p:
            phrases.add(p)
    return words, phrases


def phrase_spans(keys, phrases):
    """[(start, end)] of every occurrence of each phrase in `keys`."""
    spans = []
    for p in phrases:
        n = len(p)
        if not n:
            continue
        for i in range(len(keys) - n + 1):
            if tuple(keys[i:i + n]) == p:
                spans.append((i, i + n))
    return spans


def token_real(token, terms, words, cfg, dom_words, contractions):
    """Does this token say something on its own, after its surrounding
    punctuation is removed? A word, known term or number (token_ok), an
    English contraction (ocr_contractions, or a real word + 's), a domain
    word (ocr_domain_terms), or a hyphenated compound whose EVERY part is
    real (NEWLY-PRODUCED is NEWLY + PRODUCED; NEWLY-XQZ is not)."""
    t = _edge_strip(token, cfg)
    if not t or not re.search(r"[A-Za-z0-9]", t):
        return False
    if token_ok(t, terms, words, cfg):
        return True
    up = t.upper()
    if up in contractions:
        return True
    m = re.fullmatch(r"(.+)'S", up)
    if m and token_real(m.group(1), terms, words, cfg, dom_words, contractions):
        return True
    if _key(t) in dom_words:
        return True
    if "-" in t.strip("-"):
        parts = t.split("-")
        return all(p and token_real(p, terms, words, cfg, dom_words,
                                    contractions) for p in parts)
    return False


NUMBERISH = re.compile(r"\d{1,4}(?:[.,]\d{1,3})?(?:[-–]\d{1,4}(?:[.,]\d{1,3})?)?")


def quantity_numbers(tokens, cfg):
    """Indices of number tokens made real by a time/quantity word next to
    them in the same line (ocr_quantity_words): 'AFTER 5 DAYS', '2 WEEKS
    BEFORE', the range '5 7 DAYS' or '5-7 DAYS', 'SESSION 3'. A run of more
    than ocr_quantity_number_max_run numbers is not a quantity. This is
    separate from ocr_bare_number_requires_device_context, which applies
    only to a line made of nothing but numbers and units."""
    qwords = {_key(w) for w in _need(cfg, "ocr_quantity_words")}
    max_run = int(_need(cfg, "ocr_quantity_number_max_run"))
    isnum = [bool(NUMBERISH.fullmatch(_edge_strip(t, cfg))) for t in tokens]
    keys = [_key(t) for t in tokens]
    ok, i = set(), 0
    while i < len(tokens):
        if not isnum[i]:
            i += 1
            continue
        j = i
        while j < len(tokens) and isnum[j]:
            j += 1
        if j - i <= max_run and ((j < len(tokens) and keys[j] in qwords)
                                 or (i > 0 and keys[i - 1] in qwords)):
            ok.update(range(i, j))
        i = j
    return ok


def line_token_flags(tokens, terms, words, cfg):
    """Per token: does it say something, judged in its line? A token is real
    on its own (token_real), or as part of a multi-word known term or domain
    phrase read in full ('DR INDER', 'CO 2', 'DR LEW KAR MIN'), or as a
    number beside a quantity word."""
    dom_words, dom_phrases = domain_terms(cfg)
    contractions = {_edge_strip(str(c), cfg).upper()
                    for c in _need(cfg, "ocr_contractions")}
    flags = [token_real(t, terms, words, cfg, dom_words, contractions)
             for t in tokens]
    keys = [_key(t) for t in tokens]
    phrases = set(dom_phrases)
    phrases |= {_phrase(t, cfg) for t in terms if " " in t}
    for a, b in phrase_spans(keys, phrases):
        for i in range(a, b):
            flags[i] = True
    for i in quantity_numbers(tokens, cfg):
        flags[i] = True
    return flags


def logo_lockup(cfg):
    """(sequences, anchors, min_fragment_letters) from ocr_logo_lockup."""
    d = _need(cfg, "ocr_logo_lockup") or {}
    if isinstance(d, str):
        import json as _json
        d = _json.loads(d)
    seqs = [p for p in (_phrase(s, cfg) for s in d.get("sequences") or []) if p]
    anchors = {p for p in (_phrase(s, cfg) for s in d.get("anchor_terms") or [])
               if p}
    return seqs, anchors, int(d.get("min_fragment_letters", 2))


def logo_fragment(norm, terms, cfg):
    """None when a reading that says something may stand; else (code, why).

    Every token of 'THINK DRINDER MNG BEAUTY& HEALTH' is in the word list
    (mng is), so the word test cannot see that it is the clinic's logo
    lockup misread. This checks the reading against the lockup itself
    (ocr_logo_lockup):

      LOGO_FRAGMENT  a reading made only of lockup words that is not a known
                     term and not the whole lockup (HEALTH, BEAUTY HEALTH,
                     KLINIK DRINDER HEALTH); a reading holding a clinic
                     anchor term plus another lockup word outside it
                     (THINK DRINDER MNG BEAUTY& HEALTH); an anchor term with
                     the neighbouring lockup word cut short beside it
                     (NIK DR INDER, IK DR INDER); a short standalone token
                     that is a piece of a lockup word (ING of REDEFINING).
      WORD_FRAGMENT  any other standalone token of at most
                     ocr_standalone_fragment_max_letters letters that is not
                     a known term (MIN on its own).

    A caption that names the clinic in a sentence stands: 'OR VISIT US AT
    THE KLINIK DR INDER FOR FREE CONSULTATION.' has no other lockup word."""
    key = normalise_text(norm)
    if not key or key in terms:
        return None
    tokens = [t for t in key.split() if re.search(r"[A-Za-z0-9]", t)]
    keys = [_key(t) for t in tokens]
    if not keys:
        return None
    seqs, anchors, min_frag = logo_lockup(cfg)
    lock = {k for s in seqs for k in s}
    if tuple(keys) in seqs:
        return None
    if lock and all(k in lock for k in keys):
        return ("LOGO_FRAGMENT", "{!r} is only words of the logo lockup, "
                "not a known term and not the whole lockup".format(key[:48]))
    if len(keys) == 1:
        k = keys[0]
        if k.isalpha() and len(k) <= int(
                _need(cfg, "ocr_standalone_fragment_max_letters")):
            if any(k in w and k != w for w in lock):
                return ("LOGO_FRAGMENT", "{!r} standing alone is a piece of a "
                        "logo lockup word".format(key))
            return ("WORD_FRAGMENT", "{!r} standing alone is at most "
                    "ocr_standalone_fragment_max_letters letters and not a "
                    "known term".format(key))
        return None
    # Words of OTHER known terms are not lockup words here: DR in 'I'M DR LEW
    # HERE FROM KLINIK DR INDER' belongs to DR LEW, not to the lockup.
    others = {_phrase(t, cfg) for t in terms} - anchors
    masked = set()
    for a, b in phrase_spans(keys, others):
        masked.update(range(a, b))
    # Two or more consecutive lockup words that are not a whole anchor term
    # or the whole lockup are the lockup read in part, whatever surrounds
    # them: 'MNG BEAUTY& HEALTH', 'DR INDER REDEFINING'. 'YOUR BEAUTY AND
    # HEALTH' has no such run.
    spans = phrase_spans(keys, anchors)
    for a, b in _lockup_runs(keys, seqs, masked):
        run = tuple(keys[a:b])
        if run in seqs or any(x <= a and b <= y for x, y in spans):
            continue
        return ("LOGO_FRAGMENT", "{!r} reads part of the logo lockup ({})"
                .format(key[:48], " ".join(run)))
    if not spans:
        return None
    covered = set(masked)
    for a, b in spans:
        covered.update(range(a, b))
    for i, k in enumerate(keys):
        if i not in covered and k in lock:
            return ("LOGO_FRAGMENT", "{!r} holds a clinic name plus another "
                    "logo lockup word ({}) - a partial reading of the lockup"
                    .format(key[:48], k))
    return _cut_lockup(key, keys, spans, seqs, min_frag)


def _lockup_runs(keys, seqs, skip):
    """Maximal runs of 2+ consecutive tokens (none in `skip`) that read a
    consecutive stretch of a lockup sequence."""
    def in_seq(t):
        return any(tuple(s[p:p + len(t)]) == t
                   for s in seqs for p in range(len(s) - len(t) + 1))
    runs, i = [], 0
    while i < len(keys):
        j = i
        while j < len(keys) and j not in skip and in_seq(tuple(keys[i:j + 1])):
            j += 1
        if j - i >= 2:
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _cut_lockup(key, keys, spans, seqs, min_frag):
    """An anchor term with the lockup word next to it cut short."""
    for a, b in spans:
        anchor = tuple(keys[a:b])
        for s in seqs:
            for p in range(len(s) - len(anchor) + 1):
                if tuple(s[p:p + len(anchor)]) != anchor:
                    continue
                if a > 0 and p > 0:
                    k, w = keys[a - 1], s[p - 1]
                    if len(k) >= min_frag and k != w and w.endswith(k):
                        return ("LOGO_FRAGMENT", "{!r}: {} is {} cut short "
                                "beside the clinic name".format(key[:48], k, w))
                end = p + len(anchor)
                if b < len(keys) and end < len(s):
                    k, w = keys[b], s[end]
                    if len(k) >= min_frag and k != w and w.startswith(k):
                        return ("LOGO_FRAGMENT", "{!r}: {} is {} cut short "
                                "beside the clinic name".format(key[:48], k, w))
    return None


def evaluate_text(text, terms, words, cfg, context=None):
    """(says_something, normalized_text, normalisation or None).

    A single reading is judged as a whole first, because a device label is
    one term. Otherwise it is judged token by token, and EVERY token has to
    say something - a word from the en/ms list, a known term, or a number
    with its unit - at ocr_min_real_word_share of 1.0. 2.0.0 let a caption
    through at 60%, which is how 'THEIR PHONE AT THE IRONT' was accepted
    with IRONT in it.

    A reading made only of numbers and units always 'says something' here;
    whether it is ACCEPTED depends on a device label beside it in the same
    frame, which only build_observations can see (see numeric_line).
    """
    key = tidy_text(text, cfg)
    if not key:
        return False, key, None
    whole = match_known_term(key, terms, cfg)
    if whole is not None:
        term, dist = whole
        norm = None
        if dist > 0:
            norm = {"kind": "known_term", "from": key, "to": term,
                    "distance": dist,
                    "length_delta": abs(len(term) - len(key))}
        return True, term, norm

    # Punctuation on its own makes no claim, so it neither helps nor hurts.
    tokens = [t for t in key.split() if t and re.search(r"[A-Za-z0-9]", t)]
    if not tokens:
        return False, key, None

    if numeric_line(key, cfg):
        return True, " ".join(clean_number(t) if is_number(clean_number(t))
                              else re.sub(r"[^A-Z%]", "", t)
                              for t in tokens), None

    # A two-letter fragment is not a reading. V2's "OR" is a real word and
    # still tells nobody anything; standing alone it is a piece of a word the
    # camera cut in half.
    if len(tokens) == 1:
        alpha = re.sub(r"[^A-Za-z]", "", tokens[0])
        if alpha and len(alpha) < int(
                _need(cfg, "ocr_min_standalone_token_length")):
            if match_known_term(tokens[0], terms, cfg) is None:
                return False, key, None

    ok_flags = line_token_flags(tokens, terms, words, cfg)
    # Correction needs context. A lone token has none: V9's 'UNGER' is
    # YOUNGER with the first letters behind a head, and the word list's one
    # neighbour, UNDER, is a different word. Inside a line whose other words
    # are real - IRONT in 'their phone at the iront desk' - the neighbour is
    # constrained by the sentence around it.
    context_ok = (sum(ok_flags) >= int(
        _need(cfg, "ocr_dictionary_correct_min_line_words")))
    out, good, corrections = [], 0, []
    for t, ok in zip(tokens, ok_flags):
        if ok:
            good += 1
            out.append(clean_number(t) if is_number(clean_number(t)) else t)
            continue
        fix = dictionary_correct(t, words, cfg) if context_ok else None
        if fix is not None:
            word, dist = fix
            good += 1
            fixed = word.upper()
            out.append(fixed)
            corrections.append({"from": t, "to": fixed, "distance": dist})
            continue
        out.append(t)
    share = good / float(len(tokens))
    need_share = float(_need(cfg, "ocr_min_real_word_share"))
    norm = None
    if corrections:
        norm = {"kind": "dictionary", "from": key, "to": " ".join(out),
                "corrections": corrections,
                "distance": max(c["distance"] for c in corrections)}
    return (share >= need_share), " ".join(out), norm


# ---------------------------------------------------------------------------
# lines and caption blocks
# ---------------------------------------------------------------------------

def group_lines(readings, frame_w, frame_h, cfg):
    """Boxes on one baseline become one line, left to right.

    Noise is dropped BEFORE grouping, not after, and V9 is why. A 0.10
    confidence misread sitting on the caption's baseline was being glued onto
    it, and because a line is only as good as its worst part, the whole
    caption then carried that 0.10 and was rejected. The misread is still
    recorded as its own row with its own reason; it just does not get to drag
    a legible caption down with it.
    """
    ytol = float(_need(cfg, "ocr_line_merge_y_tolerance")) * frame_h
    xgap = float(_need(cfg, "ocr_line_merge_x_gap")) * frame_w
    group_floor = float(_need(cfg, "ocr_group_min_confidence"))
    # Sharing a baseline is not enough. V9's 26 px LUTRONIC logo sits on the
    # same baseline, within 25 px, as the 50 px caption, and 2.0.0 glued them
    # into 'LUTRONIC POV: THE PATIENT FORGOT ...' - which then failed the
    # word test as a whole. Parts of one line have letters of one size, and
    # their centres sit within a fraction of a letter height of each other.
    ratio_max = float(_need(cfg, "ocr_block_merge_height_ratio_max"))
    dy_heights = float(_need(cfg, "ocr_line_merge_max_dy_heights"))

    loose = [r for r in readings if r["confidence"] < group_floor]
    readings = [r for r in readings if r["confidence"] >= group_floor]
    items = sorted(readings, key=lambda r: (r["m"]["cy"], r["m"]["x0"]))
    lines = []
    for r in items:
        placed = False
        for ln in lines:
            last = ln["parts"][-1]
            ha, hb = last["m"]["h"], r["m"]["h"]
            same_size = (min(ha, hb) > 0
                         and max(ha, hb) / min(ha, hb) <= ratio_max)
            dy = abs(ln["cy"] - r["m"]["cy"])
            if same_size and dy <= ytol and dy <= dy_heights * min(ha, hb):
                if r["m"]["x0"] - last["m"]["x1"] <= xgap:
                    ln["parts"].append(r)
                    ln["cy"] = sum(p["m"]["cy"] for p in ln["parts"]) / len(ln["parts"])
                    placed = True
                    break
        if not placed:
            lines.append({"cy": r["m"]["cy"], "parts": [r]})

    out = []
    for ln in lines:
        parts = sorted(ln["parts"], key=lambda p: p["m"]["x0"])
        xs0 = min(p["m"]["x0"] for p in parts)
        xs1 = max(p["m"]["x1"] for p in parts)
        ys0 = min(p["m"]["y0"] for p in parts)
        ys1 = max(p["m"]["y1"] for p in parts)
        out.append({
            "raw_text": " ".join(p["raw_text"].strip() for p in parts),
            "normalized": normalise_text(
                " ".join(p["normalized"] for p in parts)),
            "confidence": min(p["confidence"] for p in parts),
            "points": [[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]],
            "m": {"x0": xs0, "x1": xs1, "y0": ys0, "y1": ys1,
                  "cx": (xs0 + xs1) / 2.0, "cy": (ys0 + ys1) / 2.0,
                  "w": xs1 - xs0, "h": ys1 - ys0},
            "variants": sorted({v for p in parts for v in p["variants"]}),
            "parts": len(parts),
            "members": [dict(p, parts=1, members=None) for p in parts],
        })
    for r in loose:
        out.append({
            "raw_text": r["raw_text"], "normalized": r["normalized"],
            "confidence": r["confidence"], "points": r["points"],
            "m": r["m"], "variants": r["variants"], "parts": 1,
        })
    return sorted(out, key=lambda l: (l["m"]["cy"], l["m"]["x0"]))


def _join(buf):
    xs0 = min(b["m"]["x0"] for b in buf)
    xs1 = max(b["m"]["x1"] for b in buf)
    ys0 = min(b["m"]["y0"] for b in buf)
    ys1 = max(b["m"]["y1"] for b in buf)
    return {
        "raw_text": " ".join(b["raw_text"] for b in buf),
        "normalized": normalise_text(" ".join(b["normalized"] for b in buf)),
        "confidence": min(b["confidence"] for b in buf),
        "points": [[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]],
        "m": {"x0": xs0, "x1": xs1, "y0": ys0, "y1": ys1,
              "cx": (xs0 + xs1) / 2.0, "cy": (ys0 + ys1) / 2.0,
              "w": xs1 - xs0, "h": ys1 - ys0},
        "variants": sorted({v for b in buf for v in b["variants"]}),
        "parts": sum(b["parts"] for b in buf),
        "caption_block_lines": len(buf),
        "line_height": sum(b.get("line_height", b["m"]["h"]) for b in buf)
        / len(buf),
        "members": [m for b in buf for m in (b.get("members") or [b])],
    }


def group_text_blocks(lines, frame_w, frame_h, cfg):
    """Stacked lines that belong together become one block.

    2.0.0 only joined lines inside a caption band AND wider than 40% of the
    frame, which is why GLOWING / YOUNGER - a poster headline, 21% wide -
    stayed two rows. The test is now what makes lines one piece of text
    regardless of where it sits: similar letter height (within
    ocr_block_merge_height_ratio_max), the next line starting within
    ocr_block_merge_max_gap_lines line-heights of the last, and the two
    aligned on their left edge or their centre (within
    ocr_block_merge_align_tolerance of the frame width). A poster headline
    beside a caption shares neither alignment, so the two stay apart - V9's
    caption is centred at 0.50 of the frame, the headline at 0.20.

    Only confident lines are joined; a low-confidence fragment stays on its
    own row so it cannot drag a legible block below the floor."""
    ratio_max = float(_need(cfg, "ocr_block_merge_height_ratio_max"))
    gap_lines = float(_need(cfg, "ocr_block_merge_max_gap_lines"))
    align_tol = float(_need(cfg, "ocr_block_merge_align_tolerance")) * frame_w
    floor = float(_need(cfg, "ocr_group_min_confidence"))

    def lh(ln):
        return ln.get("line_height", ln["m"]["h"])

    def joins(a, b):
        if a["confidence"] < floor or b["confidence"] < floor:
            return False
        ha, hb = lh(a), lh(b)
        if min(ha, hb) <= 0 or max(ha, hb) / min(ha, hb) > ratio_max:
            return False
        if b["m"]["y0"] - a["m"]["y1"] > gap_lines * max(ha, hb):
            return False
        if b["m"]["cy"] <= a["m"]["cy"]:
            return False
        return (abs(a["m"]["x0"] - b["m"]["x0"]) <= align_tol
                or abs(a["m"]["cx"] - b["m"]["cx"]) <= align_tol)

    pending = sorted(lines, key=lambda l: (l["m"]["cy"], l["m"]["x0"]))
    blocks = []
    for ln in pending:
        for blk in blocks:
            if joins(blk[-1], ln):
                blk.append(ln)
                break
        else:
            blocks.append([ln])
    out = [b[0] if len(b) == 1 else _join(b) for b in blocks]
    return sorted(out, key=lambda l: (l["m"]["cy"], l["m"]["x0"]))


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def _core(text):
    """A reading with its leading and trailing punctuation removed."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text or "")


def mark_persistence(frames, cfg, frame_w, frame_h):
    """Tag every line with how many sampled frames it appears in nearby.

    Text in the same place across frames is the same text even when the OCR
    spells it differently each time. The 0.0 setting on IMG_0988 comes back as
    "0.0,", "0.01" and "0.0, ^" from three consecutive frames; demanding an
    identical string made each of them a one-frame reading and the setting was
    never persistent. So a track matches on POSITION plus a close-enough
    string, not on an exact one.

    `frames` is [{"index","timestamp","lines":[...]}] in sample order."""
    import difflib
    tol = float(_need(cfg, "ocr_persistence_position_tolerance"))
    sim_min = float(_need(cfg, "ocr_persistence_text_similarity"))
    diag = math.hypot(frame_w, frame_h)
    tracks = []
    for fr in frames:
        for ln in fr["lines"]:
            hit = None
            for tr in tracks:
                # Compare the meaningful core. Trailing marks are exactly what
                # differs between frames - "0.01", "0.0, ^", "0.0," - and
                # letting them count drops the similarity below any sensible
                # threshold for strings this short.
                a = _core(tr["normalized"])
                b = _core(ln["normalized"])
                if a != b and difflib.SequenceMatcher(
                        None, a, b).ratio() < sim_min:
                    continue
                if math.hypot(tr["cx"] - ln["m"]["cx"],
                              tr["cy"] - ln["m"]["cy"]) <= tol * diag:
                    hit = tr
                    break
            if hit is None:
                tracks.append({"normalized": ln["normalized"],
                               "cx": ln["m"]["cx"], "cy": ln["m"]["cy"],
                               "frames": [fr["index"]], "lines": [ln]})
            else:
                if fr["index"] not in hit["frames"]:
                    hit["frames"].append(fr["index"])
                hit["lines"].append(ln)
                n = len(hit["lines"])
                hit["cx"] += (ln["m"]["cx"] - hit["cx"]) / n
                hit["cy"] += (ln["m"]["cy"] - hit["cy"]) / n
    for tr in tracks:
        seen = len(set(tr["frames"]))
        for ln in tr["lines"]:
            ln["frames_seen"] = seen
            ln["first_frame"] = min(tr["frames"])
    return tracks


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def split_failed_merges(lines, terms, words, cfg):
    """A merged line that fails the word test as a whole is judged part by
    part instead.

    Grouping is right about layout and can still cost a reading: V9's leaflet
    prints NOURKRI(N) above a line of 12 px small print, the two join into
    one block, and 'NOURKRI PRT PRO FOR' fails as a whole - losing the one
    legible term to the unreadable print under it. So when the merge fails,
    its parts go forward as separate readings and each is judged on its own.
    A merge that passes is kept whole: a caption stays one caption.

    3.1.0: two more merges are split. Before 3.1.0 they happened to fail on
    a token the new tokeniser now reads ('5', 'INDER'), so they were split
    by accident; now they are split on purpose:
      - a merge the logo guard rejects (C1 at 14 s: the lockup read as
        'KLINIK DR INDER NIK DR INDER' over 'REDEFINING BEAUTY & HEALTH' -
        split, KLINIK DR INDER and the tagline stand, NIK DR INDER does not);
      - a same-baseline merge whose pieces are each real text on their own
        and sit more than ocr_line_split_min_gap_heights letter heights
        apart - two labels side by side, not one line (C1 at 89 s: 'After
        treatment' and 'After 5 days' under two different photos)."""
    out = []
    for ln in lines:
        members = ln.get("members") or []
        if len(members) > 1:
            says, norm, _x = evaluate_text(ln["normalized"], terms, words, cfg)
            split = (not says) or logo_fragment(norm, terms, cfg) is not None
            if not split and not ln.get("caption_block_lines") \
                    and _wide_gap(members, cfg):
                split = all(evaluate_text(m["normalized"], terms, words, cfg)[0]
                            for m in members)
            if split:
                out.extend(dict(m, members=None) for m in members)
                continue
        out.append(ln)
    return out


def _wide_gap(members, cfg):
    """Do side-by-side pieces of one line sit further apart than
    ocr_line_split_min_gap_heights letter heights anywhere? 0 disables."""
    limit = float(_need(cfg, "ocr_line_split_min_gap_heights"))
    if limit <= 0:
        return False
    ms = sorted(members, key=lambda m: m["m"]["x0"])
    for a, b in zip(ms, ms[1:]):
        h = min(a["m"]["h"], b["m"]["h"])
        if h > 0 and (b["m"]["x0"] - a["m"]["x1"]) / h > limit:
            return True
    return False


def without_normalisation(cfg):
    """The same config with every substitution switched off - what the
    engine really read. Exact matches to a known term still count; only
    fuzzy known-term matches and dictionary corrections are disabled."""
    c = dict(cfg)
    c["ocr_normalize_to_known_terms"] = False
    c["ocr_dictionary_correction_enabled"] = False
    return c


def build_observations(per_frame_variants, frame_meta, cfg, lexicon,
                       usable=True, treatments=None, aliases=None):
    """Rows for ocr_observations, plus a per-asset summary.

    per_frame_variants: [{"index","timestamp","variants":{name: [reading,...]}}]
    frame_meta: {"width","height"}
    usable: False when the asset is UNUSABLE_FOOTAGE, which forces every
            reading to accepted_for_search = False.

    Pass without_normalisation(cfg) as cfg to see what the engine read with
    no substitution at all.
    """
    fw = float(frame_meta["width"])
    fh = float(frame_meta["height"])
    diag = math.hypot(fw, fh)
    floor = float(_need(cfg, "ocr_confidence_floor"))
    min_frames = int(_need(cfg, "ocr_min_consecutive_frames"))
    ctx_dist = float(_need(cfg, "ocr_device_context_max_distance")) * diag
    need_ctx = bool(_need(cfg, "ocr_bare_number_requires_device_context"))
    terms = known_terms(cfg, treatments, aliases)
    words = load_words(cfg)
    normalisations = []

    frames = []
    for fr in per_frame_variants:
        pooled = pool_variants(fr["variants"], fw, fh, cfg)
        lines = group_lines(pooled, fw, fh, cfg)
        lines = group_text_blocks(lines, fw, fh, cfg)
        frames.append({"index": fr["index"], "timestamp": fr["timestamp"],
                       "scene_id": fr.get("scene_id"),
                       "lines": split_failed_merges(lines, terms, words, cfg)})

    # Persistence needs the text to be sampled twice, which a shot shorter
    # than the sampling interval cannot give: V17's closing card (41.27-
    # 42.66s) holds CALL NOW for one keyframe. With
    # ocr_persistence_capped_by_scene_samples the requirement for a line is
    # at most the number of sampled frames in its own scene.
    cap_by_scene = bool(_need(cfg, "ocr_persistence_capped_by_scene_samples"))
    scene_samples = {}
    for fr in frames:
        if fr["scene_id"] is not None:
            scene_samples[fr["scene_id"]] = scene_samples.get(fr["scene_id"], 0) + 1

    mark_persistence(frames, cfg, fw, fh)

    import step2_analyse as s2
    rows = []
    for fr in frames:
        judged = []
        for ln in fr["lines"]:
            says, norm, normalisation = evaluate_text(
                ln["normalized"], terms, words, cfg)
            judged.append((ln, says, norm, normalisation))
        # A setting is only meaningful beside the device showing it: the
        # positions of every confident known-term line in this frame.
        anchors = [ln["m"] for ln, says, norm, _n in judged
                   if says and ln["confidence"] >= floor
                   and not numeric_line(norm, cfg)
                   and (normalise_text(norm) in terms or any(
                       t in normalise_text(norm).split() for t in terms
                       if " " not in t))]

        need_frames = min_frames
        if cap_by_scene and fr["scene_id"] is not None:
            need_frames = max(1, min(min_frames, scene_samples[fr["scene_id"]]))
        for ln, says, norm, normalisation in judged:
            seen = ln.get("frames_seen", 1)
            persistent = seen >= need_frames
            conf_ok = ln["confidence"] >= floor
            if normalisation:
                normalisation = dict(normalisation,
                                     timestamp=fr["timestamp"],
                                     confidence=round(ln["confidence"], 4))
                normalisations.append(normalisation)
            is_numeric = says and numeric_line(norm, cfg)
            has_ctx = any(math.hypot(m["cx"] - ln["m"]["cx"],
                                     m["cy"] - ln["m"]["cy"]) <= ctx_dist
                          for m in anchors)
            # Persistence assumes the text sits still long enough to be
            # sampled twice. V9's overlay sequence gives each treatment name
            # about 1.2 seconds against one keyframe a second, so ULTRAFORMER,
            # PLINEST, DOCTOR LEW and KLINIK DR INDER each appear in exactly
            # ONE frame. A confident match to a curated clinic term is
            # corroborated by the vocabulary instead of by repetition. Noise
            # is unaffected: CAXSSS is not a known term at any confidence.
            corroborated = False
            if not persistent and conf_ok and says and usable:
                hit = match_known_term(norm, terms, cfg)
                corroborated = bool(
                    _need(cfg, "ocr_known_term_bypasses_persistence")
                    and hit is not None
                    and ln["confidence"] >= float(
                        _need(cfg, "ocr_known_term_confidence_min")))
            # A subtitle can be on screen for less than two samples (V17's
            # 'Pertama' at 17s). In a caption band
            # (ocr_single_frame_caption_bands), a line made only of
            # real words, not corrected, read once at very high confidence
            # stands on its own. 0 disables.
            single_caption = False
            cap_conf = float(_need(cfg, "ocr_single_frame_caption_min_confidence"))
            if (not persistent and not corroborated and cap_conf > 0 and conf_ok
                    and says and usable and not is_numeric
                    and normalisation is None
                    and ln["confidence"] >= cap_conf):
                bands = _need(cfg, "ocr_single_frame_caption_bands")
                cy = ln["m"]["cy"]
                single_caption = (
                    ("bottom" in bands and cy >= float(_need(
                        cfg, "ocr_caption_bottom_band_fraction")) * fh)
                    or ("top" in bands and cy <= float(_need(
                        cfg, "ocr_caption_top_band_fraction")) * fh))
            ctx_ok = (not is_numeric) or has_ctx or not need_ctx
            fragment = (logo_fragment(norm, terms, cfg)
                        if says and not is_numeric else None)
            accepted = bool((persistent or corroborated or single_caption)
                            and conf_ok and says and ctx_ok and usable
                            and fragment is None)
            reason = None
            if not conf_ok:
                reason = "confidence {:.4f} below ocr_confidence_floor {}".format(
                    ln["confidence"], floor)
            elif not says:
                reason = ("NOT_MEANINGFUL: {!r} is not a known term and not "
                          "made entirely of en/ms words and numbers"
                          .format(norm[:48]))
            elif fragment is not None:
                reason = "{}: {}".format(*fragment)
            elif not ctx_ok:
                reason = ("BARE_NUMBER_NO_CONTEXT: {!r} has no device label "
                          "or screen term within ocr_device_context_max_"
                          "distance in this frame".format(norm[:24]))
            elif not persistent and not corroborated and not single_caption:
                reason = ("NOT_PERSISTENT: read in {} sampled frame(s), "
                          "ocr_min_consecutive_frames={}{}".format(
                              seen, min_frames,
                              "" if need_frames == min_frames else
                              " (capped to {} by its scene)".format(need_frames)))
            elif not usable:
                reason = ("asset flagged UNUSABLE_FOOTAGE; on-screen text is "
                          "recorded but not accepted as search text")

            rows.append({
                "raw_text": ln["raw_text"],
                "normalized_text": norm,
                "text_type": s2.classify_text(
                    ln["points"], fw, fh, normalized_text=norm, cfg=cfg,
                    lexicon={k: set(v) for k, v in (lexicon or {}).items()}),
                "confidence": round(min(1.0, max(0.0, ln["confidence"])), 4),
                "accepted_for_search": accepted,
                "rejection_reason": reason,
                "bounding_box": {"points": [[float(p[0]), float(p[1])]
                                            for p in ln["points"]]},
                "timestamp": fr["timestamp"],
                "_frame_index": fr["index"],
                "frames_seen": seen,
                "variants": ln["variants"],
                "caption_block_lines": ln.get("caption_block_lines"),
                "normalisation": normalisation,
            })

    summary = {
        "frames_ocrd": [f["index"] for f in frames],
        "frames_ocrd_count": len(frames),
        "lines": len(rows),
        "accepted": sum(1 for r in rows if r["accepted_for_search"]),
        "rejected_by": _reason_counts(rows),
        "normalisations": normalisations,
    }
    return rows, summary


def _reason_counts(rows):
    out = {}
    for r in rows:
        if r["rejection_reason"]:
            k = r["rejection_reason"].split(":")[0].split(" ")[0]
            out[k] = out.get(k, 0) + 1
    return out


def coverage_frames(times, cfg, change_scores=None):
    """Which keyframes to read. Replaces the flat 24-frame cap.

    Up to ocr_full_coverage_max_seconds of video, every keyframe is read.
    Beyond that, every ocr_long_video_stride_seconds-th keyframe, plus any
    keyframe whose picture changed by more than ocr_change_frame_threshold
    (mean absolute grey-level difference, 0-1) from the keyframe before it -
    new text arriving is a change of picture. change_scores[i] is that
    difference for keyframe i.

    The stride can never exceed 2 s (config max), so on a 1 fps grid anything
    on screen for 2 seconds or more lands on at least one read frame.
    Returns sorted indices into `times`."""
    if not times:
        return []
    duration = max(times) - min(times)
    if duration <= float(_need(cfg, "ocr_full_coverage_max_seconds")):
        return list(range(len(times)))
    stride = float(_need(cfg, "ocr_long_video_stride_seconds"))
    threshold = float(_need(cfg, "ocr_change_frame_threshold"))
    picked, last_t = [], None
    for i, t in enumerate(times):
        if last_t is None or t - last_t >= stride - 1e-9:
            picked.append(i)
            last_t = t
        elif change_scores and change_scores[i] is not None \
                and change_scores[i] > threshold:
            picked.append(i)
            last_t = t
    return picked


def choose_frames(frames, quality, cfg, cap):
    """Which frames to read when not all of them can be.

    A flat spread is the wrong sample: a device screen that is on for four
    seconds of a three-minute video is invisible to it. Frames are ranked by
    sharpness, because text that is legible to OCR is text that is in focus,
    and the spread is kept by taking the best frame from each of `cap` equal
    slices rather than the best `cap` overall."""
    if cap <= 0 or len(frames) <= cap:
        return list(range(len(frames)))
    if not bool(_need(cfg, "ocr_sampled_prefer_quality")) or not quality:
        step = len(frames) / float(cap)
        return sorted({min(len(frames) - 1, int(i * step)) for i in range(cap)})
    picked = []
    n = len(frames)
    for k in range(cap):
        lo = int(round(k * n / float(cap)))
        hi = int(round((k + 1) * n / float(cap)))
        hi = max(hi, lo + 1)
        window = range(lo, min(hi, n))
        best = max(window, key=lambda i: (
            quality[i].get("sharpness", 0.0) if i < len(quality) else 0.0, -i))
        picked.append(best)
    return sorted(set(picked))


# ---------------------------------------------------------------------------
# the config audit
# ---------------------------------------------------------------------------

# Only a key actually handed to a config accessor counts as read. Matching
# every "ocr_..." literal would count metadata field names and result keys as
# configuration, which is how an audit turns into noise nobody reads.
OCR_KEY_RES = (
    re.compile(r"""(?:_need|need|_cfg_or)\(\s*cfg\s*,\s*["'](ocr_[a-z0-9_]+)["']"""),
    re.compile(r"""cfg\.get\(\s*["'](ocr_[a-z0-9_]+)["']"""),
    re.compile(r"""cfg\[\s*["'](ocr_[a-z0-9_]+)["']\s*\]"""),
)

# Keys that are read by other modules on the OCR path, or that describe the
# stage rather than steering it. Listed explicitly so the audit stays honest:
# adding a key here is a deliberate statement that it is used elsewhere.
OCR_KEYS_ELSEWHERE = {
    "ocr_enabled", "ocr_engine", "ocr_languages", "ocr_torch_threads",
    "ocr_sampled_max_frames", "ocr_stage_base_seconds",
    "ocr_stage_seconds_per_frame",
}


def audit_config(cfg, extra_sources=()):
    """Fail loudly when config and code have drifted apart.

    This exists because three separate stages of this pipeline shipped with
    their safeguards configured and never read - the scene cap, the whole
    transcription rule set, and nine of the twenty-one ocr_* keys. A key that
    nothing reads is not a setting, it is a note, and an operator who changes
    it and sees nothing happen has been misled. So: every ocr_* key in
    pipeline_config must be read by the code, and every ocr_* key the code
    reads must exist in pipeline_config.
    """
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent
    sources = [here / "step2_ocr.py", here / "step2_analyse.py",
               here / "ocr_capture.py", here / "step2_ocr_run.py",
               here / "ocr_engine.py", here / "score_ocr.py"]
    sources.extend(_P(s) for s in extra_sources)
    referenced = set()
    for src in sources:
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        for rx in OCR_KEY_RES:
            referenced |= set(rx.findall(text))

    configured = {k for k in cfg if k.startswith("ocr_")}
    unused = sorted(configured - referenced - OCR_KEYS_ELSEWHERE)
    missing = sorted(referenced - configured)
    problems = []
    if unused:
        problems.append(
            "these ocr_* keys are in pipeline_config but nothing reads them: "
            "{}".format(", ".join(unused)))
    if missing:
        problems.append(
            "the code reads these ocr_* keys but pipeline_config has no such "
            "key: {}".format(", ".join(missing)))
    if problems:
        raise SystemExit("OCR CONFIG AUDIT FAILED\n  " + "\n  ".join(problems))
    return sorted(configured)
