#!/usr/bin/env python3
"""
Technical quality of a stored keyframe, and the stretches that are unusable.

Classical image measurement only - variance of the Laplacian, mean luma, luma
standard deviation, Canny edge density, HSV saturation, and ORB/RANSAC global
motion between consecutive frames. No model looks at these pictures and
nothing here describes their content; the question is only whether a frame is
worth looking at, not what is in it.

WHY THE OBVIOUS VERSION DOES NOT WORK

Sharpness alone cannot do this job, and IMG_6383 is the proof. Eighty-four
seconds of tattoo removal, perfectly exposed and perfectly legible, scores a
Laplacian variance of 15-70 because most of the frame is smooth skin. The two
genuinely unusable frames in IMG_3413 - the camera pressed against clothing -
score 4 and 22. Any sharpness threshold that catches the second condemns the
first, and IMG_6383 is 84 seconds of usable footage.

The same is true of flatness: measured by the share of 16x16 blocks with
almost no local variation, IMG_6383 at 83s (0.754) is FLATTER than IMG_3413 at
3s (0.742). Neither sharpness nor flatness separates them.

What separates them is that IMG_3413's bad frames have nothing to look at in
EITHER sense - no structure and no colour. Across all 537 stored keyframes,
"saturation below 40 and edge density below 0.015" selects exactly those two
frames and nothing else. IMG_0988 is desaturated but richly detailed; IMG_6383
is soft but strongly coloured; each is spared by the half of the test it
passes. So the content term asks for structure OR colour, and only a frame
with neither is treated as pointing away from the subject.

Camera shake is deliberately given the least weight, because this is handheld
clinic footage and the brief is explicit that normal wander must not be
flagged. IMG_1147 swings a long way at 33s and 140s without being unusable for
a moment; a single large reposition is a camera being aimed, not a fault.
"""
from __future__ import annotations

import math


def _need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: " + key)
    return cfg[key]


REASON_BLUR = "blur"
REASON_DARK = "dark"
REASON_SHAKE = "shake"
REASON_POINTING_AWAY = "pointing_away"

# which subscore, when it is the worst one, names the fault
REASON_OF = {
    "sharpness": REASON_BLUR,
    "brightness": REASON_DARK,
    "contrast": REASON_DARK,
    "content": REASON_POINTING_AWAY,
    "steadiness": REASON_SHAKE,
}
# a declared order so that a tie never depends on dict iteration
REASON_ORDER = ["content", "sharpness", "brightness", "contrast", "steadiness"]


def _ramp(value, floor, target):
    """0 at or below floor, 1 at or above target, linear between."""
    if target <= floor:
        return 1.0 if value >= target else 0.0
    return max(0.0, min(1.0, (float(value) - floor) / (target - floor)))


def _log_ramp(value, floor, target):
    """The same, on a log scale - the right shape for a variance that spans
    two orders of magnitude across ordinary footage."""
    v = max(float(value), 1e-6)
    if target <= floor:
        return 1.0 if v >= target else 0.0
    lo, hi = math.log(max(floor, 1e-6)), math.log(target)
    return max(0.0, min(1.0, (math.log(v) - lo) / (hi - lo)))


def measure_frame(image, cfg, prev_grey=None, matcher=None):
    """The raw numbers for one frame, plus the grey plane for the next call."""
    import cv2
    import numpy as np

    px = int(_need(cfg, "quality_frame_downscale_px"))
    h, w = image.shape[:2]
    s = px / float(max(h, w))
    small = cv2.resize(image, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    m = {
        "laplacian_variance": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 3),
        "mean_luma": round(float(grey.mean()), 3),
        "luma_std": round(float(grey.std()), 3),
        "edge_density": round(float((cv2.Canny(grey, 50, 150) > 0).mean()), 5),
        "saturation": round(float(hsv[:, :, 1].mean()), 3),
        "motion_fraction": None,
    }
    if prev_grey is not None and matcher is not None:
        m["motion_fraction"] = global_motion(prev_grey, grey, cfg, matcher)
    return m, grey


def global_motion(ga, gb, cfg, matcher):
    """How far the camera moved between two frames, as a fraction of width.

    A similarity fit through RANSAC, so a hand moving inside the frame does
    not read as the camera moving.
    """
    import cv2
    import numpy as np
    orb = matcher["orb"]
    bf = matcher["bf"]
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None
    good = [m[0] for m in bf.knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 8:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000)
    if M is None or inl is None or int(inl.sum()) < 8:
        return None
    return round(float(math.hypot(M[0, 2], M[1, 2])) / ga.shape[1], 5)


def subscores(m, cfg):
    """Each aspect of the frame, on 0-1, higher being better."""
    sharp = _log_ramp(m["laplacian_variance"],
                      float(_need(cfg, "quality_sharpness_floor_variance")),
                      float(_need(cfg, "quality_sharpness_target_variance")))

    lo = float(_need(cfg, "quality_brightness_min_luma"))
    hi = float(_need(cfg, "quality_brightness_max_luma"))
    ramp = float(_need(cfg, "quality_brightness_ramp_luma"))
    luma = m["mean_luma"]
    if luma < lo:
        bright = _ramp(luma, lo - ramp, lo)
    elif luma > hi:
        bright = 1.0 - _ramp(luma, hi, hi + ramp)
    else:
        bright = 1.0

    contrast = _ramp(m["luma_std"],
                     float(_need(cfg, "quality_contrast_floor_std")),
                     float(_need(cfg, "quality_contrast_target_std")))

    # structure OR colour - see the module docstring for why it is an OR
    edge = _ramp(m["edge_density"],
                 float(_need(cfg, "quality_content_edge_floor")),
                 float(_need(cfg, "quality_content_edge_target")))
    colour = _ramp(m["saturation"],
                   float(_need(cfg, "quality_content_saturation_floor")),
                   float(_need(cfg, "quality_content_saturation_target")))
    content = max(edge, colour)

    mf = m.get("motion_fraction")
    if mf is None:
        steadiness = 1.0
    else:
        tol = float(_need(cfg, "quality_motion_tolerance_fraction"))
        sev = float(_need(cfg, "quality_motion_severe_fraction"))
        steadiness = 1.0 - _ramp(mf, tol, sev)

    return {"sharpness": sharp, "brightness": bright, "contrast": contrast,
            "content": content, "steadiness": steadiness,
            "_edge": edge, "_colour": colour}


def score_frame(m, cfg):
    """(score, subscores, reason) for one frame.

    A weighted GEOMETRIC mean, not an average. An average lets a frame with
    nothing in it pass on the strength of being well exposed - IMG_3413 at 2s
    is evenly lit and perfectly contrasty grey fabric, and would score 0.37 on
    an arithmetic mean. A geometric mean says that a frame has to be adequate
    in every respect at once, which is what "usable" means.
    """
    sub = subscores(m, cfg)
    floor = float(_need(cfg, "quality_subscore_floor"))
    weights = {
        "sharpness": float(_need(cfg, "quality_weight_sharpness")),
        "brightness": float(_need(cfg, "quality_weight_brightness")),
        "contrast": float(_need(cfg, "quality_weight_contrast")),
        "content": float(_need(cfg, "quality_weight_content")),
        "steadiness": float(_need(cfg, "quality_weight_steadiness")),
    }
    total = sum(weights.values()) or 1.0
    acc = 0.0
    for k, w in weights.items():
        acc += (w / total) * math.log(max(floor, min(1.0, sub[k])))
    score = round(math.exp(acc), 4)

    # A reason is only meaningful for a frame that failed. Naming the weakest
    # aspect of a frame that scored 1.000 would read as a fault report on
    # footage nobody has any complaint about.
    if score >= float(_need(cfg, "quality_usable_score_min")):
        return score, sub, None
    worst = min(REASON_ORDER, key=lambda k: (sub[k], REASON_ORDER.index(k)))
    return score, sub, REASON_OF[worst]


def find_spans(frames, duration, cfg):
    """Runs of consecutive frames below the usable score, at least as long as
    quality_unusable_min_seconds.

    The keyframes are one per second, so a span covers from the first bad
    frame to the next good one, or to the end of the video. A single bad
    second is not a span: a hand passing the lens is not unusable footage.
    """
    thr = float(_need(cfg, "quality_usable_score_min"))
    min_len = float(_need(cfg, "quality_unusable_min_seconds"))
    spans, run = [], []
    for f in frames:
        if f["score"] < thr:
            run.append(f)
            continue
        if run:
            spans.append(run)
            run = []
    if run:
        spans.append(run)

    out = []
    times = [f["t"] for f in frames]
    for run in spans:
        start = run[0]["t"]
        after = [t for t in times if t > run[-1]["t"]]
        end = after[0] if after else round(float(duration), 3)
        if end - start < min_len - 1e-9:
            continue
        counts = {}
        for f in run:
            counts[f["reason"]] = counts.get(f["reason"], 0) + 1
        reason = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out.append({
            "start_seconds": round(float(start), 3),
            "end_seconds": round(float(min(end, duration)), 3),
            "reason": reason,
            "frame_count": len(run),
            "min_score": round(min(f["score"] for f in run), 4),
            "mean_score": round(sum(f["score"] for f in run) / len(run), 4),
            "reasons_seen": counts,
        })
    return out


def unusable_share(spans, duration):
    if not duration:
        return 0.0
    covered = sum(s["end_seconds"] - s["start_seconds"] for s in spans)
    return round(min(1.0, covered / float(duration)), 4)
