"""
Step 2 scene segmentation, v2 - one detector per kind of footage.

Deterministic measurement only: ffmpeg decodes, numpy compares. No AI model.

Why two paths
-------------
Step 1 labels every video RAW (straight off the camera) or EDITED (exported by
an editing tool) in asset_technical_metadata.edit_class. The two kinds of
footage break a single detector in opposite directions:

  RAW    One continuous take. There are no edits, so a sudden change is the
         camera moving (a whip pan, a hand over the lens), not a cut. What can
         change is the SITUATION, slowly. The old detector split IMG_1147 into
         14 scenes this way. The RAW path ignores frame-to-frame jumps and only
         looks for a sustained change of what is in front of the camera, then
         merges neighbouring scenes that look alike.

  EDITED An editor joined shots. Joins are hard cuts, punch-ins, or animated
         transitions (dissolve, wipe, whip-blur, light leak) that spread the
         change over several frames, so a frame-to-frame score misses them:
         at scene > 0.4 the old detector found 2 of V16's 11 real cuts. Edited
         shots are also short: V17 opens with a 0.97 s shot, so a 2 s minimum
         scene length erases real scenes. The EDITED path looks for both
         instant and spread-out changes, then checks that the picture before
         and after really is different - which is what throws out flashes,
         caption changes and effects that return to the same shot.

NEEDS_REVIEW and unknown classes use the EDITED path (scene_unclassified_mode).

The frame signature
-------------------
Every decoded frame is reduced to one signature, the metric named in
pipeline_config.scene_drift_metric = "hsv_centre_weighted+structural_ncc":

  - an HSV histogram (scene_drift_histogram_bins per channel), weighted so the
    centre (scene_centre_crop_fraction of each axis) counts fully and the edges
    less - captions and logos usually sit at the edges;
  - a small greyscale thumbnail (scene_structural_px square), compared by
    normalised cross-correlation, which sees layout that a histogram cannot.

distance = w * (1 - ncc) / 2 + (1 - w) * histogram_tvd, w = scene_structural_weight,
always in [0, 1].

Every tunable is read from pipeline_config through need(); nothing numeric is
decided here.
"""
from __future__ import annotations

import re
import subprocess
import threading

import numpy as np

MODULE_VERSION = "step2_scenes_v2/1.5.0"
SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*([0-9]+\.?[0-9]*)")


class SceneError(Exception):
    """Segmentation could not run. The caller turns this into AssetFailed."""


def _need(cfg, key):
    if key not in cfg or cfg[key] is None:
        raise SceneError("pipeline_config is missing {!r}".format(key))
    return cfg[key]


# --------------------------------------------------------------------------
# decode + signatures
# --------------------------------------------------------------------------

def _centre_weights(px, crop_fraction):
    """1.0 inside the centre box, falling linearly to 0.25 at the frame edge."""
    ax = np.abs(np.linspace(-1.0, 1.0, px))
    half = crop_fraction
    w1 = np.where(ax <= half, 1.0,
                  1.0 - 0.75 * (ax - half) / max(1e-6, 1.0 - half))
    return np.outer(w1, w1).astype(np.float32)


def _rgb_to_hsv(rgb):
    """rgb uint8 (..., 3) -> h, s, v in [0, 1)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    nz = diff > 1e-6
    rc = np.where(nz, (mx - r) / np.where(nz, diff, 1), 0)
    gc = np.where(nz, (mx - g) / np.where(nz, diff, 1), 0)
    bc = np.where(nz, (mx - b) / np.where(nz, diff, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(nz, (h / 6.0) % 1.0, 0.0)
    s = np.where(mx > 1e-6, diff / np.where(mx > 1e-6, mx, 1), 0.0)
    return h, s, mx


class Signatures:
    """Per-frame signatures for one video, every decoded frame."""

    def __init__(self, times, hists, structs, spread, luma):
        self.t = np.asarray(times, dtype=np.float64)
        self.h = np.asarray(hists, dtype=np.float32)      # (n, 3*bins), each third sums to 1
        self.g = np.asarray(structs, dtype=np.float32)    # (n, px*px), zero-mean unit-norm (or zero)
        self.spread = np.asarray(spread, dtype=np.float32)  # grey std per frame, 0-1
        self.luma = np.asarray(luma, dtype=np.float32)      # grey mean per frame, 0-1
        self.n = len(self.t)

    def fps(self):
        if self.n < 2:
            return 30.0
        d = np.diff(self.t)
        d = d[d > 0]
        return float(1.0 / np.median(d)) if len(d) else 30.0


def overlay_mask(px, cfg):
    """True where a pixel of the px-square signature frame counts, False over
    the burned-in subtitle band and the logo corner.

    Edited clips carry captions and a logo that change on their own schedule;
    a caption change on an unchanged shot (0:41.7 in the FractionalCO2Laser
    clip) is otherwise scored as a cut. Fractions are of frame height and
    width, so they hold for the squashed square the signature is taken on."""
    keep = np.ones((px, px), dtype=bool)
    bottom = float(_need(cfg, "scene_edited_mask_bottom_fraction"))
    if bottom > 0:
        keep[px - int(round(bottom * px)):, :] = False
    corner = str(_need(cfg, "scene_edited_mask_logo_corner")).lower()
    cw = int(round(float(_need(cfg, "scene_edited_mask_logo_width_fraction")) * px))
    ch = int(round(float(_need(cfg, "scene_edited_mask_logo_height_fraction")) * px))
    if corner != "none" and cw > 0 and ch > 0:
        rows = slice(0, ch) if corner.startswith("top") else slice(px - ch, px)
        cols = slice(px - cw, px) if corner.endswith("right") else slice(0, cw)
        if corner not in ("top_right", "top_left", "bottom_right", "bottom_left"):
            raise SceneError("scene_edited_mask_logo_corner must be top_right, "
                             "top_left, bottom_right, bottom_left or none")
        keep[rows, cols] = False
    if not keep.any():
        raise SceneError("scene_edited_mask_* leaves nothing of the frame")
    return keep


def compute_signatures(video_path, cfg, timeout=1800, mask=False):
    """Decode every frame once at a small fixed size and reduce each to a
    signature. Frame times come from ffmpeg's own showinfo pts_time, read in a
    thread while stdout carries the pixels, so variable frame rate footage is
    timed correctly rather than assumed from a nominal fps.

    mask=True (EDITED path) leaves the subtitle band and logo corner out of
    both the histogram and the structural thumbnail; blank detection still
    looks at the whole frame."""
    px = int(_need(cfg, "scene_signature_px"))
    bins = int(_need(cfg, "scene_drift_histogram_bins"))
    spx = int(_need(cfg, "scene_structural_px"))
    crop = float(_need(cfg, "scene_centre_crop_fraction"))
    if px % spx:
        raise SceneError("scene_signature_px={} must be a multiple of "
                         "scene_structural_px={}".format(px, spx))
    weights = _centre_weights(px, crop)
    block = px // spx
    keep_blocks = None
    if mask:
        keep = overlay_mask(px, cfg)
        weights = weights * keep
        # A thumbnail cell counts only if none of it lies under an overlay.
        keep_blocks = keep.reshape(spx, block, spx, block).all(axis=(1, 3)).reshape(-1)
        if not keep_blocks.any():
            raise SceneError("scene_edited_mask_* leaves no structural cell")
    weights = weights.reshape(-1)
    weights = weights / weights.sum()

    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
           "-i", str(video_path), "-an",
           "-vf", "scale={0}:{0}:flags=area,showinfo".format(px),
           "-fps_mode", "passthrough",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts, err_tail = [], []

    def read_err():
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace")
            if "showinfo" in line and "pts_time" in line:
                m = SHOWINFO_PTS_RE.search(line)
                if m:
                    pts.append(float(m.group(1)))
            else:
                err_tail.append(line)
                del err_tail[:-20]

    th = threading.Thread(target=read_err, daemon=True)
    th.start()

    frame_bytes = px * px * 3
    hists, structs, spread, luma = [], [], [], []
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(px, px, 3)
            h, s, v = _rgb_to_hsv(rgb)
            parts = []
            for ch in (h, s, v):
                idx = np.minimum((ch.reshape(-1) * bins).astype(np.int32), bins - 1)
                parts.append(np.bincount(idx, weights=weights, minlength=bins))
            hists.append(np.concatenate(parts).astype(np.float32))
            grey = rgb.astype(np.float32).mean(axis=2) / 255.0
            spread.append(float(grey.std()))
            luma.append(float(grey.mean()))
            small = grey.reshape(spx, block, spx, block).mean(axis=(1, 3)).reshape(-1)
            if keep_blocks is not None:
                small = small[keep_blocks]
            small = small - small.mean()
            norm = float(np.linalg.norm(small))
            structs.append(small / norm if norm > 1e-4 else np.zeros_like(small))
        proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
    th.join(timeout=10)
    if proc.returncode not in (0, None) or not hists:
        raise SceneError("signature decode failed (rc={}): {}".format(
            proc.returncode, "".join(err_tail)[-300:]))
    n = len(hists)
    if len(pts) != n:
        # Fall back to evenly spaced times only if showinfo disagreed; record it.
        raise SceneError("decoded {} frames but read {} timestamps".format(n, len(pts)))
    return Signatures(pts, hists, structs, spread, luma)


def distance(sig, i, j, w):
    """Frame distance in [0, 1]."""
    tvd = float(np.abs(sig.h[i] - sig.h[j]).sum() / 6.0)
    ncc = float(np.dot(sig.g[i], sig.g[j]))
    return w * (1.0 - ncc) / 2.0 + (1.0 - w) * tvd


def _window(sig, a, b):
    """Frame indices with a <= t <= b."""
    lo = int(np.searchsorted(sig.t, a, side="left"))
    hi = int(np.searchsorted(sig.t, b, side="right"))
    return list(range(max(0, lo), min(sig.n, hi)))


def window_distance(sig, wa, wb, w):
    """Mean distance over every pair from two frame windows."""
    if not wa or not wb:
        return 0.0
    tot = 0.0
    for i in wa:
        for j in wb:
            tot += distance(sig, i, j, w)
    return tot / (len(wa) * len(wb))


def _thin(idx, step):
    return idx[::max(1, step)]


# --------------------------------------------------------------------------
# EDITED path
# --------------------------------------------------------------------------

def heal_flashes(sig, cfg, w):
    """Remove short flashes before anything is scored.

    A flash (a white frame, a glitch) jumps away from the picture and comes
    straight back to it within a few frames. Left in, its two spikes look like
    two cuts, and its echo in the transition-window score chains it into the
    same cluster as a real cut nearby - V16's two-frame white flash at 18.24s
    captured the whip-cut at 18.78s, moved the boundary onto the flash and
    then failed confirmation. Each flash frame is replaced by the frame just
    before it, so the picture reads as never having left. Returns the healed
    signatures and [(time, reason)] for the report."""
    max_s = float(_need(cfg, "scene_edited_flash_max_seconds"))
    jump = float(_need(cfg, "scene_edited_flash_min_distance"))
    back = float(_need(cfg, "scene_edited_flash_return_max_distance"))
    if max_s <= 0 or sig.n < 3:
        return sig, []
    h, g = sig.h.copy(), sig.g.copy()
    spread, luma = sig.spread.copy(), sig.luma.copy()
    healed = []
    i = 1
    while i < sig.n:
        if distance(sig, i, i - 1, w) >= jump:
            j = i + 1
            while j < sig.n and sig.t[j] - sig.t[i] <= max_s + 1e-6:
                if distance(sig, j, i - 1, w) <= back:
                    break
                j += 1
            else:
                j = None
            if j is not None and j < sig.n:
                for f in range(i, j):
                    h[f], g[f] = h[i - 1], g[i - 1]
                    spread[f], luma[f] = spread[i - 1], luma[i - 1]
                healed.append((round(float(sig.t[i]), 3),
                               "flash: {} frame(s) away (distance {:.3f}) and back "
                               "to the same picture (distance {:.3f}); healed, "
                               "not a scene change".format(
                                   j - i, distance(sig, i, i - 1, w),
                                   distance(sig, j, i - 1, w))))
                i = j + 1
                continue
        i += 1
    return Signatures(sig.t, h, g, spread, luma), healed


def segment_edited(sig, duration, cfg):
    w = float(_need(cfg, "scene_structural_weight"))
    sig, healed = heal_flashes(sig, cfg, w)
    cut_thr = float(_need(cfg, "scene_edited_cut_threshold"))
    grad_thr = float(_need(cfg, "scene_edited_gradual_threshold"))
    win = float(_need(cfg, "scene_edited_transition_window_seconds"))
    confirm = float(_need(cfg, "scene_edited_confirm_threshold"))
    cluster_s = float(_need(cfg, "scene_edited_cluster_seconds"))
    ret_sim = float(_need(cfg, "scene_edited_return_similarity"))
    ret_ratio = float(_need(cfg, "scene_edited_return_ratio"))
    max_insert = float(_need(cfg, "scene_edited_max_insert_seconds"))
    probe = float(_need(cfg, "scene_edited_confirm_probe_seconds"))

    fps = sig.fps()
    k = max(1, int(round(win * fps)))
    n = sig.n
    d1 = np.zeros(n, dtype=np.float32)
    dk = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        d1[i] = distance(sig, i, i - 1, w)
    for i in range(k, n):
        dk[i] = distance(sig, i, i - k, w)

    cands = []  # (time, score, kind, half_width_seconds)
    for i in range(1, n):
        lo, hi = max(1, i - 2), min(n, i + 3)
        if d1[i] >= cut_thr and d1[i] >= d1[lo:hi].max():
            cands.append((float(sig.t[i]), float(d1[i]), "cut", 0.0))
    # A gradual change must stand out from the motion around it. A dissolve
    # between two settled shots peaks far above the change measured just
    # before and just after it; a pan or a moving camera changes about as
    # much on either side, so its peak is not a transition (V18's pan inside
    # the extreme close-up). contrast = peak / larger of the two backgrounds.
    contrast_min = float(_need(cfg, "scene_edited_gradual_contrast"))
    gradual_rejected = []
    for i in range(k, n):
        lo, hi = max(k, i - k), min(n, i + k + 1)
        if dk[i] >= grad_thr and dk[i] >= dk[lo:hi].max():
            mid = float((sig.t[i - k] + sig.t[i]) / 2.0)
            pre_bg = dk[max(k, i - 2 * k):max(k, i - k + 1)]
            post_bg = dk[min(n, i + k):min(n, i + 2 * k + 1)]
            bg = max(float(pre_bg.mean()) if len(pre_bg) else 0.0,
                     float(post_bg.mean()) if len(post_bg) else 0.0)
            contrast = float(dk[i]) / max(bg, 1e-3)
            if contrast < contrast_min:
                gradual_rejected.append((round(mid, 3),
                    "continuous motion, not a transition (contrast {:.2f} < {})"
                    .format(contrast, contrast_min)))
                continue
            cands.append((mid, float(dk[i]), "gradual",
                          float((sig.t[i] - sig.t[i - k]) / 2.0)))

    # One boundary per transition: cluster nearby candidates. A cut inside a
    # cluster fixes the time exactly; otherwise the strongest gradual wins.
    cands.sort()
    clusters = []
    for c in cands:
        if clusters and c[0] - clusters[-1][-1][0] <= cluster_s:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    picked = []
    for cl in clusters:
        cuts = [c for c in cl if c[2] == "cut"]
        best = max(cuts or cl, key=lambda c: c[1])
        span = max(c[3] for c in cl) + (cl[-1][0] - cl[0][0]) / 2.0
        picked.append({"t": best[0], "score": best[1], "kind": best[2],
                       "half": max(best[3], span)})

    # Confirm: the picture a little before the transition must differ from
    # the picture a little after it. Flashes, caption changes and motion blur
    # return to the same picture and fail here.
    confirmed, rejected = [], list(healed) + list(gradual_rejected)
    for b in picked:
        a0 = b["t"] - b["half"] - 0.04
        b0 = b["t"] + b["half"] + 0.04
        pre = _thin(_window(sig, a0 - probe, a0), 2)
        post = _thin(_window(sig, b0, b0 + probe), 2)
        near = window_distance(sig, pre, post, w)
        # The same test one probe-length further out on both sides. A whip or
        # shake that leaves the shot and comes straight back looks different
        # right at the edge but identical a moment further out; a real cut is
        # different at both distances. Skipped where the video has no room.
        pre_far = _thin(_window(sig, a0 - 2 * probe, a0 - probe), 2)
        post_far = _thin(_window(sig, b0 + probe, b0 + 2 * probe), 2)
        far = window_distance(sig, pre_far, post_far, w) \
            if pre_far and post_far else near
        b["confirm"] = min(near, far)
        b["pre"], b["post"] = pre, post
        if b["confirm"] >= confirm:
            confirmed.append(b)
        else:
            rejected.append((round(b["t"], 3), "same picture before and after "
                             "(confirm {:.3f} < {})".format(b["confirm"], confirm)))

    # Inserts that return: A -> effect -> A within max_insert. Both edges go.
    # "Returns" means the picture after is close to the picture before, either
    # absolutely (< ret_sim) or relative to how different the insert was from
    # both sides (< ret_ratio of the smaller one) - footage keeps moving while
    # an effect sits on it, so the return is rarely exact.
    keep = [True] * len(confirmed)
    for i, b in enumerate(confirmed):
        if not keep[i]:
            continue
        for j in range(i + 1, len(confirmed)):
            c = confirmed[j]
            if c["t"] - b["t"] > max_insert:
                break
            back = window_distance(sig, b["pre"], c["post"], w)
            mid = _thin(_window(sig, b["t"] + b["half"] + 0.04,
                                c["t"] - c["half"] - 0.04), 2)
            away = min(window_distance(sig, b["pre"], mid, w),
                       window_distance(sig, mid, c["post"], w)) if mid else 1.0
            if back < ret_sim or back < ret_ratio * away:
                for m in range(i, j + 1):
                    if keep[m]:
                        keep[m] = False
                        rejected.append((round(confirmed[m]["t"], 3),
                                         "insert that returns to the same shot "
                                         "within {:.1f}s (back {:.3f}, away {:.3f})"
                                         .format(c["t"] - b["t"], back, away)))
                break
    bounds = [b for b, k_ in zip(confirmed, keep) if k_]
    return [(round(b["t"], 3), "cut" if b["kind"] == "cut" else "transition",
             round(b["score"], 4)) for b in bounds], rejected


# --------------------------------------------------------------------------
# RAW path
# --------------------------------------------------------------------------

def _mean_sig(sig, idx):
    h = sig.h[idx].mean(axis=0)
    g = sig.g[idx].mean(axis=0)
    g = g - g.mean()
    nrm = float(np.linalg.norm(g))
    return h, (g / nrm if nrm > 1e-4 else np.zeros_like(g))


def _dist_vec(a, b, w):
    tvd = float(np.abs(a[0] - b[0]).sum() / 6.0)
    ncc = float(np.dot(a[1], b[1]))
    return w * (1.0 - ncc) / 2.0 + (1.0 - w) * tvd


def segment_raw(sig, duration, cfg):
    """Sustained change of situation within one continuous take."""
    w = float(_need(cfg, "scene_structural_weight"))
    step = float(_need(cfg, "scene_raw_sample_seconds"))
    anchor_s = float(_need(cfg, "scene_raw_anchor_seconds"))
    thr = float(_need(cfg, "scene_raw_drift_threshold"))
    sustain = float(_need(cfg, "scene_raw_sustain_seconds"))
    smooth_s = float(_need(cfg, "scene_raw_smooth_seconds"))

    bounds, diag = [], []
    anchor_start = 0.0
    anchor = _mean_sig(sig, _window(sig, 0.0, anchor_s) or [0])
    crossing = None
    t = anchor_s
    while t < duration:
        idx = _window(sig, t - smooth_s / 2.0, t + smooth_s / 2.0)
        if idx:
            cur = _mean_sig(sig, idx)
            d = _dist_vec(cur, anchor, w)
            diag.append((round(t, 2), round(d, 4)))
            if d > thr:
                if crossing is None:
                    crossing = t
                elif t - crossing >= sustain:
                    # Boundary where the change began; the new anchor is the
                    # state the picture has settled into, not the move itself.
                    bounds.append((round(crossing, 3), "drift", round(d, 4)))
                    anchor = _mean_sig(sig, _window(sig, t - anchor_s, t) or idx)
                    crossing = None
            else:
                crossing = None
        t += step
    return bounds, diag


def merge_similar(sig, scenes, cfg, w):
    """Merge neighbouring scenes whose average pictures are alike - the camera
    moved away and came back, the situation did not change."""
    sim = float(_need(cfg, "scene_merge_similarity"))
    changed = True
    notes = []
    while changed and len(scenes) > 1:
        changed = False
        best, best_i = None, None
        for i in range(len(scenes) - 1):
            a = _window(sig, scenes[i]["start"], scenes[i]["end"])
            b = _window(sig, scenes[i + 1]["start"], scenes[i + 1]["end"])
            if not a or not b:
                continue
            d = _dist_vec(_mean_sig(sig, a), _mean_sig(sig, b), w)
            if d < sim and (best is None or d < best):
                best, best_i = d, i
        if best_i is not None:
            notes.append((scenes[best_i + 1]["start"],
                          "merged: neighbours alike ({:.3f} < {})".format(best, sim)))
            scenes[best_i]["end"] = scenes[best_i + 1]["end"]
            del scenes[best_i + 1]
            changed = True
    return scenes, notes


def consolidate_edited(sig, scenes, cfg, w):
    """1.5.0 (28 Oct 2026). EDITED path, only when the shot count is above
    scene_edited_max_count_absolute_per_asset. A long compilation reel (B1-V29
    MDG_6235 322 shots in 655 s, B1-V30 MDG_0375 411 in 996 s) cuts every 1.5
    to 2 s between angles of the same patient at the same stage; a scene per
    shot is noise. Shots are grouped into the moments where something
    different happens - a new patient, consultation to marking, marking to
    implantation, post-op:
      1. a group shorter than scene_edited_consolidate_min_seconds joins the
         neighbour it looks more like, shortest first;
      2. then the two most alike neighbouring groups (average picture over the
         whole group) merge, repeatedly, until every neighbouring pair is at
         least scene_edited_consolidate_similarity apart.
    Measured on the two reels: 0.20 with 6 s gives 49 and 55 groups of about
    13 s, one per patient stage when checked by eye; 0.25 already swallows
    several patients into one 3-minute group. Transcript and OCR are per
    second and per frame and are not affected by how scenes are grouped."""
    sim = float(_need(cfg, "scene_edited_consolidate_similarity"))
    min_len = float(_need(cfg, "scene_edited_consolidate_min_seconds"))
    notes = []
    means = [_mean_sig(sig, _window(sig, s["start"], s["end"])) for s in scenes]

    def dist(i):
        if means[i] is None or means[i + 1] is None:
            return 0.0
        return _dist_vec(means[i], means[i + 1], w)

    while len(scenes) > 1:
        d = [dist(i) for i in range(len(scenes) - 1)]
        lens = [s["end"] - s["start"] for s in scenes]
        short = [i for i, n in enumerate(lens) if n < min_len - 1e-9]
        if short:
            i = min(short, key=lambda k: lens[k])
            j = 0 if i == 0 else i - 1 if i == len(scenes) - 1 else (
                i - 1 if d[i - 1] <= d[i] else i)
            why = "consolidated: group of {:.2f}s under {}s".format(lens[i], min_len)
        else:
            j = int(np.argmin(d))
            if d[j] >= sim:
                break
            why = "consolidated: groups alike ({:.3f} < {})".format(d[j], sim)
        notes.append((scenes[j + 1]["start"], why))
        scenes[j]["end"] = scenes[j + 1]["end"]
        del scenes[j + 1]
        idx = _window(sig, scenes[j]["start"], scenes[j]["end"])
        means[j] = _mean_sig(sig, idx) if idx else None
        del means[j + 1]
    return scenes, notes


# --------------------------------------------------------------------------
# shared: scenes from boundaries
# --------------------------------------------------------------------------

def scenes_from_bounds(duration, bounds):
    duration = round(float(duration), 3)
    inner = sorted({round(t, 3): (src, sc) for t, src, sc in bounds
                    if 0.0 < t < duration}.items())
    starts = [(0.0, ("start", None))] + inner
    out = []
    for i, (s, (src, sc)) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else duration
        out.append({"start": round(s, 3), "end": round(e, 3),
                    "boundary_source": src, "score": sc})
    return out


def enforce_min_duration(scenes, min_dur):
    """Remove the weaker edge of the shortest too-short scene, repeatedly. A
    first or last scene loses its only interior edge. Weaker = lower score."""
    notes = []
    while len(scenes) > 1:
        lens = [s["end"] - s["start"] for s in scenes]
        i = int(np.argmin(lens))
        if lens[i] >= min_dur - 1e-9:
            break
        if i == 0:
            drop = 1
        elif i == len(scenes) - 1:
            drop = i
        else:
            left = scenes[i]["score"] or 0.0
            right = scenes[i + 1]["score"] or 0.0
            drop = i if left <= right else i + 1
        notes.append((scenes[drop]["start"],
                      "removed: scene of {:.2f}s under {}s".format(lens[i], min_dur)))
        scenes[drop - 1]["end"] = scenes[drop]["end"]
        del scenes[drop]
    return scenes, notes


def drop_blank_scenes(sig, scenes, cfg):
    """A black or white stretch (fade out, flash frame) is not a scene. Blank
    means nearly uniform AND nearly black or nearly white - a smooth but
    coloured picture (a wall, a gradient) is content, not blank."""
    blank_std = float(_need(cfg, "scene_blank_std_max"))
    blank_luma = float(_need(cfg, "scene_blank_luma_margin"))
    notes = []
    i = 0
    while len(scenes) > 1 and i < len(scenes):
        idx = _window(sig, scenes[i]["start"], scenes[i]["end"])
        if idx:
            lum = sig.luma[idx]
            blank = (sig.spread[idx] <= blank_std) & (
                (lum <= blank_luma) | (lum >= 1.0 - blank_luma))
        if idx and float(blank.mean()) >= 0.8:
            notes.append((scenes[i]["start"], "blank scene merged into neighbour"))
            if i == 0:
                scenes[1]["start"] = scenes[0]["start"]
                scenes[1]["boundary_source"] = "start"
            else:
                scenes[i - 1]["end"] = scenes[i]["end"]
            del scenes[i]
            continue
        i += 1
    return scenes, notes


def segment(video_path, duration, edit_class, cfg):
    """Returns a dict:
        scenes      [{index, start, end, boundary_source, detection_method, confidence}]
        path        'EDITED' or 'RAW'
        boundaries  raw boundaries before min-duration/merge clean-up
        rejected    [(time, reason)] candidates thrown out, with why
        diag        path-specific diagnostics
    """
    duration = round(float(duration), 3)
    cls = (edit_class or "").upper()
    if cls in ("RAW", "EDITED"):
        path = cls
    else:
        path = str(_need(cfg, "scene_unclassified_mode")).upper()
        if path not in ("RAW", "EDITED"):
            raise SceneError("scene_unclassified_mode must be RAW or EDITED")

    min_dur = float(_need(cfg, "scene_edited_min_duration_seconds" if path == "EDITED"
                          else "scene_min_duration_seconds"))
    cut_min = float(_need(cfg, "scene_cut_detection_min_duration_seconds"))
    method = "{}:{}:{}".format(MODULE_VERSION, path.lower(),
                               _need(cfg, "scene_drift_metric"))
    if duration < cut_min:
        return {"scenes": [{"index": 0, "start": 0.0, "end": duration,
                            "boundary_source": "start",
                            "detection_method": method + ":too_short",
                            "confidence": None}],
                "path": path, "boundaries": [], "rejected": [], "diag": []}

    # Brief section 4: cuts found on a file marked raw should correct it. With
    # scene_raw_cut_check on, a RAW video is decoded once and run through the
    # EDITED cut/transition detector - cuts and transitions only, never drift.
    # EVIDENCE ONLY (user decision 29 Sep 2026): on the 14 hand-checked raw
    # test videos the edited detector reports camera moves as cuts or
    # transitions on 9 of them, so a hit is recorded for human review and
    # never reclassifies the file or changes its scenes. The scene stays one
    # single take either way; the check result rides on it as cut_check.
    if path == "RAW" and bool(_need(cfg, "scene_raw_cut_check")):
        sig = compute_signatures(video_path, cfg, mask=False)
        checked = segment_signatures(sig, duration, "EDITED", cfg)
        kept = checked["scenes"][1:]
        check = {"measured": True, "path_checked": "EDITED",
                 "cuts": [s["start"] for s in kept if s["boundary_source"] == "cut"],
                 "transitions": [s["start"] for s in kept
                                 if s["boundary_source"] == "transition"],
                 "needs_review": bool(kept)}
        if float(_need(cfg, "scene_raw_drift_threshold")) >= 1.0:
            return {"scenes": [{"index": 0, "start": 0.0, "end": duration,
                                "boundary_source": "start",
                                "detection_method": method + ":single_take:cut_checked",
                                "confidence": None, "cut_check": check}],
                    "path": path, "boundaries": [], "rejected": checked["rejected"],
                    "diag": [], "cut_check": check}
        out = segment_signatures(sig, duration, path, cfg)
        out["scenes"][0]["cut_check"] = check
        out["cut_check"] = check
        return out

    # RAW footage is one continuous take from the camera, so it has no cuts.
    # Agreed rule (25 Sep 2026, every test video checked by hand): a raw
    # video is ONE scene. scene_raw_drift_threshold >= 1.0 - distance never
    # exceeds 1 - states that rule, and skips the decode entirely. A value
    # below 1 switches the situation-change detector back on.
    if path == "RAW" and float(_need(cfg, "scene_raw_drift_threshold")) >= 1.0:
        return {"scenes": [{"index": 0, "start": 0.0, "end": duration,
                            "boundary_source": "start",
                            "detection_method": method + ":single_take",
                            "confidence": None}],
                "path": path, "boundaries": [], "rejected": [], "diag": []}

    sig = compute_signatures(video_path, cfg, mask=(path == "EDITED"))
    return segment_signatures(sig, duration, path, cfg)


def segment_signatures(sig, duration, path, cfg):
    """Everything segment() does after decoding, on given signatures."""
    duration = round(float(duration), 3)
    path = path.upper()
    min_dur = float(_need(cfg, "scene_edited_min_duration_seconds" if path == "EDITED"
                          else "scene_min_duration_seconds"))
    method = "{}:{}:{}".format(MODULE_VERSION, path.lower(),
                               _need(cfg, "scene_drift_metric"))
    w = float(_need(cfg, "scene_structural_weight"))
    if path == "EDITED":
        bounds, rejected = segment_edited(sig, duration, cfg)
        diag = []
    else:
        bounds, diag = segment_raw(sig, duration, cfg)
        rejected = []

    scenes = scenes_from_bounds(duration, bounds)
    scenes, n1 = drop_blank_scenes(sig, scenes, cfg)
    scenes, n2 = enforce_min_duration(scenes, min_dur)
    n3 = []
    if path == "RAW":
        scenes, n3 = merge_similar(sig, scenes, cfg, w)
    # 1.5.0: off unless the target's pipeline_config turns it on, so a target
    # without the keys keeps failing over-cap assets exactly as before
    consolidated = False
    if (path == "EDITED" and bool(cfg.get("scene_edited_consolidate_enabled", False))
            and len(scenes) > int(_need(cfg, "scene_edited_max_count_absolute_per_asset"))):
        shots = len(scenes)
        scenes, n4 = consolidate_edited(sig, scenes, cfg, w)
        n3 = n3 + n4
        consolidated = {"shots": shots, "groups": len(scenes)}
    rejected = rejected + [(round(t, 3), r) for t, r in n1 + n2 + n3]

    for i, sc in enumerate(scenes):
        sc["index"] = i
        sc["detection_method"] = method
        sc["confidence"] = None
        sc.pop("score", None)
    if scenes:
        scenes[0]["start"] = 0.0
        scenes[0]["boundary_source"] = "start"
        scenes[-1]["end"] = duration
    return {"scenes": scenes, "path": path, "boundaries": bounds,
            "rejected": sorted(rejected), "diag": diag,
            "consolidated": consolidated}
