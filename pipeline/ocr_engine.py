#!/usr/bin/env python3
"""
The OCR engine and how a frame is read - every choice from pipeline_config.

  ocr_engine            "rapidocr" | "easyocr"
  ocr_engine_models     for rapidocr: det/rec model version and size
  ocr_engine_benchmark  the measured comparison the choice rests on (recorded,
                        read into every run's metadata; see
                        scripts/ocr_engine_bench.py)

A frame is read in passes, each a named variant that step2_ocr pools:

  native        the whole frame as stored (the source's own pixel grid)
  crop_upscale  every box whose letter height is under
                ocr_upscale_min_text_height_px is cropped with
                ocr_crop_padding_fraction of margin, enlarged by
                ocr_upscale_factor and read again. Small text is where the
                misreads were (CLASSYS 17 px, PROFHILO 12 px); enlarging the
                whole frame costs ocr_upscale_factor^2 as much to help the
                same few boxes.
  upscaled      the whole frame enlarged, when ocr_full_frame_upscale - for
                text too small to be detected at all at native size
  contrast      CLAHE on the enlarged frame, when ocr_contrast_normalise
  inverted      the enlarged frame inverted, when ocr_try_inverted - light
                text on a dark screen

No AI vision or language model is involved: a dedicated OCR engine plus
deterministic pre-processing.
"""
from __future__ import annotations

import math


def _need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: " + key)
    return cfg[key]


APPLE_VISION_ENV = "STEP2_OCR_ENGINE"   # =apple_vision: Apple's text recognition (2 Nov 2026)


def apple_vision_selected():
    import os
    return os.environ.get(APPLE_VISION_ENV, "").strip().lower() == "apple_vision"


def make_engine(cfg):
    """callable(img_bgr) -> [{"raw_text","confidence","points"}]"""
    name = _need(cfg, "ocr_engine")
    if apple_vision_selected():
        from apple_vision_ocr import VisionReader
        rd = VisionReader(list(_need(cfg, "ocr_languages")))

        def run(img):
            return [{"raw_text": t.strip(), "confidence": float(c), "points": b}
                    for b, t, c in rd.readarray(img) if (t or "").strip()]
        return run
    if name == "easyocr":
        import easyocr
        import torch
        torch.set_num_threads(int(_need(cfg, "ocr_torch_threads")))
        rd = easyocr.Reader(list(_need(cfg, "ocr_languages")), gpu=False,
                            verbose=False)

        def run(img):
            return [{"raw_text": t.strip(), "confidence": float(c),
                     "points": [[float(x), float(y)] for x, y in b]}
                    for b, t, c in rd.readtext(img) if (t or "").strip()]
        return run

    if name == "rapidocr":
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
        m = _need(cfg, "ocr_engine_models")
        params = {
            "Global.log_level": "error",
            # Every reading comes back; the confidence floor is applied by
            # step2_ocr, where it is recorded as a rejection_reason.
            "Global.text_score": 0.0,
            "Det.ocr_version": OCRVersion(m["det_version"]),
            "Det.model_type": ModelType(m["det_type"]),
            "Rec.ocr_version": OCRVersion(m["rec_version"]),
            "Rec.model_type": ModelType(m["rec_type"]),
        }
        if m.get("rec_lang"):
            params["Rec.lang_type"] = LangRec(m["rec_lang"])
        eng = RapidOCR(params=params)

        def run(img):
            r = eng(img)
            if r.boxes is None:
                return []
            return [{"raw_text": t.strip(), "confidence": float(s),
                     "points": [[float(x), float(y)] for x, y in b]}
                    for b, t, s in zip(r.boxes, r.txts, r.scores)
                    if (t or "").strip()]
        return run

    raise SystemExit("ocr_engine {!r} is not supported".format(name))


def engine_version(cfg):
    name = _need(cfg, "ocr_engine")
    if apple_vision_selected():
        import platform
        return "apple_vision VNRecognizeTextRequest accurate, macOS {}".format(platform.mac_ver()[0])
    try:
        if name == "rapidocr":
            import rapidocr
            m = _need(cfg, "ocr_engine_models")
            return "rapidocr {} det {}/{} rec {}/{}{}".format(
                getattr(rapidocr, "__version__", "?"), m["det_version"],
                m["det_type"], m["rec_version"], m["rec_type"],
                " " + m["rec_lang"] if m.get("rec_lang") else "")
        import easyocr
        return "easyocr {}".format(easyocr.__version__)
    except Exception:
        return name


def letter_height(points):
    """The box's short edge - letter height even on a tilted label."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points[:4]
    side = (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2
    top = math.hypot(x1 - x0, y1 - y0)
    return min(side, top)


def _scaled(readings, f, dx=0.0, dy=0.0):
    for r in readings:
        r["points"] = [[x / f + dx, y / f + dy] for x, y in r["points"]]
    return readings


def read_frame(img, engine, cfg):
    """{variant: [reading, ...]} for one BGR frame."""
    import cv2
    h, w = img.shape[:2]
    up = float(_need(cfg, "ocr_upscale_factor"))
    min_h = float(_need(cfg, "ocr_upscale_min_text_height_px"))
    pad = float(_need(cfg, "ocr_crop_padding_fraction"))

    variants = {"native": engine(img)}

    if bool(_need(cfg, "ocr_crop_upscale_enabled")):
        crops = []
        for r in variants["native"]:
            if letter_height(r["points"]) >= min_h:
                continue
            xs = [p[0] for p in r["points"]]
            ys = [p[1] for p in r["points"]]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            m = pad * max(bw, bh)
            x0 = int(max(0, min(xs) - m))
            y0 = int(max(0, min(ys) - m))
            x1 = int(min(w, max(xs) + m))
            y1 = int(min(h, max(ys) + m))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = cv2.resize(img[y0:y1, x0:x1], None, fx=up, fy=up,
                              interpolation=cv2.INTER_CUBIC)
            crops.extend(_scaled(engine(crop), up, x0, y0))
        variants["crop_upscale"] = crops

    big = None
    if (bool(_need(cfg, "ocr_full_frame_upscale"))
            or bool(_need(cfg, "ocr_contrast_normalise"))
            or bool(_need(cfg, "ocr_try_inverted"))):
        big = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    if bool(_need(cfg, "ocr_full_frame_upscale")):
        variants["upscaled"] = _scaled(engine(big), up)
    if bool(_need(cfg, "ocr_contrast_normalise")):
        lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        variants["contrast"] = _scaled(engine(cv2.cvtColor(
            cv2.merge([cl, A, B]), cv2.COLOR_LAB2BGR)), up)
    if bool(_need(cfg, "ocr_try_inverted")):
        variants["inverted"] = _scaled(engine(cv2.bitwise_not(big)), up)
    return variants


def change_score(prev_img, img):
    """Mean absolute grey-level difference, 0-1, on a small copy."""
    import cv2
    import numpy as np
    if prev_img is None:
        return None
    a = cv2.resize(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), (90, 160))
    b = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (90, 160))
    return float(np.mean(np.abs(a.astype("float32") - b.astype("float32")))) / 255.0
