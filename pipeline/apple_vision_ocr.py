"""
Apple Vision text recognition with the easyocr Reader interface (2 Nov 2026,
user decision: use the M4's Neural Engine for OCR - "option 2").

    reader = VisionReader(["en"])
    reader.readtext("frame.jpg")  ->  [(bbox, text, confidence), ...]

bbox is four [x, y] pixel points (top-left, top-right, bottom-right,
bottom-left, origin top-left), exactly as easyocr returns them, so the code
that filters, classifies and stores easyocr results runs unchanged on these.
VNRecognizeTextRequest at the accurate level runs on the Neural Engine; no
network, nothing leaves the Mac.
"""
from __future__ import annotations

LANGS = {"en": "en-US", "ms": "ms-MY"}
ENGINE = "apple_vision"


class VisionReader:
    def __init__(self, langs=("en",), correction=True):
        import Vision  # noqa: F401  (pyobjc-framework-Vision)
        self.langs = [LANGS.get(x, x) for x in langs]
        self.correction = bool(correction)

    def readtext(self, path):
        import Quartz
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(path))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if src is None:
            raise RuntimeError("cannot open image {}".format(path))
        return self._read(Quartz.CGImageSourceCreateImageAtIndex(src, 0, None), path)

    def readarray(self, img_bgr):
        """The same readings for an OpenCV BGR frame (video OCR)."""
        import cv2
        import Quartz
        from Foundation import NSData
        ok, png = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("cannot encode frame")
        data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        return self._read(Quartz.CGImageSourceCreateImageAtIndex(src, 0, None), "frame")

    def _read(self, img, path):
        import Quartz
        import Vision
        w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(self.langs)
        req.setUsesLanguageCorrection_(self.correction)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError("Vision failed on {}: {}".format(path, err))

        out = []
        for obs in req.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            c = cands[0]
            # normalised, origin bottom-left -> pixels, origin top-left
            b = obs.boundingBox()
            x0, y0 = b.origin.x * w, (1.0 - b.origin.y - b.size.height) * h
            x1, y1 = (b.origin.x + b.size.width) * w, (1.0 - b.origin.y) * h
            bbox = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            out.append((bbox, str(c.string()), float(c.confidence())))
        return out
