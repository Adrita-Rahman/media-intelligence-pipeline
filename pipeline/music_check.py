"""
Music in a video's soundtrack, with Apple's built-in sound classifier
(SoundAnalysis, SNClassifierIdentifierVersion1: about 300 sound classes, on
the Neural Engine, nothing leaves the Mac). 2 Nov 2026, for the user's
raw/edited rule: a video with background music is EDITED.

    music_profile("clip.mp4") -> {"windows": 412, "music_share": 0.92,
                                  "music_seconds": 185.3, "music": True,
                                  "speech_share": 0.1, "top_music_labels": {...}}

The soundtrack is cut into the classifier's ~1 s windows (half overlapping);
a window's music score is the best score of any music class (music itself,
singing, an instrument, a music genre).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

MIN_CONFIDENCE = 0.5   # a window clearly holds music (or speech)
SOME_MUSIC = 0.3       # a window holds some music, e.g. under a voiceover
# music in the video: SOME_MUSIC in at least MUSIC_SHARE of the windows, or
# MIN_CONFIDENCE for at least MUSIC_SECONDS (measured on test videos, 2 Nov 2026)
MUSIC_SHARE = 0.30
MUSIC_SECONDS = 5.0

# classifier labels that mean music is playing
MUSIC_WORDS = ("music", "singing", "song", "choir", "guitar", "piano", "keyboard", "organ",
               "drum", "bass", "violin", "fiddle", "cello", "trumpet", "saxophone", "flute",
               "synthesizer", "orchestra", "hip_hop", "rock", "pop", "jazz", "electronic",
               "techno", "dubstep", "reggae", "disco", "beatbox", "rapping", "harp", "ukulele",
               "banjo", "marimba", "xylophone", "glockenspiel", "harmonica", "accordion",
               "bagpipes", "tabla", "sitar", "gong", "cymbal", "tambourine", "snare", "bell")
NOT_MUSIC = ("speech", "music_box_off", "doorbell", "bicycle_bell", "bell_dinging_door")


def is_music_label(label):
    l = label.lower()
    return any(w in l for w in MUSIC_WORDS) and not any(w == l for w in NOT_MUSIC)


_OBSERVER = {}


def _observer_class():
    """One Objective-C class per process (pyobjc cannot define it twice)."""
    if "cls" not in _OBSERVER:
        import objc
        from Foundation import NSObject

        class MusicCheckObserver(NSObject, protocols=[objc.protocolNamed("SNResultsObserving")]):
            def request_didProduceResult_(self, request, result):
                self.windows.append({c.identifier(): float(c.confidence())
                                     for c in result.classifications()})

            def request_didFailWithError_(self, request, error):
                self.windows.append({"__error__": str(error)})

            def requestDidComplete_(self, request):
                pass
        _OBSERVER["cls"] = MusicCheckObserver
    return _OBSERVER["cls"]


def _classify(audio_path):
    import SoundAnalysis
    from Foundation import NSURL

    req, err = SoundAnalysis.SNClassifySoundRequest.alloc().initWithClassifierIdentifier_error_(
        SoundAnalysis.SNClassifierIdentifierVersion1, None)
    if req is None:
        raise RuntimeError("sound classifier unavailable: {}".format(err))
    analyzer, err = SoundAnalysis.SNAudioFileAnalyzer.alloc().initWithURL_error_(
        NSURL.fileURLWithPath_(str(audio_path)), None)
    if analyzer is None:
        raise RuntimeError("cannot open audio {}: {}".format(audio_path, err))
    obs = _observer_class().alloc().init()
    obs.windows = windows = []
    ok, err = analyzer.addRequest_withObserver_error_(req, obs, None)
    if not ok:
        raise RuntimeError("cannot start sound analysis: {}".format(err))
    analyzer.analyze()
    windows = list(obs.windows)
    errs = [w["__error__"] for w in windows if "__error__" in w]
    if errs:
        raise RuntimeError("sound analysis failed: {}".format(errs[0]))
    wd = req.windowDuration()            # CMTime: (value, timescale, flags, epoch)
    return windows, float(wd[0]) / float(wd[1] or 1), float(req.overlapFactor())


def music_profile(media_path):
    """Music share of a video or audio file. No audio stream -> music_share 0."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media_path),
                            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
            return {"windows": 0, "music_share": 0.0, "music_seconds": 0.0, "music": False,
                    "speech_share": 0.0, "top_music_labels": {}, "audio": False}
        windows, win_s, overlap = _classify(wav)
    hop = win_s * (1.0 - overlap)
    labels, some, strong, speechy = {}, 0, 0, 0
    for w in windows:
        best = max([v for k, v in w.items() if is_music_label(k)] or [0.0])
        some += best >= SOME_MUSIC
        strong += best >= MIN_CONFIDENCE
        speechy += w.get("speech", 0.0) >= MIN_CONFIDENCE
        for k, v in w.items():
            if v >= MIN_CONFIDENCE and is_music_label(k):
                labels[k] = labels.get(k, 0) + 1
    n = len(windows)
    share = round(some / n, 3) if n else 0.0
    seconds = round(strong * hop, 1)
    return {"windows": n, "music_share": share, "music_seconds": seconds,
            "music": share >= MUSIC_SHARE or seconds >= MUSIC_SECONDS,
            "speech_share": round(speechy / n, 3) if n else 0.0, "audio": True,
            "top_music_labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])[:5])}

if __name__ == "__main__":
    import json
    import sys
    for f in sys.argv[1:]:
        print(json.dumps(music_profile(f)))
