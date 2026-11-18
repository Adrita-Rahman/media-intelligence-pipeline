"""
Whisper on the M4's GPU through MLX, with the faster-whisper WhisperModel
interface that step2_transcribe uses (2 Nov 2026, user decision: "option 2").

Chosen with the environment variable STEP2_WHISPER_ENGINE=mlx; step2_analyse
.whisper_model() then returns MLXWhisperModel instead of faster-whisper, and
the writer records provider 'mlx-whisper'. Everything around the model - the
language allowlist, quality gates, repetition limits, retries - is unchanged.

Same as faster-whisper here: the model size (transcription_model), pinned
temperature (no sampling), condition_on_previous_text, word timestamps, and
the voice-activity filter (faster-whisper's own Silero VAD, applied before the
decode and mapped back exactly as faster-whisper does).
Different, and recorded in each transcript's run metadata:
  * greedy decoding: MLX Whisper has no beam search (beam_size is ignored)
  * no hotwords: MLX Whisper cannot bias towards a word list, so the clinic
    vocabulary is not applied (initial_prompt still works)
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

ENGINE_ENV = "STEP2_WHISPER_ENGINE"
PROVIDER = "mlx-whisper"
SAMPLE_RATE = 16000


def selected():
    return os.environ.get(ENGINE_ENV, "").strip().lower() == "mlx"


def provider():
    return PROVIDER if selected() else "faster-whisper"


def model_name(configured):
    """The model the transcript rows name: the MLX repo when MLX runs."""
    if not selected():
        return configured
    return os.environ.get("STEP2_WHISPER_MLX_REPO") or "mlx-community/whisper-{}-mlx".format(configured)


class MLXWhisperModel:
    def __init__(self, size):
        self.size = size
        # STEP2_WHISPER_MLX_REPO overrides the model, e.g. mlx-community/whisper-large-v3-turbo
        self.repo = os.environ.get("STEP2_WHISPER_MLX_REPO") or "mlx-community/whisper-{}-mlx".format(size)
        self._model = None

    def _load(self):
        if self._model is None:
            from mlx_whisper.load_models import load_model
            self._model = load_model(self.repo)
        return self._model

    @staticmethod
    def _audio(audio):
        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio
            return decode_audio(audio, sampling_rate=SAMPLE_RATE)
        return np.asarray(audio, dtype=np.float32)

    def detect_language(self, audio):
        """-> (language, probability, [(language, probability), ...]) like faster-whisper."""
        import mlx.core as mx
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        model = self._load()
        # mx.array: pad_or_trim pads with mx.pad, which refuses numpy arrays -
        # so a clip shorter than 30 s failed here (2 Nov 2026)
        a = pad_or_trim(mx.array(self._audio(audio)), N_SAMPLES)
        mel = log_mel_spectrogram(a, n_mels=model.dims.n_mels)
        _tokens, probs = detect_language(model, mx.array(mel)[None])
        probs = probs[0] if isinstance(probs, list) else probs
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return ranked[0][0], float(ranked[0][1]), [(k, float(v)) for k, v in ranked]

    def transcribe(self, audio, beam_size=None, vad_filter=False,
                   condition_on_previous_text=True, temperature=(0.0,), language=None,
                   word_timestamps=False, initial_prompt=None, hotwords=None, **_ignored):
        import mlx_whisper
        a = self._audio(audio)
        ts_map = None
        if vad_filter:
            from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps
            chunks = get_speech_timestamps(a, VadOptions())
            if not chunks:
                return [], SimpleNamespace(language=language, engine=PROVIDER)
            a = np.concatenate([a[c["start"]:c["end"]] for c in chunks])
            ts_map = SpeechTimestampsMap(chunks, SAMPLE_RATE)
        res = mlx_whisper.transcribe(
            a, path_or_hf_repo=self.repo, verbose=None,
            temperature=tuple(temperature) if isinstance(temperature, (list, tuple)) else temperature,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt, word_timestamps=word_timestamps,
            language=language)

        def t(x, end=False):
            return ts_map.get_original_time(x, is_end=end) if ts_map else round(float(x), 3)

        segs = []
        for s in res.get("segments", []):
            words = [SimpleNamespace(start=t(w["start"]), end=t(w["end"], True), word=w["word"],
                                     probability=w.get("probability"))
                     for w in (s.get("words") or [])]
            segs.append(SimpleNamespace(
                start=t(s["start"]), end=t(s["end"], True), text=s.get("text", ""),
                words=words, avg_logprob=s.get("avg_logprob", 0.0),
                compression_ratio=s.get("compression_ratio", 0.0),
                no_speech_prob=s.get("no_speech_prob", 0.0)))
        # Two segments can map to the same original start (voice-activity gaps
        # collapse onto one instant); transcript chunks are unique per start, so
        # such a segment is joined onto the one before it, words and all (3 Nov 2026)
        merged = []
        for sg in segs:
            if merged and sg.start <= merged[-1].start:
                prev = merged[-1]
                prev.text = (prev.text.rstrip() + " " + sg.text.lstrip()).strip()
                prev.words = prev.words + sg.words
                prev.end = max(prev.end, sg.end)
                prev.avg_logprob = min(prev.avg_logprob, sg.avg_logprob)
                prev.compression_ratio = max(prev.compression_ratio, sg.compression_ratio)
                prev.no_speech_prob = min(prev.no_speech_prob, sg.no_speech_prob)
            else:
                merged.append(sg)
        segs = merged
        return segs, SimpleNamespace(language=res.get("language"), engine=PROVIDER,
                                     hotwords_ignored=bool(hotwords), beam_size_ignored=beam_size)
