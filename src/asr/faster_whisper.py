"""faster-whisper adapter — the first (and for now only) ASRModel implementation.

CTranslate2 backend, CPU-only everywhere (including Apple Silicon — no MPS), which
is exactly what deploys to HF Spaces / Modal. Chunk-based: each transcribe() call
pads its audio to a 30s mel window internally, so per-call cost is near-constant
regardless of chunk length — keep chunks long enough to amortize it (see compare.py).
"""

from __future__ import annotations

import numpy as np

# Absolute import: resolves to the installed `faster_whisper` package, not this module.
from faster_whisper import WhisperModel


class FasterWhisperASR:
    """Wraps faster-whisper behind the ASRModel contract (see asr/base.py)."""

    streams_natively = False  # chunk-based, not a continuous stream
    supported_languages = ["en", "fr"]  # the project's languages; the model itself supports ~99

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.name = f"faster_whisper_{model_size}"
        self._model: WhisperModel | None = None

    def load(self) -> None:
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray, source_lang: str) -> str:
        if self._model is None:
            raise RuntimeError("FasterWhisperASR.load() must be called before transcribe()")
        segments, _ = self._model.transcribe(audio, language=source_lang, beam_size=self.beam_size)
        return " ".join(seg.text.strip() for seg in segments)
