"""The ASR adapter contract — the one abstraction the project hangs on.

Every speech-to-text model hides behind this interface so the pipeline and
`compare.py` never know which model they're using. Adding a new model means
adding a new adapter here; nothing else changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRModel(Protocol):
    """A speech-to-text model that turns audio into source-language text.

    Audio is always a 1-D float32 numpy array, mono, 16 kHz (what the mic and
    the file decoder both produce). The adapter absorbs the model's quirks
    (chunk-based vs streaming, sample-rate assumptions) so callers stay clean.
    """

    name: str  # stable identifier, e.g. "faster_whisper_small" — used in config + compare tables
    streams_natively: bool  # True if the model consumes a continuous stream; False if it's chunk-based
    supported_languages: list[str]  # source languages this adapter is wired to handle, e.g. ["en", "fr"]

    def load(self) -> None:
        """Load weights into memory. Called once before the first transcribe()."""
        ...

    def transcribe(self, audio: np.ndarray, source_lang: str) -> str:
        """Transcribe one chunk of audio into text in `source_lang`."""
        ...
