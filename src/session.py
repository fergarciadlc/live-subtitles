"""Per-session live captioning pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.asr.base import ASRModel
from src.audio import SAMPLE_RATE, Segment, detect_speech, prepare
from src.captions import CaptionState, Line
from src.phrase_boundaries import PhraseBoundaryConfig, PhraseBoundaryDetector
from src.text_units import split_caption_units
from src.translate import translate

SpeechDetector = Callable[..., list[Segment]]
TranslateFn = Callable[[str, str, str], str]


@dataclass(frozen=True)
class CaptionUpdate:
    """Caption state after a feed that changed visible output."""

    interim: str
    committed: tuple[Line, ...]
    newly_committed: tuple[Line, ...] = ()


class LiveSession:
    """Stateful audio-to-caption pipeline for one browser session."""

    def __init__(
        self,
        asr: ASRModel,
        *,
        source_lang: str,
        target_lang: str,
        boundary: PhraseBoundaryConfig = PhraseBoundaryConfig(),
        interim_every: float = 1.0,
        step_seconds: float = 0.4,
        interim_asr: ASRModel | None = None,
        translate_fn: TranslateFn = translate,
        detect_speech_fn: SpeechDetector = detect_speech,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create a live captioning session."""
        self._asr = asr
        self._interim_asr = interim_asr or asr
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._detector = PhraseBoundaryDetector(boundary)
        self._interim_every = interim_every
        self._step_samples = int(SAMPLE_RATE * step_seconds)
        self._translate = translate_fn
        self._detect_speech = detect_speech_fn
        self._now = monotonic

        self._captions = CaptionState()
        self._buf = np.empty(0, dtype=np.float32)
        self._last_interim: float | None = None

    @property
    def captions(self) -> CaptionState:
        """Current caption state."""
        return self._captions

    @property
    def interim_enabled(self) -> bool:
        return self._interim_every > 0

    def feed(self, audio: np.ndarray, sample_rate: int) -> CaptionUpdate | None:
        """Process one audio chunk and return an update when captions changed."""
        self._buf = np.concatenate([self._buf, prepare(audio, sample_rate)])
        if len(self._buf) < self._step_samples:
            return None  # not enough new audio yet to bother with VAD/ASR

        buf_seconds = len(self._buf) / SAMPLE_RATE
        segments = self._detect_speech(self._buf, SAMPLE_RATE)
        if not segments:
            # Only silence so far — keep the buffer short so it doesn't pile up.
            if buf_seconds > 1.0:
                self._buf = self._buf[-SAMPLE_RATE // 2 :]
            return None

        decision = self._detector.decide(buf_seconds, segments)
        if decision.should_commit:
            newly = self._commit(self._asr.transcribe(self._buf, self.source_lang))
            self._buf = np.empty(0, dtype=np.float32)  # start the next phrase
            self._last_interim = None
            return CaptionUpdate(self._captions.interim, self._captions.committed, newly)

        # Not committing — refresh the live preview, throttled to spare CPU.
        now = self._now()
        if self.interim_enabled and (
            self._last_interim is None or now - self._last_interim >= self._interim_every
        ):
            self._captions.update_interim(self._interim_asr.transcribe(self._buf, self.source_lang))
            self._last_interim = now
            return CaptionUpdate(self._captions.interim, self._captions.committed)

        return None

    def reconfigure(
        self,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
        boundary: PhraseBoundaryConfig | None = None,
        asr: ASRModel | None = None,
        interim_asr: ASRModel | None = None,
        interim_every: float | None = None,
    ) -> None:
        """Apply live control changes."""
        lang_changed = (source_lang is not None and source_lang != self.source_lang) or (
            target_lang is not None and target_lang != self.target_lang
        )
        if source_lang is not None:
            self.source_lang = source_lang
        if target_lang is not None:
            self.target_lang = target_lang
        if boundary is not None:
            self._detector = PhraseBoundaryDetector(boundary)
        if asr is not None:
            self._asr = asr
            if interim_asr is None:
                self._interim_asr = asr
        if interim_asr is not None:
            self._interim_asr = interim_asr
        if interim_every is not None:
            self._interim_every = interim_every
        if lang_changed:
            self._reset()

    def _commit(self, text: str) -> tuple[Line, ...]:
        """Commit final ASR text and translate new lines."""
        newly: list[Line] = []
        for unit in split_caption_units(text):
            self._captions.update_interim(unit)
            line = self._captions.commit()
            if line:
                target = self._translate(line.source, self.source_lang, self.target_lang)
                self._captions.set_translation(line.id, target)
                newly.append(line)
        return tuple(newly)

    def _reset(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)
        self._last_interim = None
        self._captions.update_interim("")
