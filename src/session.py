"""Live captioning session — the streaming orchestrator (build-order step 7).

This is the per-session pipeline that `app.py` drives from a FastRTC stream. It is
the tested, transport-agnostic core that replaces the throwaway loop in `slice_2.py`:
that script *owned* a `while True` loop (block on the mic queue, drain to the live
edge, run VAD/ASR, decide commits). FastRTC instead calls a handler once per audio
frame, so the same work is turned inside-out into a per-chunk state machine here.

`feed(audio, sample_rate)` is the seam. Call it with each incoming chunk; it returns
a CaptionUpdate when the visible captions changed (a refreshed interim line and/or a
newly committed line) and None when nothing changed (still buffering, or silence).
The heavy model calls (ASR, MT) run inside feed(); the handler runs feed() on a worker
thread so the audio-receive path is never blocked — the streaming-era version of
slice_2's "drain to stay at the live edge".

Everything model- and transport-specific is injected (the ASR adapter, the translate
function, speech detection, the clock), so the orchestration is unit-testable with
fakes — no models, no onnxruntime, no real time. Production defaults wire the real
pieces from src/asr, src/translate, src/audio.

Why a custom path and not FastRTC's ReplyOnPause: ReplyOnPause runs its own VAD and
only hands you audio *after* the speaker pauses — that would kill the live interim
caption and make our adaptive endpointer (phrase_boundaries.py) and the pause/max
sliders dead code. We keep our own VAD + endpointer and use FastRTC purely as
transport.
"""

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
    """A snapshot of the captions after a feed() that changed something.

    `interim` is the live (not-yet-committed) source line; `committed` is the full
    ordered history; `newly_committed` are the lines this feed() just promoted, with
    their translations already attached (v1 translates inline on commit).
    """

    interim: str
    committed: tuple[Line, ...]
    newly_committed: tuple[Line, ...] = ()


class LiveSession:
    """Per-session streaming pipeline: audio chunks in, caption updates out.

    Holds the mutable per-session state (the not-yet-committed phrase buffer, the
    endpointer, the caption state). The ASR adapter and translate function are held
    by reference — they are shared, already-loaded resources, not owned here.

        session = LiveSession(asr, source_lang="en", target_lang="fr")
        update = session.feed(chunk, sample_rate)   # None until something changes
        if update: render(update)
    """

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
        """Wire a session.

        `interim_every` throttles the expensive live re-transcription (<= 0 disables
        the interim preview entirely — commit-only, the lightest CPU mode). `step_seconds`
        is the minimum audio accumulated before any VAD/ASR work runs. `interim_asr`
        lets a cheaper model drive the live preview while `asr` does the clean commit
        pass; it defaults to `asr`. The remaining args are injection seams for testing.
        """
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
        """The caption state, for rendering the full view between updates."""
        return self._captions

    @property
    def interim_enabled(self) -> bool:
        return self._interim_every > 0

    def feed(self, audio: np.ndarray, sample_rate: int) -> CaptionUpdate | None:
        """Process one incoming audio chunk; return a CaptionUpdate iff something changed.

        Mirrors one iteration of slice_2's loop: append to the phrase buffer, bail until
        there's enough new audio, run VAD, then either commit on a phrase boundary (a
        final clean transcription, split + translated) or refresh the throttled interim
        preview. Returns None while merely buffering or on silence.
        """
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
        interim_every: float | None = None,
    ) -> None:
        """Apply live control changes (the UI sliders/dropdowns) mid-stream.

        A language change resets the live buffer and interim line — the audio already
        buffered was spoken under the old direction, so finishing it would be incoherent.
        Committed history is left untouched. Swapping `asr` repoints both the commit and
        interim models (v1 shares one model for both).
        """
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
            self._interim_asr = asr
        if interim_every is not None:
            self._interim_every = interim_every
        if lang_changed:
            self._reset()

    def _commit(self, text: str) -> tuple[Line, ...]:
        """Promote final ASR text into one or more committed, translated caption lines."""
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
