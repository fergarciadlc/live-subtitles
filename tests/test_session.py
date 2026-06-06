"""Unit tests for LiveSession — the streaming orchestrator.

The ASR adapter, speech detection, translation and the clock are all injected with
fakes, so these assertions are deterministic: no faster-whisper, no onnxruntime, no
real time. The neural pieces are exercised elsewhere; here we test only the glue —
buffering, interim throttling, and commit/reset behaviour.
"""

from __future__ import annotations

import numpy as np

from src.audio import SAMPLE_RATE, Segment
from src.session import CaptionUpdate, LiveSession


class FakeASR:
    """An ASRModel that returns whatever text the test sets, counting its calls."""

    streams_natively = False
    supported_languages = ["en", "fr"]

    def __init__(self, text: str = "") -> None:
        self.name = "fake"
        self.text = text
        self.calls = 0

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, source_lang: str) -> str:
        self.calls += 1
        return self.text


def fake_translate(text: str, source_lang: str, target_lang: str) -> str:
    return f"<{target_lang}>{text}"


class Clock:
    """A hand-cranked monotonic clock so interim throttling is deterministic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def chunk(seconds: float) -> np.ndarray:
    """Silent 16 kHz mono audio of a given length (content is irrelevant — VAD is faked)."""
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def make_session(asr: FakeASR, *, detect, clock: Clock | None = None, **kwargs) -> LiveSession:
    return LiveSession(
        asr,
        source_lang="en",
        target_lang="fr",
        translate_fn=fake_translate,
        detect_speech_fn=detect,
        monotonic=clock or Clock(),
        **kwargs,
    )


def test_feed_buffers_until_step_threshold():
    asr = FakeASR("hello")
    session = make_session(asr, detect=lambda *a, **k: [Segment(0.0, 0.1)], step_seconds=1.0)
    assert session.feed(chunk(0.3), SAMPLE_RATE) is None  # below step → just buffer
    assert asr.calls == 0  # no VAD/ASR work attempted yet


def test_silence_does_not_transcribe():
    asr = FakeASR("hello")
    session = make_session(asr, detect=lambda *a, **k: [], step_seconds=0.4)
    assert session.feed(chunk(0.5), SAMPLE_RATE) is None
    assert asr.calls == 0  # nothing to transcribe on pure silence


def test_interim_is_throttled_by_clock():
    asr = FakeASR("partial one")
    clock = Clock()
    # Speech fills the whole buffer (zero trailing silence) → never commits, only interim.
    session = make_session(
        asr,
        detect=lambda buf, sr, **k: [Segment(0.0, len(buf) / sr)],
        clock=clock,
        step_seconds=0.4,
        interim_every=1.0,
    )

    first = session.feed(chunk(0.5), SAMPLE_RATE)
    assert isinstance(first, CaptionUpdate) and first.interim == "partial one"
    assert asr.calls == 1

    # Same clock time → throttled: no re-transcribe, nothing changes.
    asr.text = "partial two"
    assert session.feed(chunk(0.5), SAMPLE_RATE) is None
    assert asr.calls == 1

    # Advance past the interval → interim refreshes.
    clock.t = 1.0
    third = session.feed(chunk(0.5), SAMPLE_RATE)
    assert isinstance(third, CaptionUpdate) and third.interim == "partial two"
    assert asr.calls == 2


def test_commit_on_trailing_silence():
    asr = FakeASR("Hello world.")
    # 1st feed: speech ongoing (short trailing gap). 2nd: long enough gap to commit.
    segs = iter([[Segment(0.0, 0.7)], [Segment(0.0, 0.9)]])
    session = make_session(
        asr, detect=lambda *a, **k: next(segs), step_seconds=0.4, interim_every=0.0
    )

    assert session.feed(chunk(0.8), SAMPLE_RATE) is None  # waiting
    update = session.feed(chunk(0.8), SAMPLE_RATE)
    assert isinstance(update, CaptionUpdate)
    assert len(update.newly_committed) == 1
    line = update.newly_committed[0]
    assert line.source == "Hello world."
    assert line.target == "<fr>Hello world."
    assert update.interim == ""  # interim cleared once committed


def test_commit_splits_into_caption_units():
    asr = FakeASR("Bonjour. Comment ca va?")
    session = make_session(
        asr,
        detect=lambda buf, sr, **k: [Segment(0.0, 0.1)],  # big trailing gap → commit
        step_seconds=0.4,
        interim_every=0.0,
    )
    update = session.feed(chunk(0.8), SAMPLE_RATE)
    assert isinstance(update, CaptionUpdate)
    assert [line.source for line in update.newly_committed] == ["Bonjour.", "Comment ca va?"]


def test_interim_disabled_skips_live_transcription():
    asr = FakeASR("done.")
    segs = iter([[Segment(0.0, 0.7)], [Segment(0.0, 0.9)]])
    session = make_session(
        asr, detect=lambda *a, **k: next(segs), step_seconds=0.4, interim_every=0.0
    )

    assert session.feed(chunk(0.8), SAMPLE_RATE) is None
    assert asr.calls == 0  # no interim transcription happened
    update = session.feed(chunk(0.8), SAMPLE_RATE)
    assert update.newly_committed[0].source == "done."
    assert asr.calls == 1  # only the commit pass transcribed


def test_reconfigure_language_resets_live_state():
    asr = FakeASR("partial")
    session = make_session(
        asr,
        detect=lambda buf, sr, **k: [Segment(0.0, len(buf) / sr)],  # never commits
        step_seconds=0.4,
        interim_every=1.0,
    )
    session.feed(chunk(0.5), SAMPLE_RATE)
    assert session.captions.interim == "partial"

    session.reconfigure(source_lang="fr", target_lang="en")
    assert session.source_lang == "fr" and session.target_lang == "en"
    assert session.captions.interim == ""  # live line cleared
    # Buffer was reset too: a sub-step chunk buffers from scratch again.
    assert session.feed(chunk(0.3), SAMPLE_RATE) is None
