"""Deterministic tests for src/audio.py.

Shaping and phrasing are pure and asserted exactly. The neural VAD is exercised by
one real-clip integration test (skipped if the clip is absent) — never asserted
against synthetic signals, which Silero is not trained to fire on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio import (
    SAMPLE_RATE,
    Segment,
    detect_speech,
    group_into_phrases,
    prepare,
    resample,
    split_phrases,
    to_mono,
)


# --- shaping: to_mono -------------------------------------------------------


def test_to_mono_passthrough_1d():
    a = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = to_mono(a)
    assert out.ndim == 1
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, a)


def test_to_mono_averages_frames_by_channels():
    # (samples, channels): channel axis is the minor one.
    stereo = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.5, 3.0])


def test_to_mono_averages_channels_by_frames():
    # (channels, samples): also handled — channel axis is still the minor one.
    stereo = np.array([[0.0, 2.0, 4.0], [1.0, 2.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.5, 2.0, 5.0])


def test_to_mono_rejects_3d():
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


# --- shaping: resample ------------------------------------------------------


def test_resample_noop_when_rates_match():
    a = np.linspace(-1, 1, 1000, dtype=np.float32)
    out = resample(a, SAMPLE_RATE, SAMPLE_RATE)
    np.testing.assert_array_equal(out, a)


def test_resample_changes_length_by_ratio():
    a = np.zeros(48000, dtype=np.float32)
    out = resample(a, 48000, 16000)
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 1  # 3:1 downsample, allow off-by-one rounding


def test_resample_preserves_constant_signal():
    # A DC signal has no high frequencies, so a good resampler reproduces it in
    # steady state. The polyphase filter rings briefly at the edges (warm-up) —
    # expected; check the interior, away from the boundaries.
    a = np.ones(48000, dtype=np.float32)
    out = resample(a, 48000, 16000)
    np.testing.assert_allclose(out[200:-200], 1.0, atol=1e-3)


def test_resample_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        resample(np.zeros(10, dtype=np.float32), 0, 16000)


def test_prepare_combines_mono_and_resample():
    stereo_48k = np.ones((48000, 2), dtype=np.float32)
    out = prepare(stereo_48k, 48000, 16000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 1


# --- VAD wrapper options ----------------------------------------------------


def test_detect_speech_defaults_to_no_padding(monkeypatch):
    seen = {}

    def fake_get_speech_timestamps(audio, options, sampling_rate):  # noqa: ANN001
        seen["speech_pad_ms"] = options.speech_pad_ms
        return []

    monkeypatch.setattr("src.audio.get_speech_timestamps", fake_get_speech_timestamps)
    detect_speech(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert seen["speech_pad_ms"] == 0


def test_detect_speech_accepts_custom_padding(monkeypatch):
    seen = {}

    def fake_get_speech_timestamps(audio, options, sampling_rate):  # noqa: ANN001
        seen["speech_pad_ms"] = options.speech_pad_ms
        return []

    monkeypatch.setattr("src.audio.get_speech_timestamps", fake_get_speech_timestamps)
    detect_speech(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, speech_pad_ms=120)
    assert seen["speech_pad_ms"] == 120


# --- phrasing: group_into_phrases (pure boundary logic) ---------------------


def test_group_empty():
    assert group_into_phrases([], 0.5) == []


def test_group_single_segment_unchanged():
    segs = [Segment(1.0, 2.0)]
    assert group_into_phrases(segs, 0.5) == [Segment(1.0, 2.0)]


def test_group_merges_short_gap():
    # gap of 0.3s < 0.5s pause -> one phrase spanning both.
    segs = [Segment(0.0, 1.0), Segment(1.3, 2.0)]
    assert group_into_phrases(segs, 0.5) == [Segment(0.0, 2.0)]


def test_group_splits_long_gap():
    # gap of 0.8s >= 0.5s pause -> two phrases.
    segs = [Segment(0.0, 1.0), Segment(1.8, 2.5)]
    assert group_into_phrases(segs, 0.5) == [Segment(0.0, 1.0), Segment(1.8, 2.5)]


def test_group_gap_exactly_pause_splits():
    # gap == pause is a boundary (merge is strictly "< pause").
    segs = [Segment(0.0, 1.0), Segment(1.5, 2.0)]
    assert group_into_phrases(segs, 0.5) == [Segment(0.0, 1.0), Segment(1.5, 2.0)]


def test_group_mixed_run():
    segs = [
        Segment(0.0, 1.0),
        Segment(1.2, 1.8),  # merge (gap 0.2)
        Segment(3.0, 3.5),  # split (gap 1.2)
        Segment(3.6, 4.0),  # merge (gap 0.1)
    ]
    assert group_into_phrases(segs, 0.5) == [Segment(0.0, 1.8), Segment(3.0, 4.0)]


def test_group_extends_over_overlap():
    # Overlapping/nested segments still produce a single covering span.
    segs = [Segment(0.0, 2.0), Segment(1.0, 1.5)]
    assert group_into_phrases(segs, 0.5) == [Segment(0.0, 2.0)]


# --- VAD integration (neural; real clip, skipped if absent) -----------------

_CLIP = Path("clips/en/live.wav")


@pytest.mark.skipif(not _CLIP.exists(), reason="real clip not present")
def test_detect_speech_finds_speech_in_real_clip():
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(_CLIP), sampling_rate=SAMPLE_RATE)
    segments = detect_speech(audio, SAMPLE_RATE)
    assert segments, "VAD found no speech in a clip that is mostly speech"
    duration = len(audio) / SAMPLE_RATE
    covered = sum(s.duration for s in segments)
    assert 0.0 < covered <= duration + 1e-6
    assert covered > 0.5 * duration  # the clip is predominantly speech


@pytest.mark.skipif(not _CLIP.exists(), reason="real clip not present")
def test_split_phrases_returns_segments_for_real_clip():
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(_CLIP), sampling_rate=SAMPLE_RATE)
    phrases = split_phrases(audio, SAMPLE_RATE, pause_seconds=0.5)
    assert phrases
    # Phrases are ordered and non-overlapping.
    for earlier, later in zip(phrases, phrases[1:]):
        assert earlier.end <= later.start
