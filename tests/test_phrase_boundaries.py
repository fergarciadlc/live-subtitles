"""Deterministic tests for streaming phrase-boundary decisions."""

from __future__ import annotations

import pytest

from src.audio import Segment
from src.phrase_boundaries import PhraseBoundaryConfig, PhraseBoundaryDetector


def detector() -> PhraseBoundaryDetector:
    return PhraseBoundaryDetector(
        PhraseBoundaryConfig(
            normal_pause_seconds=0.5,
            long_phrase_seconds=4.0,
            long_phrase_pause_seconds=0.2,
            max_phrase_seconds=7.0,
        )
    )


def test_no_speech_waits():
    decision = detector().decide(1.0, [])
    assert not decision.should_commit
    assert decision.reason == "no_speech"
    assert decision.trailing_silence == 1.0
    assert decision.required_pause_seconds == 0.5


def test_no_speech_waits_even_at_max_phrase_duration():
    decision = detector().decide(7.0, [])
    assert not decision.should_commit
    assert decision.reason == "no_speech"
    assert decision.trailing_silence == 7.0
    assert decision.required_pause_seconds == 0.2


def test_short_phrase_commits_after_normal_pause():
    decision = detector().decide(2.0, [Segment(0.0, 1.5)])
    assert decision.should_commit
    assert decision.reason == "pause"
    assert decision.trailing_silence == 0.5
    assert decision.required_pause_seconds == 0.5


def test_short_phrase_waits_before_normal_pause():
    decision = detector().decide(2.0, [Segment(0.0, 1.7)])
    assert not decision.should_commit
    assert decision.reason == "waiting"
    assert decision.trailing_silence == pytest.approx(0.3)
    assert decision.required_pause_seconds == 0.5


def test_long_phrase_commits_after_short_pause():
    decision = detector().decide(4.5, [Segment(0.0, 4.3)])
    assert decision.should_commit
    assert decision.reason == "pause"
    assert decision.trailing_silence == pytest.approx(0.2)
    assert decision.required_pause_seconds == 0.2


def test_long_phrase_waits_before_short_pause():
    decision = detector().decide(4.5, [Segment(0.0, 4.31)])
    assert not decision.should_commit
    assert decision.reason == "waiting"
    assert decision.trailing_silence == pytest.approx(0.19)
    assert decision.required_pause_seconds == 0.2


def test_max_phrase_commits_even_without_pause():
    decision = detector().decide(7.0, [Segment(0.0, 7.0)])
    assert decision.should_commit
    assert decision.reason == "max_phrase"
    assert decision.required_pause_seconds == 0.2


def test_exact_pause_boundary_commits():
    decision = detector().decide(3.0, [Segment(0.0, 2.5)])
    assert decision.should_commit
    assert decision.reason == "pause"


def test_rejects_negative_config_values():
    with pytest.raises(ValueError):
        PhraseBoundaryConfig(normal_pause_seconds=-0.1)


def test_rejects_long_phrase_pause_larger_than_normal_pause():
    with pytest.raises(ValueError):
        PhraseBoundaryConfig(normal_pause_seconds=0.3, long_phrase_pause_seconds=0.5)


def test_rejects_max_less_than_long_phrase_threshold():
    with pytest.raises(ValueError):
        PhraseBoundaryConfig(long_phrase_seconds=4.0, max_phrase_seconds=3.0)


def test_rejects_negative_phrase_duration():
    with pytest.raises(ValueError):
        detector().decide(-0.1, [])
