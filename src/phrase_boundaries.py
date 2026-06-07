"""Streaming phrase-boundary decisions for live captions.

`audio.py` detects speech spans. This module decides whether the current live
phrase has reached a boundary and should be committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.audio import Segment

BoundaryReason = Literal["no_speech", "waiting", "pause", "max_phrase"]


@dataclass(frozen=True)
class PhraseBoundaryConfig:
    """Timing policy for committing live speech into caption phrases."""

    normal_pause_seconds: float = 0.5
    long_phrase_seconds: float = 4.0
    long_phrase_pause_seconds: float = 0.2
    max_phrase_seconds: float = 7.0

    def __post_init__(self) -> None:
        values = {
            "normal_pause_seconds": self.normal_pause_seconds,
            "long_phrase_seconds": self.long_phrase_seconds,
            "long_phrase_pause_seconds": self.long_phrase_pause_seconds,
            "max_phrase_seconds": self.max_phrase_seconds,
        }
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.long_phrase_pause_seconds > self.normal_pause_seconds:
            raise ValueError("long_phrase_pause_seconds must be <= normal_pause_seconds")
        if self.max_phrase_seconds < self.long_phrase_seconds:
            raise ValueError("max_phrase_seconds must be >= long_phrase_seconds")


@dataclass(frozen=True)
class PhraseBoundaryDecision:
    """Result of evaluating the current phrase buffer."""

    should_commit: bool
    reason: BoundaryReason
    phrase_seconds: float
    trailing_silence: float
    required_pause_seconds: float


class PhraseBoundaryDetector:
    """Decide when a live phrase is complete enough to commit."""

    def __init__(self, config: PhraseBoundaryConfig = PhraseBoundaryConfig()) -> None:
        self.config = config

    def decide(
        self,
        phrase_seconds: float,
        speech_segments: Sequence[Segment],
    ) -> PhraseBoundaryDecision:
        """Return whether the current phrase buffer should be committed."""
        if phrase_seconds < 0:
            raise ValueError(f"phrase_seconds must be non-negative, got {phrase_seconds}")

        required_pause = self._required_pause(phrase_seconds)
        if not speech_segments:
            return PhraseBoundaryDecision(
                should_commit=False,
                reason="no_speech",
                phrase_seconds=phrase_seconds,
                trailing_silence=phrase_seconds,
                required_pause_seconds=required_pause,
            )

        trailing_silence = max(0.0, phrase_seconds - speech_segments[-1].end)
        if phrase_seconds >= self.config.max_phrase_seconds:
            return PhraseBoundaryDecision(
                should_commit=True,
                reason="max_phrase",
                phrase_seconds=phrase_seconds,
                trailing_silence=trailing_silence,
                required_pause_seconds=required_pause,
            )

        should_commit = trailing_silence >= required_pause
        return PhraseBoundaryDecision(
            should_commit=should_commit,
            reason="pause" if should_commit else "waiting",
            phrase_seconds=phrase_seconds,
            trailing_silence=trailing_silence,
            required_pause_seconds=required_pause,
        )

    def _required_pause(self, phrase_seconds: float) -> float:
        if phrase_seconds >= self.config.long_phrase_seconds:
            return self.config.long_phrase_pause_seconds
        return self.config.normal_pause_seconds
