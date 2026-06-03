"""Audio plumbing — build-order step 5 (the deterministic core).

Get raw input audio into the shape the ASR contract expects (1-D float32, mono,
16 kHz; see asr/base.py) and decide where phrases end. NO model knowledge lives
here — this module never imports an ASR adapter.

Three layers, deliberately separated so the deterministic bits stay unit-testable:
  1. shaping   — to_mono / resample / prepare: pure array math, fully tested.
  2. VAD       — detect_speech: a thin wrapper over faster-whisper's bundled
                 Silero VAD (onnxruntime, CPU-only — the same backend that deploys
                 to HF Spaces). Neural, so it is exercised by a real-clip
                 integration test, not asserted against synthetic signals.
  3. phrasing  — group_into_phrases: pure logic that turns speech regions into
                 phrases by a pause threshold. This is where the "splits a clip at
                 the expected boundaries" tests live, decoupled from the model.

soxr is the resampler (high-quality polyphase, tiny prebuilt-wheel footprint —
the right trade for a CPU deploy). Linear interpolation would alias the signal we
feed to ASR; scipy/torch would bloat the image for no quality gain here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soxr
from faster_whisper.vad import VadOptions, get_speech_timestamps

SAMPLE_RATE = 16000  # what the mic, the file decoder, and the ASR contract all use


# --- 1. shaping -------------------------------------------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse to a contiguous 1-D float32 mono signal.

    Accepts 1-D audio (returned as-is) or 2-D multi-channel audio. For 2-D, the
    channel axis is taken to be the *minor* one (fewer entries) — true for any
    real recording, where samples vastly outnumber channels — and averaged out.
    """
    a = np.asarray(audio)
    if a.ndim == 1:
        mono = a
    elif a.ndim == 2:
        channel_axis = 0 if a.shape[0] < a.shape[1] else 1
        mono = a.mean(axis=channel_axis)
    else:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {a.shape}")
    return np.ascontiguousarray(mono, dtype=np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample 1-D audio from orig_sr to target_sr. No-op when rates match."""
    if orig_sr <= 0:
        raise ValueError(f"orig_sr must be positive, got {orig_sr}")
    a = np.ascontiguousarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return a
    return soxr.resample(a, orig_sr, target_sr).astype(np.float32, copy=False)


def prepare(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """One call from raw input to ASR-ready audio: mono, then resampled to target."""
    return resample(to_mono(audio), orig_sr, target_sr)


# --- 2. VAD (neural; thin wrapper) ------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A span of audio in seconds. Used for both raw speech regions and phrases."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
) -> list[Segment]:
    """Find speech regions via faster-whisper's bundled Silero VAD.

    Returns segments in *seconds* (the model works in samples; we convert so the
    downstream phrasing logic is sample-rate-independent). `min_silence_ms` only
    controls when the VAD itself ends a region — phrase boundaries are decided
    separately by group_into_phrases, which is the testable part.
    """
    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    raw = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    return [Segment(d["start"] / sample_rate, d["end"] / sample_rate) for d in raw]


# --- 3. phrasing (pure logic) -----------------------------------------------


def group_into_phrases(segments: list[Segment], pause_seconds: float) -> list[Segment]:
    """Merge speech segments into phrases, splitting on pauses.

    Consecutive segments separated by a gap shorter than `pause_seconds` belong to
    the same phrase and are merged into one span. A gap of `pause_seconds` or more
    is a phrase boundary — the point where the app commits a line and translates it.

    Assumes `segments` is sorted and non-overlapping (as VAD output is). Pure and
    deterministic: no audio, no model — this is the unit under test for boundaries.
    """
    if not segments:
        return []
    phrases: list[Segment] = []
    start, end = segments[0].start, segments[0].end
    for seg in segments[1:]:
        if seg.start - end < pause_seconds:
            end = max(end, seg.end)  # same phrase: extend (max guards overlaps)
        else:
            phrases.append(Segment(start, end))  # pause: close the phrase
            start, end = seg.start, seg.end
    phrases.append(Segment(start, end))
    return phrases


def split_phrases(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    pause_seconds: float = 0.5,
    **vad_kwargs,
) -> list[Segment]:
    """Convenience: detect speech, then group it into phrases. (VAD + phrasing.)"""
    return group_into_phrases(detect_speech(audio, sample_rate, **vad_kwargs), pause_seconds)
