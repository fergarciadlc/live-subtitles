"""Offline smoke test for the live caption pipeline on one audio file.

This replays a file through the same core path as `slice_2.py`: VAD, phrase
boundary detection, final ASR, sentence-sized caption units, and optional
translation. It is for tuning live behavior against a known transcript without
manual microphone playback.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from src.audio import SAMPLE_RATE, detect_speech, prepare
from src.phrase_boundaries import PhraseBoundaryConfig, PhraseBoundaryDetector
from src.text_units import split_caption_units
from src.translate import translate


@dataclass(frozen=True)
class CaptionRow:
    source: str
    target: str | None
    reason: str
    phrase_seconds: float
    trailing_silence: float
    required_pause: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one file through the live caption pipeline.")
    parser.add_argument("--audio", type=Path, required=True, help="Audio file to replay.")
    parser.add_argument("--reference", type=Path, help="Optional transcript for WER/MER/WIL.")
    parser.add_argument("--lang", default="fr", choices=["en", "fr"], help="Source language.")
    parser.add_argument("--target", choices=["en", "fr"], help="Target language. Defaults to the opposite of --lang.")
    parser.add_argument("--model", default="small", help="faster-whisper model size.")
    parser.add_argument("--pause", type=float, default=0.35, help="Normal pause threshold in seconds.")
    parser.add_argument("--long", type=float, default=4.0, help="Phrase duration that enables shorter pause threshold.")
    parser.add_argument("--short-pause", type=float, default=0.2, help="Pause threshold once phrase is long.")
    parser.add_argument("--max", dest="max_phrase", type=float, default=7.0, help="Hard phrase cap in seconds.")
    parser.add_argument("--step", type=float, default=0.4, help="Simulated live processing step in seconds.")
    parser.add_argument("--beam", type=int, default=1, help="faster-whisper beam size for final ASR.")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation output.")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"podcastfrancaisfacile\s*com", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_reference(path: Path) -> str:
    """Load the exercise-style transcript body without labels/watermark."""
    lines: list[str] = []
    for index, raw_line in enumerate(path.read_text().splitlines()):
        line = raw_line.strip()
        if not line or line == "---":
            continue
        lowered = line.lower()
        if index == 0 or "podcastfrancaisfacile.com" in lowered:
            continue
        line = re.sub(r"^(?:Le vendeur|La cliente)\s*:\s*", "", line)
        if line:
            lines.append(line)
    return " ".join(lines)


def transcribe(model: WhisperModel, audio: np.ndarray, source_lang: str, beam_size: int) -> str:
    segments, _ = model.transcribe(audio, language=source_lang, beam_size=beam_size)
    return " ".join(segment.text.strip() for segment in segments)


def simulate(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    source_lang: str,
    target_lang: str,
    config: PhraseBoundaryConfig,
    step_seconds: float,
    beam_size: int,
    should_translate: bool,
) -> tuple[list[CaptionRow], float]:
    detector = PhraseBoundaryDetector(config)
    phrase_buf = np.empty(0, dtype=np.float32)
    step_samples = int(SAMPLE_RATE * step_seconds)
    rows: list[CaptionRow] = []
    total_asr_seconds = 0.0

    for start in range(0, len(audio), step_samples):
        chunk = audio[start : start + step_samples]
        phrase_buf = np.concatenate([phrase_buf, prepare(chunk, SAMPLE_RATE)])
        if len(phrase_buf) < step_samples:
            continue

        buf_seconds = len(phrase_buf) / SAMPLE_RATE
        segments = detect_speech(phrase_buf, SAMPLE_RATE)
        if not segments:
            if buf_seconds > 1.0:
                phrase_buf = phrase_buf[-SAMPLE_RATE // 2 :]
            continue

        decision = detector.decide(buf_seconds, segments)
        if not decision.should_commit:
            continue

        t0 = time.perf_counter()
        finalized = transcribe(model, phrase_buf, source_lang, beam_size)
        total_asr_seconds += time.perf_counter() - t0
        rows.extend(
            rows_for_text(
                finalized,
                source_lang=source_lang,
                target_lang=target_lang,
                should_translate=should_translate,
                reason=decision.reason,
                phrase_seconds=decision.phrase_seconds,
                trailing_silence=decision.trailing_silence,
                required_pause=decision.required_pause_seconds,
            )
        )
        phrase_buf = np.empty(0, dtype=np.float32)

    if len(phrase_buf) > step_samples:
        segments = detect_speech(phrase_buf, SAMPLE_RATE)
        if segments:
            t0 = time.perf_counter()
            finalized = transcribe(model, phrase_buf, source_lang, beam_size)
            total_asr_seconds += time.perf_counter() - t0
            rows.extend(
                rows_for_text(
                    finalized,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    should_translate=should_translate,
                    reason="end_of_file",
                    phrase_seconds=len(phrase_buf) / SAMPLE_RATE,
                    trailing_silence=0.0,
                    required_pause=0.0,
                )
            )
    return rows, total_asr_seconds


def rows_for_text(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
    should_translate: bool,
    reason: str,
    phrase_seconds: float,
    trailing_silence: float,
    required_pause: float,
) -> list[CaptionRow]:
    rows: list[CaptionRow] = []
    for unit in split_caption_units(text):
        target = translate(unit, source_lang, target_lang) if should_translate else None
        rows.append(CaptionRow(unit, target, reason, phrase_seconds, trailing_silence, required_pause))
    return rows


def print_metrics(rows: list[CaptionRow], reference_path: Path | None) -> None:
    if reference_path is None:
        return
    reference = normalize(load_reference(reference_path))
    hypothesis = normalize(" ".join(row.source for row in rows))
    print(f"WER: {jiwer.wer(reference, hypothesis) * 100:.1f}%")
    print(f"MER: {jiwer.mer(reference, hypothesis) * 100:.1f}%")
    print(f"WIL: {jiwer.wil(reference, hypothesis) * 100:.1f}%")
    print(f"Words: reference={len(reference.split())} hypothesis={len(hypothesis.split())}")


def main() -> None:
    args = parse_args()
    target_lang = args.target or ("fr" if args.lang == "en" else "en")
    audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    config = PhraseBoundaryConfig(
        normal_pause_seconds=args.pause,
        long_phrase_seconds=args.long,
        long_phrase_pause_seconds=args.short_pause,
        max_phrase_seconds=args.max_phrase,
    )

    print(f"Audio: {args.audio}")
    print(
        f"Replay: lang={args.lang} target={target_lang} model={args.model} "
        f"pause={args.pause:.2f}s max={args.max_phrase:.1f}s duration={duration:.2f}s"
    )
    print(f"Loading faster-whisper '{args.model}' (CPU)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    t0 = time.perf_counter()
    rows, total_asr_seconds = simulate(
        model,
        audio,
        source_lang=args.lang,
        target_lang=target_lang,
        config=config,
        step_seconds=args.step,
        beam_size=args.beam,
        should_translate=not args.no_translate,
    )
    elapsed = time.perf_counter() - t0

    print()
    print_metrics(rows, args.reference)
    print(f"Lines: {len(rows)}")
    print(f"ASR RTF: {total_asr_seconds / duration:.2f}x")
    print(f"Wall time: {elapsed:.2f}s")
    print()

    for index, row in enumerate(rows):
        print(
            f"[{index:02d}] {row.reason} dur={row.phrase_seconds:.1f}s "
            f"trail={row.trailing_silence:.2f}/{row.required_pause:.2f}"
        )
        print(f"{args.lang.upper()}: {row.source}")
        if row.target is not None:
            print(f"{target_lang.upper()}: {row.target}")
        print()


if __name__ == "__main__":
    main()
