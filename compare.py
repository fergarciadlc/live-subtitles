"""Offline ASR model comparison — build-order step 3.

Loops each ASR adapter over every clip in clips/<lang>/*.wav (each with a
<name>.txt reference sidecar) and prints a table of:
  - WER  : word error rate vs the reference, per language (lower is better)
  - RTF  : real-time factor = compute-seconds / audio-seconds (under 1.0 keeps up live)

This is the probabilistic stage's measurement, NOT a pass/fail test. Never assert
exact transcripts here (see the plan): output varies by model and isn't reproducible.

The adapter list is hardcoded for now; it grows a config wiring when config.py lands
(step 5). Never imported by the app.

Usage:  pixi run compare            # or: python compare.py [--verbose]
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import jiwer
from faster_whisper.audio import decode_audio

from src.asr.faster_whisper import FasterWhisperASR

SAMPLE_RATE = 16000
CLIPS_DIR = Path("clips")

# The models under comparison. Add a new adapter here and it joins the table —
# nothing else changes (the whole point of the ASRModel contract).
ADAPTERS = [
    FasterWhisperASR("tiny"),
    FasterWhisperASR("small"),
]


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Keeps accented letters."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # \w keeps unicode word chars (é, à, ...)
    return re.sub(r"\s+", " ", text).strip()


def load_clips() -> list[tuple[str, Path, str]]:
    """Return (lang, wav_path, reference_text) for every clip with a sidecar."""
    clips = []
    for wav in sorted(CLIPS_DIR.glob("*/*.wav")):
        ref = wav.with_suffix(".txt")
        if not ref.exists():
            print(f"  (skipping {wav}: no .txt reference)", file=sys.stderr)
            continue
        clips.append((wav.parent.name, wav, ref.read_text().strip()))
    return clips


def main() -> None:
    verbose = "--verbose" in sys.argv[1:]
    clips = load_clips()
    if not clips:
        sys.exit(f"No clips found under {CLIPS_DIR}/<lang>/*.wav with .txt sidecars.")

    langs = sorted({lang for lang, _, _ in clips})
    print(f"{len(clips)} clips across {len(langs)} language(s): {', '.join(langs)}\n")

    rows = []
    for adapter in ADAPTERS:
        print(f"Loading {adapter.name}...", file=sys.stderr)
        adapter.load()

        # Collect normalized refs/hyps per language, plus timing for RTF.
        refs: dict[str, list[str]] = {lang: [] for lang in langs}
        hyps: dict[str, list[str]] = {lang: [] for lang in langs}
        total_infer_s = 0.0
        total_audio_s = 0.0

        for lang, wav, reference in clips:
            audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
            t0 = time.perf_counter()
            hypothesis = adapter.transcribe(audio, lang)
            total_infer_s += time.perf_counter() - t0
            total_audio_s += len(audio) / SAMPLE_RATE

            refs[lang].append(normalize(reference))
            hyps[lang].append(normalize(hypothesis))
            if verbose:
                print(f"  [{adapter.name} {lang} {wav.stem}] {hypothesis!r}", file=sys.stderr)

        wer = {lang: jiwer.wer(refs[lang], hyps[lang]) if refs[lang] else None for lang in langs}
        rtf = total_infer_s / total_audio_s if total_audio_s else float("nan")
        rows.append((adapter.name, wer, rtf))

    # --- table ---
    name_w = max(len(r[0]) for r in rows)
    header = f"{'model':<{name_w}}  " + "  ".join(f"WER:{lang:<4}" for lang in langs) + "   RTF"
    print("\n" + header)
    print("-" * len(header))
    for name, wer, rtf in rows:
        cells = "  ".join(
            (f"{wer[lang] * 100:6.1f}%" if wer[lang] is not None else f"{'-':>7}") for lang in langs
        )
        print(f"{name:<{name_w}}  {cells}   {rtf:.2f}x")
    print("\nWER lower is better; RTF under 1.0 keeps up live.")


if __name__ == "__main__":
    main()
