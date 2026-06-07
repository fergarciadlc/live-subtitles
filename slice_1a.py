"""Vertical slice 1a (THROWAWAY) — build-order step 1a.

One audio file -> faster-whisper small -> printed text, timed.
Pure CPU-latency reality check on this machine. No mic, no VAD, no UI.
Delete once the real pipeline lands.

Usage:  python slice-1a [path/to/audio.wav] [--lang en]
"""

import sys
import time

from faster_whisper import WhisperModel

MODEL_SIZE = "small"
# CTranslate2 is CPU-only on macOS (no MPS). int8 is the fast CPU path.
COMPUTE_TYPE = "int8"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lang = next((a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--lang")), None)
    audio_path = args[0] if args else "clips/sample.wav"

    print(f"Loading faster-whisper '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE}, CPU)...")
    t0 = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s\n")

    print(f"Transcribing {audio_path}" + (f" (lang={lang})" if lang else " (autodetect)"))
    t0 = time.perf_counter()
    segments, info = model.transcribe(audio_path, language=lang, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)  # generator runs here
    infer_s = time.perf_counter() - t0

    audio_s = info.duration
    rtf = infer_s / audio_s if audio_s else float("nan")

    print(f"\n--- transcript ({info.language}, p={info.language_probability:.2f}) ---")
    print(text)
    print("\n--- timing ---")
    print(f"audio length:      {audio_s:.1f}s")
    print(f"inference time:    {infer_s:.1f}s")
    print(f"real-time factor:  {rtf:.2f}x  ({'keeps up live' if rtf < 1 else 'TOO SLOW for live'})")


if __name__ == "__main__":
    main()
