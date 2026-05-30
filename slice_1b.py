"""Vertical slice 1b (THROWAWAY) — build-order step 1b.

Mic -> fixed N-second chunks -> faster-whisper small -> rolling terminal text.
Validates the *live feel* and chunking latency. Still NO VAD, NO translation, NO UI.
Delete once the real pipeline lands.

Expect choppy text and words split at chunk boundaries: each chunk is transcribed
in isolation with no context carried across. That is exactly the problem VAD +
the caption state machine solve later (steps 5). Here we only care about: does the
mic work, does macOS grant permission, and can we process a chunk faster than it
takes to record one (RTF < 1)?

Usage:  python slice-1b [--chunk 1.0] [--lang en]   (Ctrl+C to stop)
"""

import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # what faster-whisper expects
CHANNELS = 1
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"  # CPU-only path on macOS


def parse_args() -> tuple[float, str | None]:
    chunk = next((float(a.split("=")[-1]) for a in sys.argv[1:] if a.startswith("--chunk")), 1.0)
    lang = next((a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--lang")), None)
    return chunk, lang


def main() -> None:
    chunk_seconds, lang = parse_args()
    chunk_samples = int(SAMPLE_RATE * chunk_seconds)

    print(f"Loading faster-whisper '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE}, CPU)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

    # Mic callback runs on a separate thread; just hand frames to a queue.
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())  # mono channel

    print(f"Listening: {chunk_seconds:.1f}s chunks, lang={lang or 'auto'}. Ctrl+C to stop.\n")
    buf = np.empty(0, dtype=np.float32)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", callback=callback):
            while True:
                buf = np.concatenate([buf, audio_q.get()])
                if len(buf) < chunk_samples:
                    continue
                chunk, buf = buf[:chunk_samples], buf[chunk_samples:]

                t0 = time.perf_counter()
                segments, _ = model.transcribe(chunk, language=lang, beam_size=1)
                text = " ".join(s.text.strip() for s in segments)
                proc_s = time.perf_counter() - t0

                rtf = proc_s / chunk_seconds
                flag = "" if rtf < 1 else "  <-- SLOWER THAN REALTIME"
                print(f"[{proc_s:4.2f}s | {rtf:4.2f}x{flag}] {text or '...'}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
