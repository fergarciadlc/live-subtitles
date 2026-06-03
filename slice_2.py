"""Vertical slice 2 (THROWAWAY) — audio.py + captions.py working together.

Mic -> VAD pause detection (audio.py) -> faster-whisper -> caption state machine
(captions.py). The first time the real modules are wired live, so you can watch the
interim->committed handoff happen at each pause. Delete once app.py (step 7) lands.

What to watch:
  - The "…interim" line (carriage-return, overwrites itself) is the live source text,
    re-transcribed every ~1s as you keep talking — it flickers and changes. Expected.
  - When you PAUSE (>= --pause of silence), audio.py reports the phrase ended,
    captions.py commits it, and it prints as a fixed numbered line + a translation.

Why it's structured this way: re-transcribing the whole phrase on CPU is expensive,
so we (a) drain the mic queue every loop to stay real-time, (b) only do work once per
~STEP seconds of audio, and (c) throttle the live preview. The pause commit always
runs a final clean transcription, then real Opus-MT translation (src/translate.py).
Build the MT models first: pixi run -e convert convert-mt

Adaptive endpointing: a single pause threshold can't serve both choppy and fluent
speech (too short fragments mid-clause; too long lets continuous speech balloon into
paragraph-sized, laggy commits). So we commit on a normal pause, OR at a *shorter*
gap once a phrase has run long, OR at a hard cap — whichever comes first. That keeps
phrases complete yet bounded.

Usage:  python slice_2.py [--lang en] [--pause 0.5] [--max 7] [--model small]
        [--no-interim]   (Ctrl+C to stop)
"""

import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from src.audio import SAMPLE_RATE, detect_speech, prepare
from src.captions import CaptionState
from src.translate import translate

CHANNELS = 1
COMPUTE_TYPE = "int8"  # CPU-only path
STEP_SECONDS = 0.4  # do VAD/ASR work once per this much accumulated audio
INTERIM_EVERY = 1.0  # re-transcribe the live preview at most this often (CPU budget)
SOFT_MAX_SECONDS = 4.0  # past this, a phrase is "long" — commit at the shorter gap
SHORT_PAUSE = 0.2  # the reduced pause used once a phrase is long (fluent speech)


def parse_args() -> tuple[str | None, float, float, str, bool]:
    lang = next((a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--lang")), None)
    pause = next((float(a.split("=")[-1]) for a in sys.argv[1:] if a.startswith("--pause")), 0.5)
    max_phrase = next((float(a.split("=")[-1]) for a in sys.argv[1:] if a.startswith("--max")), 7.0)
    model = next((a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--model")), "small")
    interim = "--no-interim" not in sys.argv[1:]
    return lang, pause, max_phrase, model, interim


def main() -> None:
    lang, pause_seconds, max_phrase_seconds, model_size, show_interim = parse_args()
    # Translation needs an explicit source; with --lang auto we assume EN source.
    src_lang = lang or "en"
    tgt_lang = "fr" if src_lang == "en" else "en"
    print(f"Loading faster-whisper '{model_size}' (CPU)...")
    model = WhisperModel(model_size, device="cpu", compute_type=COMPUTE_TYPE)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())  # mono channel

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language=lang, beam_size=1)
        return " ".join(s.text.strip() for s in segments)

    def drain_blocking() -> np.ndarray:
        """Block for the next mic buffer, then grab everything else already queued.

        Draining keeps us at the live edge of the audio instead of falling behind one
        slow transcription at a time.
        """
        parts = [audio_q.get()]
        try:
            while True:
                parts.append(audio_q.get_nowait())
        except queue.Empty:
            pass
        return np.concatenate(parts)

    state = CaptionState()
    phrase_buf = np.empty(0, dtype=np.float32)  # audio for the not-yet-committed phrase
    step_samples = int(SAMPLE_RATE * STEP_SECONDS)
    last_interim = 0.0

    def commit_and_print() -> None:
        line = state.commit()
        if line:
            state.set_translation(line.id, translate(line.source, src_lang, tgt_lang))
            committed = state.committed[-1]
            print(f"\r[{committed.id}] {committed.source:<72}")
            print(f"      -> {committed.target}")

    print(f"Listening: lang={lang or 'auto'}, pause={pause_seconds:.2f}s, "
          f"max={max_phrase_seconds:.1f}s. Ctrl+C to stop.\n")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", callback=callback):
            while True:
                # Mic is already 16k mono; route through prepare() to prove the path.
                phrase_buf = np.concatenate([phrase_buf, prepare(drain_blocking(), SAMPLE_RATE)])
                if len(phrase_buf) < step_samples:
                    continue  # not enough new audio yet to bother with VAD/ASR
                buf_seconds = len(phrase_buf) / SAMPLE_RATE

                segments = detect_speech(phrase_buf, SAMPLE_RATE)
                if not segments:
                    # Only silence so far — keep the buffer short so it doesn't pile up.
                    if buf_seconds > 1.0:
                        phrase_buf = phrase_buf[-SAMPLE_RATE // 2:]
                    continue

                # Live preview: re-transcribe the whole phrase, throttled to spare CPU.
                now = time.perf_counter()
                if show_interim and now - last_interim >= INTERIM_EVERY:
                    state.update_interim(transcribe(phrase_buf))
                    last_interim = now
                    print(f"\r  …{state.interim:<72}", end="", flush=True)

                # Adaptive endpointing: a long phrase commits at a shorter gap so
                # fluent speech doesn't balloon; the hard cap is the last resort.
                trailing_silence = buf_seconds - segments[-1].end
                required_pause = SHORT_PAUSE if buf_seconds >= SOFT_MAX_SECONDS else pause_seconds
                if trailing_silence >= required_pause or buf_seconds >= max_phrase_seconds:
                    state.update_interim(transcribe(phrase_buf))  # final, clean pass
                    commit_and_print()
                    phrase_buf = np.empty(0, dtype=np.float32)  # start the next phrase
                    last_interim = 0.0
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
