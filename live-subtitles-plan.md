# Live subtitles — project plan

A browser-based live captioning app that transcribes English or French speech and shows a translation beneath it. Built to make the speech-to-text model swappable so different models can be compared on accuracy and latency.

## Architecture in one line

Browser captures mic audio → streams to server (FastRTC) → VAD gate → ASR (source-language text, live) → on each pause, translate the finalized phrase (Opus-MT) → display source + target captions stacked.

Inference runs server-side, not in the browser. Translation appears **on each VAD pause**, not continuously — so during long unbroken speech the target captions lag the source by design (this keeps the slow MT step off the latency-critical path).

**Deploy target:** Hugging Face Spaces free CPU first (simplest), with Modal's free tier as the fallback for the ASR stage if CPU latency proves unacceptable. Note: the deployable ASR model (faster-whisper / CTranslate2) is **CPU-only everywhere**, including on Apple Silicon — it never uses MPS. MPS-backed models (e.g. `mlx-whisper`) are useful for *fast local dev and comparison only*; being Apple-only, they cannot run on HF Spaces or Modal and are not deploy candidates.

## Guiding principles

- **One real abstraction up front: the ASR adapter.** Every speech-to-text model hides behind a single interface so the pipeline never knows which model it's using. Everything else stays concrete until it needs to flex.
- **Two kinds of tests, kept separate.** Deterministic plumbing gets pass/fail unit tests. ASR models get a measurement-based comparison (WER + latency), not pass/fail.
- **Config is for what varies between runs.** Model choice, chunk size, languages. Validated on load. Secrets never live in config.
- **Vertical slice before structure.** Prove faster-whisper-on-CPU works end to end before investing in the full module layout.
- **Add structure when it earns itself.** A folder is for multiple files; a second interface is for a second implementation. Until then, a flat file is clearer.

## Proposed structure

```
live-subtitles/
├── CLAUDE.md              # project conventions for Claude Code (see sketch below)
├── pyproject.toml
├── config/
│   └── default.yaml       # the one config that varies between runs
├── src/
│   ├── config.py          # load + validate (pydantic model lives here)
│   ├── audio.py           # resample to 16kHz mono + Silero VAD / pause detection
│   ├── captions.py        # caption state machine: interim vs committed
│   ├── translate.py       # Opus-MT, just a function
│   ├── asr/
│   │   ├── base.py        # the one real interface (the contract)
│   │   └── faster_whisper.py   # the first and only adapter for now
│   └── app.py             # Gradio UI + wiring
├── compare.py             # the whole model comparison — one script
├── clips/                 # test audio + reference transcripts
└── tests/
    ├── test_config.py
    ├── test_audio.py
    └── test_captions.py
```

Future ASR adapters (Whisper Turbo, Parakeet) are not files yet — they get added to `src/asr/` when the first adapter runs and you're ready to compare. The point of the interface is that adding them later is trivial.

## The ASR adapter interface

Every model implements the same contract. This is the only abstraction that earns its keep on day one, because swapping models is the whole point.

```python
class ASRModel(Protocol):
    name: str
    streams_natively: bool
    supported_languages: list[str]

    def load(self) -> None: ...
    def transcribe(self, audio_chunk, source_lang: str) -> str: ...
```

Each adapter absorbs that model's quirks (chunk-based vs streaming, sample-rate assumptions) so the pipeline and `compare.py` both stay clean. Anything proven in the comparison transfers to the app with no rework, because both call this same interface.

## Module responsibilities (separation of concerns)

- **audio.py** — get raw mic audio into the right shape (16kHz mono) and decide where phrases end. No model knowledge.
- **asr/** — turn audio into source-language text. Pluggable; the only folder, because it holds multiple models.
- **translate.py** — turn a finalized source phrase into target-language text. One function, one model. Becomes an interface the day a second MT model appears.
- **captions.py** — the interim-vs-committed state machine. The fork in the design: ASR output streams live to the source caption; the VAD pause-point commits a line and triggers translation. Keeps the slow MT step off the latency-critical path.
- **app.py** — UI and wiring only. No transcription logic lives here.
- **compare.py** — offline comparison of models. Never imported by the app.

## Testing strategy — two distinct things

**Unit tests (`tests/`)** — for deterministic plumbing only. Real pass/fail assertions:
- config loads and rejects a malformed file
- resample produces 16kHz mono of expected length
- VAD splits a known clip at the expected boundaries
- caption state machine promotes interim → committed correctly

**Model comparison (`compare.py`)** — for the probabilistic ASR stage. One script that loops over the adapters, runs each on the `clips/` folder, and prints a table of:
- **WER** against reference transcripts (EN and FR separately) — one call to the `jiwer` library, no separate metrics module needed
- **real-time factor** — compute-seconds per audio-second; under 1.0 means it keeps up live
- **behavior notes** — hallucination on silence, French quality, etc.

Do not write `assert transcription == "..."` tests for ASR. Output varies by model and isn't exactly reproducible; those tests are flaky and get deleted. Measure and compare instead.

The test set matters more than the comparison code. Source clips from Common Voice (Mozilla) for free EN/FR audio with reference transcripts; bias toward the audio you actually care about (accents, noise, fast speech).

## Config approach

One `default.yaml`, validated by a pydantic model that lives in `config.py`, so typos fail loudly at startup. Example shape:

```yaml
asr:
  model: faster_whisper_small   # the swap point
  chunk_seconds: 1.0            # the latency/accuracy knob
translate:
  model: opus_mt
languages:
  source: en
  target: fr
```

Secrets (HF tokens) come from environment variables, never the YAML. Don't configure things you'll never change.

## Build order

1. **Vertical slice (throwaway), in two parts.** Both disposable; deleted after.
   - **1a — file, no mic.** One audio file → faster-whisper small → printed text, timed. The pure CPU-latency reality check: "is the model fast enough on this M-series chip at all?" No audio plumbing, no UI. ~30 lines.
   - **1b — mic, minimal loop.** Mic (`sounddevice`/PortAudio, macOS prompts for permission once) → 1-second chunks → same model → rolling text in the terminal. Validates the *live* feel and chunking latency — still no VAD, no translation, no UI. ~50 lines. Keeping it separate from 1a isolates model speed from buffering/chunking lag.
2. **ASR adapter interface + faster-whisper adapter.** The contract first, one real implementation.
3. **compare.py + test clips.** Lock in WER and real-time-factor measurement against the one adapter.
4. **Add 1–2 more adapters** (Whisper Turbo, Parakeet, and `mlx-whisper` as the Apple-Silicon GPU/ANE comparison point — local dev only, not a deploy candidate) and run `compare.py` → pick a default *deployable* model and learn the latency budget. `mlx-whisper` slotting in with "nothing else changes" is the proof the ASR interface earns its keep.
5. **audio.py + captions.py with unit tests.** The deterministic core.
6. **translate.py (Opus-MT).** Wire in after-pause translation.
7. **Gradio app** around the winning adapter, with a dropdown to swap models live.

Each step is a small, independently verifiable target — a good unit of work for a single Claude Code session.

## CLAUDE.md sketch (keep it lean)

Claude Code reads this at the start of every session, so keep it short and high-signal — commands, layout, and "always do X" rules, not prose.

```markdown
# Live subtitles

Browser-based live captioning. EN/FR speech → transcription → translation.
Server-side inference. Two-stage pipeline: ASR (live) + translate (on pause).

## Commands
- Run tests:       pytest tests/
- Compare models:  python compare.py --config config/default.yaml
- Run app:         python -m src.app

## Layout
- src/asr/base.py is the ASR interface — all models implement it.
- compare.py is offline comparison; never imported by the app.
- Config is loaded and validated in src/config.py.

## Conventions
- New ASR models = new adapter in src/asr/ implementing ASRModel. Nothing else changes.
- Deterministic code gets pytest tests. ASR accuracy is measured in compare.py, not asserted.
- Secrets come from env vars, never config files.
- Make minimal changes — do not refactor unrelated code.
- Run pytest after a series of changes.
- One commit per logical unit (one module / one adapter).
```

A second short `CLAUDE.md` inside `src/asr/` describing the adapter contract is worth adding once you have a couple of adapters — Claude Code loads it lazily when working in that folder.
