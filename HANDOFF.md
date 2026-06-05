# Live Subtitles — handoff

Snapshot of project state. Companion to `live-subtitles-plan.md` (the design) — this
doc is "where we actually are." Last updated: 2026-06-03.

## What this is

Browser-based live captioning: EN/FR speech → transcription (live) → translation
(on each pause). Server-side inference, swappable ASR behind one interface. Deploy
target is **HF Spaces free CPU** (linux-64); dev machine is osx-arm64. Everything
shipped runs **CPU-only, no torch at runtime**.

## Build-order progress

| Step | What | Status |
|------|------|--------|
| 1 | Vertical slices (file, mic) | ✅ `slice_1a.py`, `slice_1b.py` (throwaway) |
| 2 | ASR adapter interface + faster-whisper | ✅ `src/asr/base.py`, `src/asr/faster_whisper.py` |
| 3 | `compare.py` + test clips | ✅ WER + RTF table |
| 4 | More adapters → pick a default | ⏳ **gated on real clips** (see below) |
| 5 | `audio.py` + `captions.py` + unit tests | ✅ done this session |
| 6 | `translate.py` (Opus-MT) | ✅ done this session |
| 7 | Gradio app | ⏳ not started |

Build order forked at step 4 (needs real accented/noisy audio — on clean TTS every
model scores near-perfect, so the comparison can't pick a winner). We did 5 and 6
instead; step 4 is still open and waiting on real EN/FR clips in `clips/en` + `clips/fr`
(each `.wav` needs a matching `.txt`).

## Module map (current)

- `src/asr/base.py` — the `ASRModel` Protocol (the one real abstraction). Audio is
  always 1-D float32 mono 16 kHz.
- `src/asr/faster_whisper.py` — the only adapter so far (CTranslate2, int8, CPU).
- `src/audio.py` — **step 5.** Three layers: (1) shaping `to_mono`/`resample`/`prepare`
  (soxr), (2) `detect_speech` — thin wrapper over faster-whisper's bundled Silero VAD
  (onnxruntime, no torch), (3) `group_into_phrases`/`split_phrases` — pure pause-based
  phrasing. No model knowledge.
- `src/captions.py` — **step 5.** `CaptionState`: interim (live) vs committed lines.
  `update_interim` → `commit` (on pause, returns a `Line`) → `set_translation(id, …)`
  (id-keyed so out-of-order MT results land correctly).
- `src/translate.py` — **step 6.** `translate(text, src, tgt)`. Opus-MT on CTranslate2/
  int8. Lazy-loads + caches per direction. Models in `models/opus-mt-<src>-<tgt>/`.
- `compare.py` — offline WER/RTF comparison (step 3). Never imported by the app.
- `convert_opus_mt.py` — one-time Opus-MT → CTranslate2 conversion (build step).
- `slice_2.py` — **live demo** wiring audio + captions + translate together (throwaway,
  deleted once `app.py` lands). Has adaptive endpointing (see decisions).

## Commands

```
pixi run -e test test                       # 33 tests (audio, captions, translate)
pixi run compare                            # WER + RTF table over clips/
pixi run -e convert convert-mt              # build Opus-MT models (one-time, needs torch)
pixi run python slice_2.py --lang=fr --pause=0.5   # live end-to-end demo
```

`slice_2.py` flags: `--lang en|fr`, `--pause 0.5`, `--max 7`, `--model small|tiny`,
`--no-interim`.

## Key decisions (and why)

- **Resampling = soxr.** High-quality polyphase, tiny prebuilt wheels (linux-64 +
  osx-arm64). Beats numpy-linear (aliasing into the ASR feed) without scipy/torch bulk.
- **VAD = faster-whisper's bundled Silero (onnxruntime).** Real Silero, zero new heavy
  deps, no torch. The neural part is a thin wrapper; the *testable* logic is the pure
  `group_into_phrases` pause-grouping, kept separate so boundary tests are deterministic.
- **Translation = CTranslate2 + Opus-MT** (Helsinki-NLP), same backend as faster-whisper.
  Tokenizer is `MarianTokenizer` directly (NOT `AutoTokenizer` — its fast-conversion
  fails without a tokenizers backend). ~40–60 ms/phrase, ~0.4 s one-time load.
- **torch is build-time only.** Conversion needs it; runtime/deploy doesn't. torch lives
  in its own `convert` pixi env with a **separate solve group**, so it never enters the
  default/deploy lock. Runtime deps: ctranslate2, transformers (tokenizer only),
  sentencepiece, sacremoses.
- **Adaptive endpointing (in slice_2).** A single pause threshold can't serve both
  choppy and fluent speech. Commit on a 0.5 s pause, OR at a 0.2 s gap once a phrase
  exceeds 4 s, OR a hard 7 s cap. Keeps phrases complete yet bounded (low lag, small
  MT chunks). See "open items" — this should be lifted into a tested component for the app.

## Open items / next session

1. **Commit the uncommitted work.** Steps 5 + 6 + slice_2 changes are NOT committed yet
   (see git state below). Suggested logical commits: (a) step 5 audio+captions+tests,
   (b) step 6 translate+convert+models wiring.
2. **Step 7 (Gradio app).** When building it, **lift the adaptive endpointing out of
   slice_2** into a small tested `Endpointer` (streaming commit-timing — does NOT belong
   in `audio.py`'s offline `group_into_phrases`). Then delete the slices.
3. **Step 4 (more adapters + pick default).** Blocked on real EN/FR clips (accents, noise,
   fast speech). Add Whisper Turbo / Parakeet / mlx-whisper adapters, run `compare`.
4. **config.py + default.yaml** (pydantic) — planned but not built; `compare.py` still
   hardcodes its adapter list.
5. **Model shipping for deploy.** `models/` is gitignored (~75 MB/dir). Decide: upload to
   the HF Space, or convert in the Space build.

## Git state

- Branch: `main`. Last commit `3a5b649` = steps 2–3 scaffolding.
- **Uncommitted:** `src/audio.py`, `src/captions.py`, `src/translate.py`, `tests/`,
  `convert_opus_mt.py`, `live-subtitles-plan.md`, and modified `slice_2.py`, `.gitignore`,
  `pyproject.toml`, `pixi.lock`.
- `models/` and audio files (`*.wav`/`*.mp3`) are gitignored; clip `.txt` references are
  tracked.
