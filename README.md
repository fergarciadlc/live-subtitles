# Live Subtitles

Live Subtitles is a local-first Gradio app for browser microphone captioning:
speech is transcribed with faster-whisper, segmented after pauses, and translated
with converted Opus-MT CTranslate2 models.

The app currently targets French/English live translation, with English/Spanish
and French/Spanish available when the optional converted models are present.

## Project Layout

- `src/app.py`: Gradio UI and microphone streaming transport.
- `src/session.py`: per-session audio-to-caption pipeline.
- `src/audio.py`: audio shaping, VAD integration, and phrase grouping helpers.
- `src/translate.py`: CTranslate2 Opus-MT translation adapter.
- `tests/`: deterministic unit and smoke tests.
- `hf_deploy/`: Hugging Face deploy scripts, Space templates, and model-card files.
- `mockup/` and `tmp/`: design/prototype scratch space, not part of deploy.

## Local Development

Install and run through Pixi:

```bash
pixi run app
```

Then open:

```text
http://127.0.0.1:7860
```

Run tests:

```bash
pixi run -e test pytest tests
```

## Translation Models

Converted MT models are expected under:

```text
models/opus-mt-<source>-<target>/
```

The required base directions are:

- `opus-mt-fr-en`
- `opus-mt-en-fr`

Optional Spanish target directions:

- `opus-mt-en-es`
- `opus-mt-fr-es`

Build converted models with:

```bash
pixi run -e convert convert-mt
```

## Hugging Face Deploy

The deploy artifacts for the Space are kept separate from the source repo root:

- `hf_deploy/space/app.py`
- `hf_deploy/space/README.md`
- `hf_deploy/space/requirements.txt`
- `hf_deploy/space/.hfignore`

Deploy with:

```bash
hf_deploy/deploy.sh
```

For app-only updates after the model repo is already populated:

```bash
hf_deploy/deploy.sh --app-only
```

See [hf_deploy/README.md](hf_deploy/README.md) for repository setup and deploy
variants.
