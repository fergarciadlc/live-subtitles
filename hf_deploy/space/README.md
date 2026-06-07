---
title: Live Subtitles
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
python_version: 3.12
---

# Live Subtitles

Browser-based live captioning with server-side ASR and after-pause translation.

## Runtime

- App entrypoint: `app.py`
- ASR: faster-whisper, CPU/int8
- Translation: converted Opus-MT CTranslate2 models in `models/opus-mt-<src>-<tgt>/`

## Space Configuration

Set these Space variables/secrets before relying on deployed translation:

- `LIVE_SUBTITLES_MT_REPO`: optional Hub repo containing converted MT folders.
- `HF_TOKEN`: optional Hub access token for private model repos or higher Hub rate
  limits during model downloads.

The base deploy expects `opus-mt-fr-en` and `opus-mt-en-fr`. Spanish target models
(`opus-mt-en-es`, `opus-mt-fr-es`) are optional and appear in the UI only when present.
