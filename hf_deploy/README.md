# Hugging Face Deploy

This folder contains the repeatable deploy path for the two Hugging Face repos:

- Space repo: `fergarciadlc/live-subtitles` for app code only.
- MT repo: `fergarciadlc/live-subtitles-mt` for converted `models/opus-mt-*` CTranslate2 folders.

The Space-specific files live under `hf_deploy/space/` so the source repo root can
keep its normal project README:

- `hf_deploy/space/app.py` becomes `app.py` in the Space.
- `hf_deploy/space/README.md` becomes the Space README/model card.
- `hf_deploy/space/requirements.txt` becomes the Space runtime requirements.
- `hf_deploy/space/.hfignore` becomes the Space upload ignore file.

Set `.env` at the project root:

```bash
export HF_USER="fergarciadlc"
export SPACE_REPO="$HF_USER/live-subtitles"
export MT_REPO="$HF_USER/live-subtitles-mt"
# Optional, but useful for configuring the Space automatically:
# If omitted, deploy.sh will use the token from `hf auth login` when available.
# Useful when the MT repo is private or Hub downloads need authenticated access.
# export HF_TOKEN="hf_..."
```

Run:

```bash
hf auth login
hf_deploy/deploy.sh
```

The deploy uploads `hf_deploy/mt_model_card.md` as `README.md` in the MT repo, so
the converted artifacts keep references back to the original Helsinki-NLP models.

Useful variants:

```bash
hf_deploy/deploy.sh --app-only
hf_deploy/deploy.sh --models-only
hf_deploy/deploy.sh --skip-tests
```
