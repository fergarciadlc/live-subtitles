#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
SPACE_TEMPLATE_DIR="$ROOT_DIR/hf_deploy/space"

UPLOAD_APP=1
UPLOAD_MODELS=1
RUN_TESTS=1
CONFIGURE_SPACE=1
PRIVATE_REPOS=0

usage() {
  cat <<'USAGE'
Usage: hf_deploy/deploy.sh [options]

Options:
  --app-only       Upload only the Space app files.
  --models-only    Upload only converted MT models.
  --skip-models    Create/configure repos and upload app, but skip model upload.
  --skip-tests     Do not run the test suite before uploading the app.
  --no-config      Do not set Space variables/secrets through the Hub API.
  --private        Create repos as private when they do not already exist.
  -h, --help       Show this help.

Required env (.env is sourced automatically if present):
  SPACE_REPO       e.g. fergarciadlc/live-subtitles
  MT_REPO          e.g. fergarciadlc/live-subtitles-mt

Optional env:
  HF_TOKEN         Used to set the Space HF_TOKEN secret for private model access
                   or authenticated Hub downloads.
                   If absent, the token from `hf auth login` is used when available.
  ENV_FILE         Alternate env file path.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --app-only)
      UPLOAD_MODELS=0
      ;;
    --models-only)
      UPLOAD_APP=0
      RUN_TESTS=0
      CONFIGURE_SPACE=0
      ;;
    --skip-models)
      UPLOAD_MODELS=0
      ;;
    --skip-tests)
      RUN_TESTS=0
      ;;
    --no-config)
      CONFIGURE_SPACE=0
      ;;
    --private)
      PRIVATE_REPOS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  # Auto-export so the hf CLI subprocess inherits HF_TOKEN etc., even when .env
  # uses plain `KEY=value` without an `export` prefix.
  set -a
  source "$ENV_FILE"
  set +a
fi

: "${SPACE_REPO:?Set SPACE_REPO in .env or the environment.}"
: "${MT_REPO:?Set MT_REPO in .env or the environment.}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
else
  echo "Missing required command: hf or huggingface-cli" >&2
  exit 1
fi

create_repo() {
  local repo_id="$1"
  shift
  if [[ "$PRIVATE_REPOS" -eq 1 ]]; then
    "${HF_CLI[@]}" repo create "$repo_id" "$@" --private
  else
    "${HF_CLI[@]}" repo create "$repo_id" "$@"
  fi
}

if [[ "$RUN_TESTS" -eq 1 ]]; then
  require_cmd pixi
  echo "==> Running tests"
  (cd "$ROOT_DIR" && pixi run -e test pytest tests)
fi

echo "==> Ensuring Hub repos exist"
create_repo "$SPACE_REPO" \
  --repo-type space \
  --space_sdk gradio \
  --exist-ok

create_repo "$MT_REPO" \
  --repo-type model \
  --exist-ok

if [[ "$UPLOAD_MODELS" -eq 1 ]]; then
  echo "==> Checking converted MT models"
  for model in opus-mt-fr-en opus-mt-en-fr; do
    if [[ ! -f "$ROOT_DIR/models/$model/model.bin" ]]; then
      echo "Missing required converted model: models/$model/model.bin" >&2
      echo "Build models with: pixi run -e convert convert-mt" >&2
      exit 1
    fi
  done

  echo "==> Uploading MT models to $MT_REPO"
  "${HF_CLI[@]}" upload "$MT_REPO" "$ROOT_DIR/models" . \
    --repo-type model \
    --include "opus-mt-*/**" \
    --commit-message "Upload converted Opus-MT models"

  echo "==> Uploading MT model card"
  "${HF_CLI[@]}" upload "$MT_REPO" "$ROOT_DIR/hf_deploy/mt_model_card.md" README.md \
    --repo-type model \
    --commit-message "Document converted Opus-MT sources"
fi

if [[ "$CONFIGURE_SPACE" -eq 1 ]]; then
  echo "==> Configuring Space variables"
  if command -v pixi >/dev/null 2>&1; then
    PYTHON_CMD=(pixi run python)
  else
    PYTHON_CMD=(python)
  fi
  "${PYTHON_CMD[@]}" - \
    "$SPACE_REPO" \
    "$MT_REPO" \
    "${HF_TOKEN:-}" <<'PY'
import sys
from huggingface_hub import HfApi, get_token

space_repo, mt_repo, hf_token = sys.argv[1:4]
auth_token = hf_token or get_token()
api = HfApi(token=auth_token or None)
api.add_space_variable(repo_id=space_repo, key="LIVE_SUBTITLES_MT_REPO", value=mt_repo)
if auth_token:
    api.add_space_secret(repo_id=space_repo, key="HF_TOKEN", value=auth_token)
    print("HF_TOKEN Space secret configured.")
else:
    print("HF_TOKEN not found; set it manually only if the Space needs authenticated Hub access.")
PY
fi

if [[ "$UPLOAD_APP" -eq 1 ]]; then
  echo "==> Staging Space app files"
  STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/live-subtitles-space.XXXXXX")"
  cleanup() {
    rm -rf "$STAGE_DIR"
  }
  trap cleanup EXIT

  cp "$SPACE_TEMPLATE_DIR/app.py" "$STAGE_DIR/"
  cp "$SPACE_TEMPLATE_DIR/README.md" "$STAGE_DIR/"
  cp "$SPACE_TEMPLATE_DIR/requirements.txt" "$STAGE_DIR/"
  cp "$SPACE_TEMPLATE_DIR/.hfignore" "$STAGE_DIR/"
  cp -R "$ROOT_DIR/src" "$STAGE_DIR/src"
  find "$STAGE_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +

  echo "==> Uploading Space app to $SPACE_REPO"
  "${HF_CLI[@]}" upload "$SPACE_REPO" "$STAGE_DIR" . \
    --repo-type space \
    --commit-message "Deploy live subtitles app"
fi

echo "==> Deploy complete"
echo "Space: https://huggingface.co/spaces/$SPACE_REPO"
