"""Convert Opus-MT models to CTranslate2 format — a one-time build step.

translate.py runs Opus-MT on the same CTranslate2/int8 backend faster-whisper uses,
which needs the weights converted from their Hugging Face (PyTorch) form first. This
script does that once per direction, writing into models/. Runs in the `convert` env
(the only place torch lives — see pyproject.toml); runtime/deploy never imports torch.

Usage:  pixi run -e convert convert-mt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path("models")

# The Helsinki-NLP Opus-MT models for the project's directions. Add a pair here and
# re-run; translate.py resolves the same models/opus-mt-<src>-<tgt> directories.
DIRECTIONS = [
    ("en", "fr"),
    ("fr", "en"),
]


def convert(src: str, tgt: str) -> None:
    hf_model = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    out_dir = MODELS_DIR / f"opus-mt-{src}-{tgt}"
    if (out_dir / "model.bin").exists():
        print(f"  {out_dir} already converted — skipping.")
        return
    print(f"Converting {hf_model} -> {out_dir} (int8)...")
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", hf_model,
            "--output_dir", str(out_dir),
            "--quantization", "int8",
            # Bundle the tokenizer assets so the model dir is self-contained at
            # runtime (no Hugging Face download, no torch) — what deploys.
            "--copy_files", "source.spm", "target.spm", "vocab.json", "tokenizer_config.json",
        ],
        check=True,
    )


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for src, tgt in DIRECTIONS:
        convert(src, tgt)
    print(f"\nDone. {len(DIRECTIONS)} direction(s) in {MODELS_DIR}/.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Conversion failed (exit {e.returncode}). Is the `convert` env active? "
                 f"Run: pixi run -e convert convert-mt")
