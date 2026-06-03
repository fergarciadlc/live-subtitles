"""Opus-MT translation — build-order step 6.

Turns a finalized source phrase into target-language text. Runs Helsinki-NLP Opus-MT
on the SAME CTranslate2/int8 backend faster-whisper uses — CPU-only, no torch at
runtime (torch is needed only to convert the weights; see convert_opus_mt.py). The
caption state machine calls this on each VAD pause, off the latency-critical path.

Per the plan this is deliberately one function, not an interface — it grows an
interface the day a second MT model appears. Models load lazily and are cached, so
the first translation in a direction pays the load cost and the rest are fast.

Models live in models/opus-mt-<src>-<tgt>/ (override the root with
LIVE_SUBTITLES_MODELS_DIR). Build them once with: pixi run -e convert convert-mt
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import ctranslate2
import transformers
from transformers import MarianTokenizer

# Opus-MT tokenizers are slow (sentencepiece) only; quiet the "PyTorch not found"
# notice — that's expected and correct here (runtime is torch-free by design).
transformers.utils.logging.set_verbosity_error()

COMPUTE_TYPE = "int8"  # matches the conversion quantization


def _models_root() -> Path:
    return Path(os.environ.get("LIVE_SUBTITLES_MODELS_DIR", "models"))


@lru_cache(maxsize=None)
def _engine(source_lang: str, target_lang: str) -> tuple[MarianTokenizer, ctranslate2.Translator]:
    """Load (and cache) the tokenizer + CTranslate2 translator for one direction."""
    model_dir = _models_root() / f"opus-mt-{source_lang}-{target_lang}"
    if not (model_dir / "model.bin").exists():
        raise FileNotFoundError(
            f"No converted Opus-MT model at {model_dir}. "
            f"Build it with: pixi run -e convert convert-mt"
        )
    tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
    translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type=COMPUTE_TYPE)
    return tokenizer, translator


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate `text` from source_lang to target_lang (e.g. "en" -> "fr").

    Empty/whitespace input returns ""; a same-language request returns the input
    unchanged. Otherwise runs Opus-MT and returns the decoded translation.
    """
    text = text.strip()
    if not text:
        return ""
    if source_lang == target_lang:
        return text

    tokenizer, translator = _engine(source_lang, target_lang)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    result = translator.translate_batch([tokens])
    output_ids = tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()
