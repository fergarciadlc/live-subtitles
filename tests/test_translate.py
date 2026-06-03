"""Tests for src/translate.py.

The plumbing (empty input, same-language pass-through, missing-model error) is pure
and asserted exactly. The actual Opus-MT output is probabilistic — per the plan we
measure rather than assert exact strings, so the real-translation case is a smoke
test (non-empty, changed) that skips if the converted model isn't present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.translate import translate


# --- plumbing (no model needed) ---------------------------------------------


def test_empty_returns_empty():
    assert translate("", "en", "fr") == ""
    assert translate("   ", "en", "fr") == ""


def test_same_language_returns_input_unchanged():
    # Short-circuits before loading any model.
    assert translate("hello there", "en", "en") == "hello there"


def test_missing_model_raises_with_guidance():
    # No en-de model is ever converted, so this exercises the error path regardless
    # of which directions are present locally.
    with pytest.raises(FileNotFoundError, match="convert-mt"):
        translate("hello", "en", "de")


# --- real translation (smoke test; skipped if model absent) -----------------

_FR_EN = Path("models/opus-mt-fr-en")
_EN_FR = Path("models/opus-mt-en-fr")


@pytest.mark.skipif(not (_FR_EN / "model.bin").exists(), reason="fr-en model not built")
def test_translates_fr_to_en():
    out = translate("Bonjour, je suis canadien.", "fr", "en")
    assert isinstance(out, str) and out.strip()
    assert out != "Bonjour, je suis canadien."  # something actually changed


@pytest.mark.skipif(not (_EN_FR / "model.bin").exists(), reason="en-fr model not built")
def test_translates_en_to_fr():
    out = translate("Hello, I am Canadian.", "en", "fr")
    assert isinstance(out, str) and out.strip()
    assert out != "Hello, I am Canadian."
