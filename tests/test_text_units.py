"""Deterministic tests for caption text segmentation."""

from __future__ import annotations

from src.text_units import split_caption_units


def test_split_mixed_dialogue_sentences():
    text = (
        "Bonjour, je peux vous aider ? Oui, je cherche une cravate pour mon mari. "
        "Une cravate ? Alors suivez-moi. Voilà."
    )
    assert split_caption_units(text) == [
        "Bonjour, je peux vous aider ?",
        "Oui, je cherche une cravate pour mon mari.",
        "Une cravate ?",
        "Alors suivez-moi.",
        "Voilà.",
    ]


def test_collapses_whitespace():
    assert split_caption_units("  Bonjour,\n je peux vous aider ?   Oui. ") == [
        "Bonjour, je peux vous aider ?",
        "Oui.",
    ]


def test_keeps_unsentenced_text_as_one_unit():
    assert split_caption_units("travail il est ingenieur en agronomie") == [
        "travail il est ingenieur en agronomie"
    ]


def test_splits_ellipsis():
    assert split_caption_units("Oui… je cherche une cravate... Voilà.") == [
        "Oui…",
        "je cherche une cravate...",
        "Voilà.",
    ]


def test_does_not_split_domain_periods():
    assert split_caption_units("podcastfrancaisfacile.com Bonjour.") == [
        "podcastfrancaisfacile.com Bonjour."
    ]


def test_empty_text_returns_empty_list():
    assert split_caption_units("   ") == []
