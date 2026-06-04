"""Text segmentation for caption display and translation."""

from __future__ import annotations

import re

_CLOSING_PUNCTUATION = "\"')]}»"


def split_caption_units(text: str) -> list[str]:
    """Split finalized ASR text into sentence-sized caption units.

    VAD decides when audio is ready to finalize. This function decides how many
    text lines to render and translate from that finalized ASR text. Keeping
    translation units short matters because Opus-MT drops content on long
    mixed-dialogue chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    units: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        end = _terminator_end(text, i)
        if end is None:
            i += 1
            continue

        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        unit = text[start:end].strip()
        if unit:
            units.append(unit)
        while end < len(text) and text[end].isspace():
            end += 1
        start = end
        i = end

    tail = text[start:].strip()
    if tail:
        units.append(tail)
    return units


def _terminator_end(text: str, index: int) -> int | None:
    if text.startswith("...", index):
        return index + 3

    char = text[index]
    if char in "?!…":
        return index + 1

    if char != ".":
        return None

    # Avoid splitting domains/abbreviations without whitespace after the period,
    # e.g. "podcastfrancaisfacile.com".
    next_index = index + 1
    if next_index == len(text) or text[next_index].isspace() or text[next_index] in _CLOSING_PUNCTUATION:
        return next_index
    return None
