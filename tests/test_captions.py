"""Deterministic tests for src/captions.py — the interim→committed state machine."""

from __future__ import annotations

import pytest

from src.captions import CaptionState, Line


def test_starts_empty():
    state = CaptionState()
    assert state.interim == ""
    assert state.committed == ()


def test_update_interim_is_live_and_trimmed():
    state = CaptionState()
    state.update_interim("  hello wor")
    assert state.interim == "hello wor"
    state.update_interim("hello world")  # replaces, not appends
    assert state.interim == "hello world"
    assert state.committed == ()  # nothing committed until a pause


def test_commit_promotes_interim_and_clears_it():
    state = CaptionState()
    state.update_interim("hello world")
    line = state.commit()
    assert line == Line(id=0, source="hello world", target=None)
    assert state.interim == ""
    assert state.committed == (line,)


def test_commit_empty_interim_returns_none_and_adds_nothing():
    state = CaptionState()
    assert state.commit() is None
    state.update_interim("   ")  # whitespace only
    assert state.commit() is None
    assert state.committed == ()


def test_commits_accumulate_in_order_with_unique_ids():
    state = CaptionState()
    state.update_interim("first")
    state.commit()
    state.update_interim("second")
    state.commit()
    sources = [(l.id, l.source) for l in state.committed]
    assert sources == [(0, "first"), (1, "second")]


def test_double_commit_does_not_duplicate():
    state = CaptionState()
    state.update_interim("only once")
    state.commit()
    assert state.commit() is None
    assert len(state.committed) == 1


def test_set_translation_attaches_by_id():
    state = CaptionState()
    state.update_interim("bonjour")
    line = state.commit()
    state.set_translation(line.id, "  hello ")
    assert state.committed[0].target == "hello"  # trimmed


def test_set_translation_lands_on_right_line_out_of_order():
    state = CaptionState()
    state.update_interim("un")
    first = state.commit()
    state.update_interim("deux")
    second = state.commit()
    # Second phrase's translation finishes first — still lands correctly.
    state.set_translation(second.id, "two")
    state.set_translation(first.id, "one")
    assert [l.target for l in state.committed] == ["one", "two"]


def test_set_translation_unknown_id_raises():
    state = CaptionState()
    with pytest.raises(KeyError):
        state.set_translation(99, "nope")


def test_committed_is_immutable_view():
    state = CaptionState()
    state.update_interim("x")
    state.commit()
    assert isinstance(state.committed, tuple)
