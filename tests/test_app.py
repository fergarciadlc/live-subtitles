"""Smoke and unit tests for the Gradio app shell."""

from __future__ import annotations

import numpy as np

import gradio as gr

from src.app import (
    CSS,
    DIRECTIONS,
    _drop_stream_state,
    _flush_stream,
    _normalise_audio_array,
    _normalise_stream_controls,
    _stream_audio,
    _available_directions,
    _reset_view,
    build_ui,
    render_captions,
    render_status,
    replay_file,
)
from src.captions import Line


def _touch_mt_model(root, src: str, tgt: str) -> None:
    model_dir = root / f"opus-mt-{src}-{tgt}"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").touch()


def test_render_empty_prompts_to_speak():
    html = render_captions("", ())
    assert "caption-board" in html and "Listening" in html


def test_render_shows_source_and_translation():
    html = render_captions("", (Line(id=0, source="Hello there", target="Bonjour"),))
    assert "Hello there" in html and "Bonjour" in html and "caption-board" in html


def test_caption_css_weights_translation_over_source():
    assert ".cap .src" in CSS and "font-weight: 400" in CSS
    assert ".cap .tgt" in CSS and "font-weight: 750" in CSS
    assert "body.dark .gradio-container" in CSS and "--ls-bg: #181513" in CSS


def test_directions_include_spanish_target_only():
    assert DIRECTIONS["en→es"] == ("en", "es")
    assert DIRECTIONS["fr→es"] == ("fr", "es")
    assert not any(src == "es" for src, _ in DIRECTIONS.values())


def test_available_directions_hide_missing_spanish_models(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_SUBTITLES_MODELS_DIR", str(tmp_path))
    _touch_mt_model(tmp_path, "fr", "en")
    _touch_mt_model(tmp_path, "en", "fr")

    choices = _available_directions()

    assert "fr→en" in choices and "en→fr" in choices
    assert "fr→es" not in choices and "en→es" not in choices


def test_available_directions_show_spanish_models_when_present(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_SUBTITLES_MODELS_DIR", str(tmp_path))
    for src, tgt in DIRECTIONS.values():
        _touch_mt_model(tmp_path, src, tgt)

    choices = _available_directions()

    assert "fr→es" in choices and "en→es" in choices


def test_render_orders_newest_first_for_pinned_scroll():
    html = render_captions(
        "current",
        (
            Line(id=0, source="older", target="plus ancien"),
            Line(id=1, source="newer", target="plus recent"),
        ),
    )
    assert html.index("current") < html.index("newer") < html.index("older")


def test_render_interim_is_marked_and_dimmed():
    html = render_captions("live words", (Line(id=0, source="done", target="fini"),))
    assert "live words" in html and "cap interim" in html and "fini" in html


def test_render_escapes_html():
    html = render_captions("", (Line(id=0, source="a<b>", target="x&y"),))
    assert "a&lt;b&gt;" in html and "x&amp;y" in html
    assert "<b>" not in html  # not injected raw


def test_render_pending_translation_placeholder():
    html = render_captions("", (Line(id=0, source="hi", target=None),))
    assert "translating" in html


def test_render_status_escapes_html():
    html = render_status("<ready>", direction="en→fr", model_size="tiny", detail="x<y")
    assert "&lt;ready&gt;" in html and "x&lt;y" in html
    assert "<ready>" not in html


def test_normalise_audio_array_normalises_int16():
    frame = np.array([[16384, -16384]], dtype=np.int16)
    out = _normalise_audio_array(frame)
    assert out.dtype == np.float32
    assert out.shape == frame.shape
    assert np.allclose(out, [[0.5, -0.5]], atol=1e-3)


def test_controls_validate_unexpected_values():
    assert _normalise_stream_controls("??", "nope", "p", "q") == ("fr→en", "small", 0.5, 7.0, 0)


def test_controls_fall_back_when_direction_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_SUBTITLES_MODELS_DIR", str(tmp_path))
    _touch_mt_model(tmp_path, "fr", "en")
    _touch_mt_model(tmp_path, "en", "fr")

    assert _normalise_stream_controls("fr→es", "tiny", 0.3, 5.0, 2) == ("fr→en", "tiny", 0.3, 5.0, 2)


def test_stream_audio_without_audio_returns_status():
    stream_id = "test-stream-empty"
    try:
        captions, status, returned_id = _stream_audio(None, stream_id, "fr→en", "tiny", 0.5, 7.0, 0)
    finally:
        _drop_stream_state(stream_id)
    assert returned_id == stream_id
    assert "Listening" in captions
    assert "Waiting for microphone audio" in status


def test_flush_stream_without_session_returns_status():
    stream_id = "test-stream-flush-empty"
    try:
        captions, status, returned_id = _flush_stream(stream_id, "fr→en", "tiny", 0.5, 7.0, 0)
    finally:
        _drop_stream_state(stream_id)
    assert returned_id == stream_id
    assert "Listening" in captions
    assert "Microphone stopped" in status


def test_reset_view_increments_token_and_clears_display():
    token, captions, status, stream_id = _reset_view(3, "en→fr", "base", 0.4, 6.0, "old-stream")
    assert token == 4
    assert stream_id != "old-stream"
    assert "Listening" in captions
    assert "reset" in status and "base" in status


def test_replay_file_without_audio_returns_status():
    captions, status = replay_file(None, "fr→en", "tiny", 0.5, 7.0)
    assert "Listening" in captions
    assert "No audio file selected" in status


def test_build_ui_constructs():
    demo = build_ui()
    assert isinstance(demo, gr.Blocks)
