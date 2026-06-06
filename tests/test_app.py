"""Smoke + unit tests for the app shell (src/app.py).

These cover the pieces that don't need a browser or models: the pure caption renderer,
int16→float frame normalisation, control parsing, and that the Blocks UI actually wires
up (a build-time check that the FastRTC/Gradio plumbing is consistent).
"""

from __future__ import annotations

import numpy as np

import gradio as gr

from src.app import (
    CaptioningHandler,
    CSS,
    MIC_BUTTON_LABELS,
    _frames_to_audio,
    _push_outputs,
    _reset_view,
    build_ui,
    render_captions,
    render_status,
    replay_file,
)
from src.captions import Line


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


def test_frames_to_audio_normalises_int16():
    frame = np.array([[16384, -16384]], dtype=np.int16)  # shape (1, n), as WebRTC delivers
    out = _frames_to_audio([frame])
    assert out.dtype == np.float32
    assert out.shape == (2,)
    assert np.allclose(out, [0.5, -0.5], atol=1e-3)


def test_controls_prefer_latest_args_tail():
    handler = CaptioningHandler()
    # FastRTC prefixes latest_args with transport metadata; the controls are the last 5.
    handler.latest_args = ["__webrtc_value__", "webrtc-id", "en→fr", "tiny", 0.3, 5.0, 2]
    assert handler._controls() == ("en→fr", "tiny", 0.3, 5.0, 2)


def test_controls_support_old_four_value_tail():
    handler = CaptioningHandler()
    handler.latest_args = ["__webrtc_value__", "webrtc-id", "en→fr", "tiny", 0.3, 5.0]
    assert handler._controls() == ("en→fr", "tiny", 0.3, 5.0, 0)


def test_controls_fall_back_to_store_without_stream_args():
    from src import app

    handler = CaptioningHandler()  # fresh: no per-stream latest_args yet
    app._LIVE_CONTROLS.update(app.DEFAULTS)
    try:
        app._apply_controls("en→fr", "base", 0.4, 6.0)  # what a UI change event does
        assert handler._controls() == ("en→fr", "base", 0.4, 6.0, 0)
    finally:
        app._LIVE_CONTROLS.update(app.DEFAULTS)


def test_controls_validate_unexpected_values():
    handler = CaptioningHandler()
    handler.latest_args = ["x", "y", "??", "nope", "p", "q"]  # garbage in the tail
    assert handler._controls() == ("fr→en", "small", 0.5, 7.0, 0)


def test_push_outputs_routes_caption_and_status():
    assert _push_outputs("<caption>", "<status>") == ("<caption>", "<status>")


def test_reset_view_increments_token_and_clears_display():
    token, captions, status = _reset_view(3, "en→fr", "base", 0.4, 6.0)
    assert token == 4
    assert "Listening" in captions
    assert "reset" in status and "base" in status


def test_replay_file_without_audio_returns_status():
    captions, status = replay_file(None, "fr→en", "tiny", 0.5, 7.0)
    assert "Listening" in captions
    assert "No audio file selected" in status


def test_build_ui_constructs():
    demo = build_ui()
    assert isinstance(demo, gr.Blocks)


def test_mic_button_labels_are_explicit():
    assert MIC_BUTTON_LABELS["start"] == "Start listening"
    assert MIC_BUTTON_LABELS["stop"] == "Stop listening"
