"""Gradio UI and streaming transport for live subtitles."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from uuid import uuid4
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np

import gradio as gr

from src.asr.faster_whisper import FasterWhisperASR
from src.audio import SAMPLE_RATE, prepare
from src.captions import Line
from src.phrase_boundaries import PhraseBoundaryConfig
from src.session import LiveSession

logger = logging.getLogger(__name__)

MODEL_SIZES = ["tiny", "base", "small"]
DIRECTIONS = {
    "fr→en": ("fr", "en"),
    "en→fr": ("en", "fr"),
    "en→es": ("en", "es"),
    "fr→es": ("fr", "es"),
}
DEFAULTS = {"direction": "fr→en", "model": "small", "pause": 0.5, "max": 7.0, "reset": 0}
REQUIRED_DIRECTION_LABELS = ("fr→en", "en→fr")
LIVE_STREAM_EVERY_SECONDS = 0.75
LIVE_INTERIM_EVERY_SECONDS = 1.5
LIVE_INTERIM_MODEL = "tiny"


# --- shared model loading ---------------------------------------------------


@lru_cache(maxsize=None)
def get_asr(model_size: str) -> FasterWhisperASR:
    """Load and cache one ASR adapter per model size."""
    asr = FasterWhisperASR(model_size)
    asr.load()
    return asr


def ensure_mt_models(
    directions: tuple[tuple[str, str], ...] | None = None,
    required: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Download missing converted MT models when LIVE_SUBTITLES_MT_REPO is set."""
    if directions is None:
        directions = tuple(dict.fromkeys(DIRECTIONS.values()))
    if required is None:
        required = tuple(DIRECTIONS[label] for label in REQUIRED_DIRECTION_LABELS)
    required_set = set(required)
    repo = os.environ.get("LIVE_SUBTITLES_MT_REPO")
    if not repo:
        return
    from huggingface_hub import snapshot_download

    root = _mt_models_root()
    for src, tgt in directions:
        name = f"opus-mt-{src}-{tgt}"
        if not (root / name / "model.bin").exists():
            logger.info("downloading %s from %s", name, repo)
            try:
                snapshot_download(repo_id=repo, allow_patterns=f"{name}/*", local_dir=str(root))
            except Exception:
                if (src, tgt) in required_set:
                    raise
                logger.warning("optional MT model %s unavailable in %s; skipping", name, repo)
                continue
        if not (root / name / "model.bin").exists():
            message = f"No converted Opus-MT model at {root / name}"
            if (src, tgt) in required_set:
                raise FileNotFoundError(message)
            logger.warning("%s; optional direction will stay hidden", message)


def _mt_models_root() -> Path:
    return Path(os.environ.get("LIVE_SUBTITLES_MODELS_DIR", "models"))


def _mt_model_ready(source_lang: str, target_lang: str) -> bool:
    return (_mt_models_root() / f"opus-mt-{source_lang}-{target_lang}" / "model.bin").exists()


def _available_directions() -> dict[str, tuple[str, str]]:
    """Return directions whose converted MT model is present."""
    available = {label: pair for label, pair in DIRECTIONS.items() if _mt_model_ready(*pair)}
    if DEFAULTS["direction"] not in available:
        available[DEFAULTS["direction"]] = DIRECTIONS[DEFAULTS["direction"]]
    return available


def _normalise_direction(direction: str | None) -> str:
    available = _available_directions()
    if direction in available:
        return direction
    if DEFAULTS["direction"] in available:
        return DEFAULTS["direction"]
    return next(iter(available))


# --- rendering (pure) -------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1180px !important;
    --ls-bg: #fbfaf5;
    --ls-surface: #fffdf8;
    --ls-border: #e5ded2;
    --ls-text: #3f352c;
    --ls-muted: #8a7d6f;
    --ls-faint: #b4a99e;
    --ls-accent: #c96d36;
    --ls-interim-bg: #f6e7d8;
    --ls-interim-border: #efd8c2;
    --ls-mic-bg: #fffaf0;
}
body.dark .gradio-container,
html.dark .gradio-container,
.dark .gradio-container,
.gradio-container.dark,
[data-theme="dark"] .gradio-container {
    --ls-bg: #181513;
    --ls-surface: #221d19;
    --ls-border: #4a3d33;
    --ls-text: #f1e8dc;
    --ls-muted: #c3b3a3;
    --ls-faint: #998a7b;
    --ls-accent: #ffb076;
    --ls-interim-bg: #33261d;
    --ls-interim-border: #6e4a33;
    --ls-mic-bg: #251f1a;
}
.app-title h1 { margin-bottom: .1rem; }
.app-shell { align-items: stretch; }
.caption-pane { min-width: 0; }
.caption-pane .icon-button-wrapper,
.caption-pane .copy-button {
    display: none !important;
}
.control-rail {
    border: 1px solid var(--ls-border);
    border-radius: 8px;
    padding: 14px;
    background: var(--ls-surface);
}
.rail-heading {
    margin: 0 0 10px;
    font-weight: 700;
    color: var(--ls-text);
}
.caption-board {
    display: flex;
    flex-direction: column-reverse;
    gap: 0;
    min-height: 430px;
    max-height: 62vh;
    overflow-y: auto;
    padding: 18px;
    background: var(--ls-bg);
    border: 1px solid var(--ls-border);
    border-radius: 8px;
    color: var(--ls-text);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
}
.caption-board > * { flex: 0 0 auto; }
.caption-line {
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr);
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--ls-border);
}
.caption-line:last-child { border-bottom: 0; }
.caption-line.interim {
    grid-template-columns: 1fr;
    margin-bottom: 8px;
    border: 1px solid var(--ls-interim-border);
    border-radius: 8px;
    background: var(--ls-interim-bg);
    padding: 12px 14px;
}
.line-id {
    color: var(--ls-faint);
    font-variant-numeric: tabular-nums;
    font-size: .86rem;
    padding-top: 3px;
}
.caption-copy { min-width: 0; }
.cap .src {
    color: var(--ls-text);
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.45;
    overflow-wrap: anywhere;
}
.cap .tgt {
    color: var(--ls-accent);
    font-size: 1.12rem;
    font-weight: 750;
    line-height: 1.45;
    margin-top: 4px;
    overflow-wrap: anywhere;
}
.cap.interim .src {
    color: var(--ls-muted);
    font-style: italic;
}
.cap.interim .cursor {
    font-style: normal;
    color: var(--ls-accent);
    animation: blink 1s steps(1) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.pending {
    color: var(--ls-faint);
    font-style: italic;
    font-weight: 400;
}
.empty {
    min-height: 370px;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--ls-faint);
}
@media (max-width: 820px) {
    .caption-board { min-height: 340px; max-height: 54vh; }
}
.mic-compact {
    min-height: auto !important;
}
.mic-compact > div,
.mic-compact .wrap {
    min-height: 88px !important;
    border-color: var(--ls-border) !important;
    border-radius: 8px !important;
    background: var(--ls-mic-bg) !important;
    color: var(--ls-text) !important;
}
.mic-compact button {
    border-radius: 8px !important;
}
.status-card {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
    padding: 12px 14px;
    border: 1px solid var(--ls-border);
    border-radius: 8px;
    background: var(--ls-surface);
    color: var(--ls-text);
}
.status-item { min-width: 0; }
.status-label {
    display: block;
    color: var(--ls-muted);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.status-value {
    display: block;
    margin-top: 2px;
    overflow-wrap: anywhere;
    font-weight: 650;
}
.status-detail {
    grid-column: 1 / -1;
    color: var(--ls-muted);
    font-size: .9rem;
}
@media (max-width: 820px) {
    .status-card { grid-template-columns: 1fr; }
}
"""


def render_captions(
    interim: str,
    committed: tuple[Line, ...],
    *,
    max_lines: int = 50,
    direction: str = DEFAULTS["direction"],
    state: str = "ready",
) -> str:
    """Render committed and interim captions."""
    del direction, state
    rows: list[str] = []
    if interim:
        rows.append(
            "<div class='cap interim caption-line'>"
            f"<div class='src'>{escape(interim)} <span class='cursor'>|</span></div>"
            "<div class='tgt pending'>transcribing...</div>"
            "</div>"
        )
    for line in reversed(committed[-max_lines:]):
        target = escape(line.target) if line.target is not None else "<span class='pending'>translating...</span>"
        rows.append(
            "<div class='cap caption-line'>"
            f"<div class='line-id'>{line.id + 1:02d}</div>"
            "<div class='caption-copy'>"
            f"<div class='src'>{escape(line.source)}</div>"
            f"<div class='tgt'>{target}</div>"
            "</div>"
            "</div>"
        )
    body = "".join(rows) or "<div class='empty'>Listening... start speaking.</div>"
    return f"<div class='caption-board'>{body}</div>"


def render_status(
    state: str,
    *,
    direction: str = DEFAULTS["direction"],
    model_size: str = DEFAULTS["model"],
    committed: int = 0,
    detail: str = "Ready",
) -> str:
    """Render the stream status panel."""
    return (
        "<div class='status-card'>"
        "<div class='status-item'><span class='status-label'>State</span>"
        f"<span class='status-value'>{escape(state)}</span></div>"
        "<div class='status-item'><span class='status-label'>Direction</span>"
        f"<span class='status-value'>{escape(direction)}</span></div>"
        "<div class='status-item'><span class='status-label'>ASR</span>"
        f"<span class='status-value'>{escape(model_size)} · {committed} lines</span></div>"
        f"<div class='status-detail'>{escape(detail)}</div>"
        "</div>"
    )


def _reset_value(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(DEFAULTS["reset"])


def _normalise_audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.size == 0:
        return np.empty(0, dtype=np.float32)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return (array.astype(np.float32) / scale).astype(np.float32, copy=False)
    array = array.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array.astype(np.float32, copy=False)


def _audio_input_to_float32(audio: tuple[int, np.ndarray] | None) -> tuple[int, np.ndarray]:
    if audio is None:
        return SAMPLE_RATE, np.empty(0, dtype=np.float32)
    sample_rate, samples = audio
    return sample_rate, _normalise_audio_array(samples)


# --- transport adapter ------------------------------------------------------


@dataclass
class GradioStreamState:
    session: LiveSession | None = None
    applied: tuple[str, str, float, float] | None = None
    reset_token: int = DEFAULTS["reset"]
    chunks: int = 0
    audio_seconds: float = 0.0

    def reset_audio_stats(self) -> None:
        self.chunks = 0
        self.audio_seconds = 0.0


_STREAM_STATES: dict[str, GradioStreamState] = {}
_STREAM_LOCK = threading.Lock()


def _new_stream_id() -> str:
    return uuid4().hex


def _get_stream_state(stream_id: str | None) -> tuple[str, GradioStreamState]:
    stream_id = stream_id or _new_stream_id()
    with _STREAM_LOCK:
        state = _STREAM_STATES.setdefault(stream_id, GradioStreamState())
    return stream_id, state


def _drop_stream_state(stream_id: str | None) -> None:
    if stream_id:
        with _STREAM_LOCK:
            _STREAM_STATES.pop(stream_id, None)


def _interim_model_size(model_size: str) -> str:
    return model_size if model_size == LIVE_INTERIM_MODEL else LIVE_INTERIM_MODEL


def _audio_duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0 or samples.size == 0:
        return 0.0
    return float(samples.shape[0]) / float(sample_rate)


def _stream_detail(
    action: str,
    samples: np.ndarray,
    sample_rate: int,
    state: GradioStreamState,
) -> str:
    seconds = _audio_duration_seconds(samples, sample_rate)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return (
        f"{action} · chunk {seconds:.2f}s @ {sample_rate / 1000:.1f} kHz"
        f" · peak {peak:.2f} · {state.chunks} chunks / {state.audio_seconds:.1f}s"
    )


def _normalise_stream_controls(
    direction: str | None,
    model_size: str | None,
    pause: float | None,
    max_phrase: float | None,
    reset_token: int | None = None,
) -> tuple[str, str, float, float, int]:
    direction = _normalise_direction(direction)
    model_size = model_size if model_size in MODEL_SIZES else DEFAULTS["model"]
    try:
        pause_value = float(pause)
        max_phrase_value = float(max_phrase)
    except (TypeError, ValueError):
        pause_value = float(DEFAULTS["pause"])
        max_phrase_value = float(DEFAULTS["max"])
    return direction, model_size, pause_value, max_phrase_value, _reset_value(reset_token)


def _sync_stream_session(
    state: GradioStreamState,
    direction: str,
    model_size: str,
    pause: float,
    max_phrase: float,
    reset_token: int,
) -> LiveSession:
    if reset_token != state.reset_token:
        state.session = None
        state.applied = None
        state.reset_token = reset_token
        state.reset_audio_stats()
    if state.session is not None and state.applied == (direction, model_size, pause, max_phrase):
        return state.session

    src, tgt = DIRECTIONS[direction]
    boundary = PhraseBoundaryConfig(normal_pause_seconds=pause, max_phrase_seconds=max_phrase)
    interim_model_size = _interim_model_size(model_size)
    if state.session is None:
        state.session = LiveSession(
            get_asr(model_size),
            source_lang=src,
            target_lang=tgt,
            boundary=boundary,
            interim_asr=get_asr(interim_model_size),
            interim_every=LIVE_INTERIM_EVERY_SECONDS,
        )
    else:
        model_changed = state.applied is None or model_size != state.applied[1]
        state.session.reconfigure(
            source_lang=src,
            target_lang=tgt,
            boundary=boundary,
            asr=get_asr(model_size) if model_changed else None,
            interim_asr=get_asr(interim_model_size) if model_changed else None,
            interim_every=LIVE_INTERIM_EVERY_SECONDS,
        )
    state.applied = (direction, model_size, pause, max_phrase)
    return state.session


def _stream_audio(
    audio: tuple[int, np.ndarray] | None,
    stream_id: str | None,
    direction: str,
    model_size: str,
    pause: float,
    max_phrase: float,
    reset_token: int,
) -> tuple[str, str, str]:
    stream_id, state = _get_stream_state(stream_id)
    direction, model_size, pause, max_phrase, reset_token = _normalise_stream_controls(
        direction, model_size, pause, max_phrase, reset_token
    )
    sample_rate, samples = _audio_input_to_float32(audio)
    if samples.size == 0:
        return (
            render_captions(state.session.captions.interim, state.session.captions.committed, direction=direction)
            if state.session
            else render_captions("", (), direction=direction),
            render_status("listening", direction=direction, model_size=model_size, detail="Waiting for microphone audio."),
            stream_id,
        )

    state.chunks += 1
    state.audio_seconds += _audio_duration_seconds(samples, sample_rate)
    try:
        session = _sync_stream_session(state, direction, model_size, pause, max_phrase, reset_token)
        update = session.feed(samples, sample_rate)
    except FileNotFoundError as exc:
        logger.warning("required translation model unavailable; resetting stream: %s", exc)
        _drop_stream_state(stream_id)
        return (
            render_captions("", (), direction=direction, state="error"),
            render_status("missing model", direction=direction, model_size=model_size, detail=str(exc)),
            _new_stream_id(),
        )
    except Exception:
        logger.exception("pipeline error; dropping this streamed audio chunk")
        return (
            render_captions(
                state.session.captions.interim,
                state.session.captions.committed,
                direction=direction,
                state="error",
            )
            if state.session
            else render_captions("", (), direction=direction, state="error"),
            render_status(
                "error",
                direction=direction,
                model_size=model_size,
                detail=_stream_detail("Pipeline error; chunk dropped", samples, sample_rate, state),
            ),
            stream_id,
        )

    committed = session.captions.committed
    return (
        render_captions(session.captions.interim, committed, direction=direction, state="listening"),
        render_status(
            "listening",
            direction=direction,
            model_size=model_size,
            committed=len(committed),
            detail=_stream_detail("Updated" if update else "Listening", samples, sample_rate, state),
        ),
        stream_id,
    )


def _flush_stream(
    stream_id: str | None,
    direction: str,
    model_size: str,
    pause: float,
    max_phrase: float,
    reset_token: int,
) -> tuple[str, str, str]:
    stream_id, state = _get_stream_state(stream_id)
    direction, model_size, pause, max_phrase, reset_token = _normalise_stream_controls(
        direction, model_size, pause, max_phrase, reset_token
    )
    if state.session is None:
        return (
            render_captions("", (), direction=direction),
            render_status("stopped", direction=direction, model_size=model_size, detail="Microphone stopped."),
            stream_id,
        )
    try:
        session = _sync_stream_session(state, direction, model_size, pause, max_phrase, reset_token)
        flush_seconds = max(1.0, pause + 0.5)
        session.feed(np.zeros(int(SAMPLE_RATE * flush_seconds), dtype=np.float32), SAMPLE_RATE)
    except Exception:
        logger.exception("pipeline error while flushing stream")
    committed = state.session.captions.committed
    return (
        render_captions(state.session.captions.interim, committed, direction=direction, state="stopped"),
        render_status("stopped", direction=direction, model_size=model_size, committed=len(committed), detail="Microphone stopped."),
        stream_id,
    )


# --- UI ---------------------------------------------------------------------


def _reset_view(
    reset_token: int,
    direction: str,
    model: str,
    pause: float,
    max_phrase: float,
    stream_id: str | None = None,
) -> tuple[int, str, str, str]:
    direction, model, _, _, _ = _normalise_stream_controls(direction, model, pause, max_phrase)
    _drop_stream_state(stream_id)
    new_stream_id = _new_stream_id()
    new_token = _reset_value(reset_token) + 1
    return (
        new_token,
        render_captions("", (), direction=direction, state="ready"),
        render_status("reset", direction=direction, model_size=model, committed=0, detail="Caption history cleared."),
        new_stream_id,
    )


def replay_file(
    audio: tuple[int, np.ndarray] | None,
    direction: str,
    model_size: str,
    pause: float,
    max_phrase: float,
) -> tuple[str, str]:
    """Run an uploaded file through the captioning pipeline."""
    if audio is None:
        return (
            render_captions("", (), direction=direction, state="file"),
            render_status("file", direction=direction, model_size=model_size, detail="No audio file selected."),
        )

    direction, model_size, pause, max_phrase, _ = _normalise_stream_controls(
        direction, model_size, pause, max_phrase
    )

    sample_rate, samples = _audio_input_to_float32(audio)
    prepared = prepare(samples, sample_rate)
    if prepared.size == 0:
        return (
            render_captions("", (), direction=direction, state="file"),
            render_status("file", direction=direction, model_size=model_size, detail="Selected file had no audio samples."),
        )

    src, tgt = DIRECTIONS[direction]
    boundary = PhraseBoundaryConfig(normal_pause_seconds=pause, max_phrase_seconds=max_phrase)
    session = LiveSession(get_asr(model_size), source_lang=src, target_lang=tgt, boundary=boundary, interim_every=0.0)
    step_samples = int(SAMPLE_RATE * 0.4)
    try:
        for start in range(0, len(prepared), step_samples):
            session.feed(prepared[start : start + step_samples], SAMPLE_RATE)
        flush_seconds = max(1.0, pause + 0.5)
        session.feed(np.zeros(int(SAMPLE_RATE * flush_seconds), dtype=np.float32), SAMPLE_RATE)
    except FileNotFoundError as exc:
        logger.warning("file replay translation model unavailable: %s", exc)
        return (
            render_captions(session.captions.interim, session.captions.committed, direction=direction, state="file"),
            render_status("missing model", direction=direction, model_size=model_size, detail=str(exc)),
        )
    except Exception:
        logger.exception("file replay failed")
        return (
            render_captions(session.captions.interim, session.captions.committed, direction=direction, state="file"),
            render_status("file error", direction=direction, model_size=model_size, detail="File replay failed."),
        )

    duration = len(prepared) / SAMPLE_RATE
    committed = session.captions.committed
    return (
        render_captions(session.captions.interim, committed, direction=direction, state="file"),
        render_status(
            "file complete",
            direction=direction,
            model_size=model_size,
            committed=len(committed),
            detail=f"Processed {duration:.1f}s of uploaded audio.",
        ),
    )


def build_ui() -> gr.Blocks:
    available_directions = _available_directions()
    default_direction = _normalise_direction(DEFAULTS["direction"])
    with gr.Blocks(css=CSS, title="Live Subtitles") as demo:
        reset_token = gr.State(value=DEFAULTS["reset"])
        stream_id = gr.State(value=_new_stream_id, delete_callback=_drop_stream_state)
        gr.Markdown("# Live Subtitles", elem_classes=["app-title"])
        with gr.Row(equal_height=True, elem_classes=["app-shell"]):
            with gr.Column(scale=2, min_width=420, elem_classes=["caption-pane"]):
                captions = gr.HTML(
                    render_captions("", ()),
                    elem_id="captions-panel",
                    show_label=False,
                    padding=False,
                    autoscroll=True,
                )
                status = gr.HTML(
                    render_status("ready"),
                    elem_id="status-panel",
                    show_label=False,
                    padding=False,
                )
            with gr.Column(scale=1, min_width=320, elem_classes=["control-rail"]):
                gr.Markdown("### Live input", elem_classes=["rail-heading"])
                live_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label=None,
                    show_label=False,
                    show_download_button=False,
                    show_share_button=False,
                    editable=False,
                    waveform_options={"show_recording_waveform": False},
                    elem_classes=["mic-compact"],
                )
                direction = gr.Dropdown(list(available_directions), value=default_direction, label="Direction")
                model = gr.Dropdown(MODEL_SIZES, value=DEFAULTS["model"], label="ASR model")
                with gr.Accordion("Timing", open=True):
                    pause = gr.Slider(0.2, 1.5, value=DEFAULTS["pause"], step=0.05, label="Pause to commit (s)")
                    max_phrase = gr.Slider(4.0, 12.0, value=DEFAULTS["max"], step=0.5, label="Max phrase (s)")
                reset = gr.Button("Reset captions", variant="secondary")
                with gr.Accordion("File test", open=False):
                    file_audio = gr.Audio(label="Audio file", sources=["upload"], type="numpy")
                    run_file = gr.Button("Run file", variant="primary")

        control_inputs = [direction, model, pause, max_phrase, reset_token]

        reset.click(
            _reset_view,
            inputs=[reset_token, direction, model, pause, max_phrase, stream_id],
            outputs=[reset_token, captions, status, stream_id],
            queue=False,
            show_progress="hidden",
        )
        run_file.click(
            replay_file,
            inputs=[file_audio, direction, model, pause, max_phrase],
            outputs=[captions, status],
            concurrency_limit=1,
            show_progress="minimal",
        )
        live_audio.stream(
            _stream_audio,
            inputs=[live_audio, stream_id, *control_inputs],
            outputs=[captions, status, stream_id],
            stream_every=LIVE_STREAM_EVERY_SECONDS,
            time_limit=600,
            concurrency_limit=1,
            show_progress="hidden",
        )
        live_audio.stop_recording(
            _flush_stream,
            inputs=[stream_id, *control_inputs],
            outputs=[captions, status, stream_id],
            queue=True,
            concurrency_limit=1,
            show_progress="hidden",
        )
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_mt_models()
    get_asr(DEFAULTS["model"])  # warm the default so the first phrase isn't slow
    if _interim_model_size(DEFAULTS["model"]) != DEFAULTS["model"]:
        get_asr(_interim_model_size(DEFAULTS["model"]))
    default_server_name = "0.0.0.0" if os.environ.get("SYSTEM") == "spaces" else "127.0.0.1"
    launch_kwargs = {"server_name": os.environ.get("GRADIO_SERVER_NAME", default_server_name)}
    if port := os.environ.get("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = int(port)
    build_ui().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
