"""Gradio + FastRTC app — build-order step 7. UI and wiring only.

No transcription logic lives here (that is src/session.py's LiveSession). This module
is the transport + UI shell:

  browser mic --WebRTC--> CaptioningHandler.receive() --queue--> worker thread
       worker thread: LiveSession.feed() --> render_captions() --queue-->
  CaptioningHandler.emit() returns AdditionalOutputs --> caption + status displays

We use FastRTC's low-level StreamHandler (not ReplyOnPause): ReplyOnPause runs its own
VAD and only yields audio after a pause, which would kill the live interim caption and
make our adaptive endpointer + the pause/max sliders meaningless. So FastRTC is pure
transport; LiveSession owns VAD, endpointing, ASR and translation.

The heavy per-chunk work runs on a worker thread so the WebRTC receive path is never
blocked — the streaming-era version of slice_2's "drain the queue to stay at the live
edge". ASR weights are shared across sessions (get_asr, lru-cached by size); each
connection gets its own LiveSession via StreamHandler.copy().

Two FastRTC gotchas this file handles explicitly (both were "the app looks broken" bugs):
  - The WebRTC widget defaults to a fullscreen-capable mode that can overlay the page and
    hide the controls/record button — we pass full_screen=False (+ a CSS backstop).
  - Control values arrive in latest_args with transport metadata PREFIXED, so the real
    controls are the TAIL (latest_args[-5:]), not [1:].

Run locally:  pixi run app      (opens http://127.0.0.1:7860)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np
from fastrtc import AdditionalOutputs, StreamHandler, WebRTC

import gradio as gr

from src.asr.faster_whisper import FasterWhisperASR
from src.audio import SAMPLE_RATE, prepare, to_mono
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
MIC_BUTTON_LABELS = {
    "start": "Start listening",
    "stop": "Stop listening",
    "waiting": "Connecting...",
}

# Fallback control store, written by the UI change events. The authoritative per-connection
# values come through the stream inputs (latest_args); this store covers non-stream contexts
# (and tests). Process-global — fine for the single-presenter use this demo targets.
_LIVE_CONTROLS: dict = dict(DEFAULTS)


def _apply_controls(direction: str, model: str, pause: float, max_phrase: float, reset_token: int = 0) -> None:
    """UI change-event sink: remember the latest control values (fallback path)."""
    direction = _normalise_direction(direction)
    _LIVE_CONTROLS.update(direction=direction, model=model, pause=pause, max=max_phrase, reset=reset_token)


# --- shared model loading ---------------------------------------------------


@lru_cache(maxsize=None)
def get_asr(model_size: str) -> FasterWhisperASR:
    """Load (and cache) one ASR adapter per model size, shared across all sessions.

    CTranslate2 models are safe to run concurrently, so sessions share the weights
    read-only — only the per-session LiveSession state differs.
    """
    asr = FasterWhisperASR(model_size)
    asr.load()
    return asr


def ensure_mt_models(directions: tuple[tuple[str, str], ...] | None = None) -> None:
    """Pull the Opus-MT models from the Hub when running on a fresh deploy.

    Local dev already has them (built once via `pixi run -e convert convert-mt`), so
    this is a no-op unless LIVE_SUBTITLES_MT_REPO points at a Hub repo holding the
    converted CTranslate2 dirs — the chosen deploy strategy: ship nothing in the repo,
    download on first boot like faster-whisper already does for the ASR weights.
    """
    if directions is None:
        directions = tuple(dict.fromkeys(DIRECTIONS.values()))
    repo = os.environ.get("LIVE_SUBTITLES_MT_REPO")
    if not repo:
        return
    from huggingface_hub import snapshot_download

    root = _mt_models_root()
    for src, tgt in directions:
        name = f"opus-mt-{src}-{tgt}"
        if not (root / name / "model.bin").exists():
            logger.info("downloading %s from %s", name, repo)
            snapshot_download(repo_id=repo, allow_patterns=f"{name}/*", local_dir=str(root))


def _mt_models_root() -> Path:
    return Path(os.environ.get("LIVE_SUBTITLES_MODELS_DIR", "models"))


def _mt_model_ready(source_lang: str, target_lang: str) -> bool:
    return (_mt_models_root() / f"opus-mt-{source_lang}-{target_lang}" / "model.bin").exists()


def _available_directions() -> dict[str, tuple[str, str]]:
    """Configured directions whose converted MT model is present.

    Keep the default visible even in a half-configured checkout so the UI has a sane
    fallback and any missing-model error can be reported in the status panel.
    """
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


def _rtc_configuration():
    """No TURN locally; use the Space's HF community TURN server when deployed."""
    if os.environ.get("SYSTEM") == "spaces":
        try:
            from fastrtc import get_hf_turn_credentials

            return get_hf_turn_credentials()
        except Exception:  # pragma: no cover - deploy-only path
            logger.warning("HF TURN credentials unavailable; falling back to none")
    return None


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
.mic-compact [title="grant webcam access"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    min-height: 70px !important;
    border: 1px solid var(--ls-border) !important;
    border-radius: 8px !important;
    background: var(--ls-mic-bg) !important;
    color: var(--ls-text) !important;
}
.mic-compact [title="grant webcam access"]::after {
    content: "Enable microphone";
    pointer-events: none;
    font-weight: 650;
}
.mic-compact .gradio-webrtc-waveContainer,
.mic-compact .wave-container,
.mic-compact .wave-svg,
.mic-compact .standard-player {
    display: none !important;
}
.mic-compact .audio-container {
    min-height: 70px !important;
    height: 70px !important;
    justify-content: center !important;
}
.mic-compact .button-wrap {
    margin: 6px auto !important;
    padding: 8px 12px !important;
    box-shadow: none !important;
}
.mic-compact .icon-with-text {
    min-width: auto !important;
    margin: 0 6px !important;
    gap: 8px !important;
}
.mic-compact button[aria-label="select input source"],
.mic-compact .source-selection,
.mic-compact .select-wrap {
    display: none !important;
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

/* Backstop for full_screen=False: if the FastRTC audio widget still toggles a
   full-screen class, keep it inline so the controls/record button stay visible. */
.audio-container.full-screen,
.gradio-webrtc-waveContainer.full-screen,
.wave-container.full-screen,
.wave-svg.full-screen,
.button-wrap.full-screen {
    position: relative !important;
    top: auto !important;
    left: auto !important;
    width: 100% !important;
    height: auto !important;
    max-height: 280px !important;
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
    """Build the caption scrollback: committed lines plus the in-progress interim line."""
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
    """Small status panel that makes the live system easier to follow while testing."""
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


def _frames_to_audio(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate WebRTC frames into 1-D float32. WebRTC delivers int16; normalise it."""
    parts = []
    for arr in frames:
        parts.append(to_mono(_normalise_audio_array(arr)))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _audio_input_to_float32(audio: tuple[int, np.ndarray] | None) -> tuple[int, np.ndarray]:
    if audio is None:
        return SAMPLE_RATE, np.empty(0, dtype=np.float32)
    sample_rate, samples = audio
    return sample_rate, _normalise_audio_array(samples)


# --- transport adapter ------------------------------------------------------


class CaptioningHandler(StreamHandler):
    """FastRTC transport glue around a per-connection LiveSession.

    receive() just enqueues frames (must return fast); a worker thread drains them,
    runs the pipeline, and queues rendered caption HTML that emit() hands back as an
    AdditionalOutputs. Control values arrive per-connection via self.latest_args.
    """

    def __init__(self) -> None:
        # Ask FastRTC to deliver mono 16 kHz so prepare() is a near no-op; we still
        # pass the frame's own rate to feed() and resample defensively.
        super().__init__("mono", input_sample_rate=SAMPLE_RATE)
        self._in: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._out: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._session: LiveSession | None = None
        self._applied: tuple | None = None
        self._reset_token = int(DEFAULTS["reset"])

    def copy(self) -> "CaptioningHandler":
        return CaptioningHandler()

    def start_up(self) -> None:
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self._in.put(frame)

    def emit(self):
        try:
            return AdditionalOutputs(*self._out.get_nowait())
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        self._stop.set()

    def _controls(self) -> tuple[str, str, float, float, int]:
        # Prefer per-connection values from FastRTC latest_args. StreamHandler.set_args
        # prefixes transport metadata (webrtc id / sentinel), so the controls are the TAIL.
        values = getattr(self, "latest_args", None)
        if isinstance(values, list) and len(values) >= 5 and values[-5] in DIRECTIONS:
            raw_direction, raw_model, raw_pause, raw_max, raw_reset = values[-5:]
            c = {"direction": raw_direction, "model": raw_model, "pause": raw_pause, "max": raw_max}
            reset_token = _reset_value(raw_reset)
        elif isinstance(values, list) and len(values) >= 4 and values[-4] in DIRECTIONS:
            raw_direction, raw_model, raw_pause, raw_max = values[-4:]
            c = {"direction": raw_direction, "model": raw_model, "pause": raw_pause, "max": raw_max}
            reset_token = _reset_value(_LIVE_CONTROLS.get("reset"))
        else:
            # Fallback for non-stream contexts/tests where latest_args is absent.
            c = _LIVE_CONTROLS
            reset_token = _reset_value(c.get("reset"))
        d = DEFAULTS
        direction = _normalise_direction(c.get("direction"))
        model = c.get("model") if c.get("model") in MODEL_SIZES else d["model"]
        try:
            pause, max_phrase = float(c.get("pause")), float(c.get("max"))
        except (TypeError, ValueError):
            pause, max_phrase = d["pause"], d["max"]
        return direction, model, pause, max_phrase, reset_token

    def _sync_session(self) -> tuple[str, str, float, float, int]:
        """Create the session, or apply any UI control change since the last frame."""
        direction, model_size, pause, max_phrase, reset_token = self._controls()
        if reset_token != self._reset_token:
            self._session = None
            self._applied = None
            self._reset_token = reset_token
        if self._session is not None and (direction, model_size, pause, max_phrase) == self._applied:
            return direction, model_size, pause, max_phrase, reset_token
        src, tgt = DIRECTIONS[direction]
        boundary = PhraseBoundaryConfig(normal_pause_seconds=pause, max_phrase_seconds=max_phrase)
        if self._session is None:
            self._session = LiveSession(get_asr(model_size), source_lang=src, target_lang=tgt, boundary=boundary)
        else:
            model_changed = model_size != self._applied[1] if self._applied else True
            self._session.reconfigure(
                source_lang=src,
                target_lang=tgt,
                boundary=boundary,
                asr=get_asr(model_size) if model_changed else None,
            )
        self._applied = (direction, model_size, pause, max_phrase)
        return direction, model_size, pause, max_phrase, reset_token

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample_rate, first = self._in.get(timeout=0.1)
            except queue.Empty:
                continue
            frames = [first]
            while True:  # drain whatever else queued up, to stay at the live edge
                try:
                    frames.append(self._in.get_nowait()[1])
                except queue.Empty:
                    break
            direction = DEFAULTS["direction"]
            model_size = DEFAULTS["model"]
            try:
                direction, model_size, *_ = self._sync_session()
                update = self._session.feed(_frames_to_audio(frames), sample_rate)
            except FileNotFoundError as exc:
                logger.warning("required translation model unavailable; resetting session: %s", exc)
                self._session = None
                self._applied = None
                self._out.put(
                    (
                        render_captions("", (), direction=direction, state="error"),
                        render_status("missing model", direction=direction, model_size=model_size, detail=str(exc)),
                    )
                )
                continue
            except Exception:
                logger.exception("pipeline error; dropping this chunk")
                self._out.put(
                    (
                        render_captions("", (), state="error"),
                        render_status("error", detail="Pipeline error; this chunk was dropped."),
                    )
                )
                continue
            if update is not None:
                self._out.put(
                    (
                        render_captions(update.interim, update.committed, direction=direction, state="listening"),
                        render_status(
                            "listening",
                            direction=direction,
                            model_size=model_size,
                            committed=len(update.committed),
                            detail=f"Updated from {len(frames)} audio frame(s).",
                        ),
                    )
                )


# --- UI ---------------------------------------------------------------------


def _push_outputs(captions_html: str, status_html: str) -> tuple[str, str]:
    """on_additional_outputs callback: route handler HTML to the visible panels."""
    return captions_html, status_html


def _reset_view(
    reset_token: int,
    direction: str,
    model: str,
    pause: float,
    max_phrase: float,
) -> tuple[int, str, str]:
    direction = _normalise_direction(direction)
    new_token = _reset_value(reset_token) + 1
    _apply_controls(direction, model, pause, max_phrase, new_token)
    return (
        new_token,
        render_captions("", (), direction=direction, state="ready"),
        render_status("reset", direction=direction, model_size=model, committed=0, detail="Caption history cleared."),
    )


def replay_file(
    audio: tuple[int, np.ndarray] | None,
    direction: str,
    model_size: str,
    pause: float,
    max_phrase: float,
) -> tuple[str, str]:
    """Run an uploaded file through the same LiveSession core as the live stream."""
    if audio is None:
        return (
            render_captions("", (), direction=direction, state="file"),
            render_status("file", direction=direction, model_size=model_size, detail="No audio file selected."),
        )

    direction = _normalise_direction(direction)
    if model_size not in MODEL_SIZES:
        model_size = DEFAULTS["model"]
    try:
        pause = float(pause)
        max_phrase = float(max_phrase)
    except (TypeError, ValueError):
        pause = float(DEFAULTS["pause"])
        max_phrase = float(DEFAULTS["max"])

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
                webrtc = WebRTC(
                    modality="audio",
                    mode="send-receive",
                    rtc_configuration=_rtc_configuration(),
                    label=None,
                    show_label=False,
                    height=88,
                    elem_classes=["mic-compact"],
                    button_labels=MIC_BUTTON_LABELS,
                    full_screen=False,
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

        # Keep a shared fallback store via change events, and also pass controls through
        # the stream input path so each connection gets authoritative per-stream values.
        control_inputs = [direction, model, pause, max_phrase, reset_token]
        control_components = [direction, model, pause, max_phrase]
        for component in control_components:
            component.change(_apply_controls, inputs=control_inputs, outputs=None, queue=False, show_progress="hidden")

        reset.click(
            _reset_view,
            inputs=[reset_token, direction, model, pause, max_phrase],
            outputs=[reset_token, captions, status],
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
        webrtc.stream(fn=CaptioningHandler(), inputs=[webrtc, *control_inputs], outputs=[webrtc])
        webrtc.on_additional_outputs(_push_outputs, outputs=[captions, status], queue=False, show_progress="hidden")
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_mt_models()
    get_asr(DEFAULTS["model"])  # warm the default so the first phrase isn't slow
    launch_kwargs = {"server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")}
    if port := os.environ.get("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = int(port)
    build_ui().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
