"""Gradio app for browser-based live subtitles."""

from __future__ import annotations

import html
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import gradio as gr
import numpy as np

from src.asr.faster_whisper import FasterWhisperASR
from src.audio import SAMPLE_RATE, detect_speech, prepare
from src.captions import CaptionState, Line
from src.phrase_boundaries import PhraseBoundaryConfig, PhraseBoundaryDetector
from src.text_units import split_caption_units
from src.translate import translate

STREAM_EVERY_SECONDS = 0.5
INTERIM_EVERY_SECONDS = 1.0
MIN_SILENCE_KEEP_SECONDS = 0.5
APP_THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
APP_CSS = """
.gradio-container { max-width: 1180px !important; }
.caption-board {
  min-height: 420px;
  padding: 18px;
  background: #101418;
  border: 1px solid #2b333b;
  border-radius: 8px;
  color: #f4f6f8;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.caption-line {
  display: grid;
  grid-template-columns: 44px 1fr;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #2b333b;
}
.caption-line:last-child { border-bottom: 0; }
.caption-line.interim {
  grid-template-columns: 1fr;
  opacity: 0.72;
}
.line-id {
  color: #94a3b8;
  font-variant-numeric: tabular-nums;
  font-size: 0.86rem;
  padding-top: 3px;
}
.source {
  color: #f8fafc;
  font-size: 1.1rem;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.target {
  color: #9ccfd8;
  font-size: 1rem;
  line-height: 1.45;
  margin-top: 4px;
  overflow-wrap: anywhere;
}
.empty-state {
  display: flex;
  min-height: 360px;
  align-items: center;
  justify-content: center;
  color: #94a3b8;
}
"""


@dataclass(frozen=True)
class AppConfig:
    source_lang: str
    target_lang: str
    model_size: str
    beam_size: int
    normal_pause_seconds: float
    long_phrase_seconds: float
    long_phrase_pause_seconds: float
    max_phrase_seconds: float
    input_gain: float
    vad_threshold: float
    should_translate: bool
    show_interim: bool


@dataclass
class LiveSession:
    config: AppConfig
    captions: CaptionState = field(default_factory=CaptionState)
    phrase_buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    last_interim_seconds: float = 0.0
    chunks_seen: int = 0
    committed_phrases: int = 0


def _config(
    source_lang: str,
    target_lang: str,
    model_size: str,
    beam_size: int,
    normal_pause_seconds: float,
    long_phrase_seconds: float,
    long_phrase_pause_seconds: float,
    max_phrase_seconds: float,
    input_gain: float,
    vad_threshold: float,
    should_translate: bool,
    show_interim: bool,
) -> AppConfig:
    return AppConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        model_size=model_size,
        beam_size=int(beam_size),
        normal_pause_seconds=float(normal_pause_seconds),
        long_phrase_seconds=float(long_phrase_seconds),
        long_phrase_pause_seconds=float(long_phrase_pause_seconds),
        max_phrase_seconds=float(max_phrase_seconds),
        input_gain=float(input_gain),
        vad_threshold=float(vad_threshold),
        should_translate=bool(should_translate),
        show_interim=bool(show_interim),
    )


@lru_cache(maxsize=4)
def _asr(model_size: str, beam_size: int) -> FasterWhisperASR:
    model = FasterWhisperASR(model_size=model_size, beam_size=beam_size)
    model.load()
    return model


def _audio_chunk_to_float32(
    audio: tuple[int, np.ndarray] | None,
    input_gain: float = 1.0,
) -> tuple[int, np.ndarray]:
    if audio is None:
        return SAMPLE_RATE, np.empty(0, dtype=np.float32)

    sample_rate, samples = audio
    array = np.asarray(samples)
    if array.size == 0:
        return sample_rate, np.empty(0, dtype=np.float32)

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / max_value
    else:
        array = array.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if peak > 1.0:
            array = array / peak

    prepared = prepare(array, sample_rate)
    if input_gain != 1.0:
        prepared = np.clip(prepared * input_gain, -1.0, 1.0).astype(np.float32, copy=False)
    return sample_rate, prepared


def _boundary_config(config: AppConfig) -> PhraseBoundaryConfig:
    return PhraseBoundaryConfig(
        normal_pause_seconds=config.normal_pause_seconds,
        long_phrase_seconds=config.long_phrase_seconds,
        long_phrase_pause_seconds=config.long_phrase_pause_seconds,
        max_phrase_seconds=config.max_phrase_seconds,
    )


def _session(session: LiveSession | None, config: AppConfig) -> LiveSession:
    if session is None or session.config != config:
        return LiveSession(config=config)
    return session


def _transcribe(config: AppConfig, audio: np.ndarray) -> str:
    return _asr(config.model_size, config.beam_size).transcribe(audio, config.source_lang)


def _commit_text(session: LiveSession, text: str) -> None:
    for unit in split_caption_units(text):
        session.captions.update_interim(unit)
        line = session.captions.commit()
        if line is None:
            continue
        if session.config.should_translate:
            session.captions.set_translation(
                line.id,
                translate(line.source, session.config.source_lang, session.config.target_lang),
            )
        session.committed_phrases += 1


def _process_chunk(session: LiveSession, chunk: np.ndarray, *, update_interim: bool) -> str:
    if chunk.size == 0:
        return "waiting"

    session.chunks_seen += 1
    session.phrase_buffer = np.concatenate([session.phrase_buffer, chunk])
    phrase_seconds = len(session.phrase_buffer) / SAMPLE_RATE

    segments = detect_speech(
        session.phrase_buffer,
        SAMPLE_RATE,
        threshold=session.config.vad_threshold,
    )
    if not segments:
        if phrase_seconds > 1.0:
            keep = int(SAMPLE_RATE * MIN_SILENCE_KEEP_SECONDS)
            session.phrase_buffer = session.phrase_buffer[-keep:]
        return "silence"

    now = time.perf_counter()
    if update_interim and session.config.show_interim and now - session.last_interim_seconds >= INTERIM_EVERY_SECONDS:
        session.captions.update_interim(_transcribe(session.config, session.phrase_buffer))
        session.last_interim_seconds = now

    detector = PhraseBoundaryDetector(_boundary_config(session.config))
    decision = detector.decide(phrase_seconds, segments)
    status = (
        f"{decision.reason} | {decision.phrase_seconds:.1f}s phrase | "
        f"{decision.trailing_silence:.2f}/{decision.required_pause_seconds:.2f}s pause"
    )
    if decision.should_commit:
        _commit_text(session, _transcribe(session.config, session.phrase_buffer))
        session.phrase_buffer = np.empty(0, dtype=np.float32)
        session.last_interim_seconds = 0.0
        status = f"committed | {session.committed_phrases} lines"

    return status


def stream_captions(
    audio: tuple[int, np.ndarray] | None,
    session: LiveSession | None,
    source_lang: str,
    target_lang: str,
    model_size: str,
    beam_size: int,
    normal_pause_seconds: float,
    long_phrase_seconds: float,
    long_phrase_pause_seconds: float,
    max_phrase_seconds: float,
    input_gain: float,
    vad_threshold: float,
    should_translate: bool,
    show_interim: bool,
) -> tuple[LiveSession, str, str, str]:
    config = _config(
        source_lang,
        target_lang,
        model_size,
        beam_size,
        normal_pause_seconds,
        long_phrase_seconds,
        long_phrase_pause_seconds,
        max_phrase_seconds,
        input_gain,
        vad_threshold,
        should_translate,
        show_interim,
    )
    session = _session(session, config)
    _, chunk = _audio_chunk_to_float32(audio, config.input_gain)
    status = _process_chunk(session, chunk, update_interim=True)

    return session, render_captions(session.captions), session.captions.interim, status


def replay_file(
    audio: tuple[int, np.ndarray] | None,
    source_lang: str,
    target_lang: str,
    model_size: str,
    beam_size: int,
    normal_pause_seconds: float,
    long_phrase_seconds: float,
    long_phrase_pause_seconds: float,
    max_phrase_seconds: float,
    input_gain: float,
    vad_threshold: float,
    should_translate: bool,
) -> tuple[None, str, str, str]:
    config = _config(
        source_lang,
        target_lang,
        model_size,
        beam_size,
        normal_pause_seconds,
        long_phrase_seconds,
        long_phrase_pause_seconds,
        max_phrase_seconds,
        input_gain,
        vad_threshold,
        should_translate,
        False,
    )
    _, prepared = _audio_chunk_to_float32(audio, config.input_gain)
    session = LiveSession(config=config)
    if prepared.size == 0:
        return None, render_captions(session.captions, max_lines=None), "", "no file"

    step_samples = int(SAMPLE_RATE * STREAM_EVERY_SECONDS)
    for start in range(0, len(prepared), step_samples):
        _process_chunk(session, prepared[start : start + step_samples], update_interim=False)

    if session.phrase_buffer.size:
        segments = detect_speech(
            session.phrase_buffer,
            SAMPLE_RATE,
            threshold=session.config.vad_threshold,
        )
        if segments:
            _commit_text(session, _transcribe(session.config, session.phrase_buffer))
            session.phrase_buffer = np.empty(0, dtype=np.float32)

    duration = len(prepared) / SAMPLE_RATE
    status = f"file replay | {session.committed_phrases} lines | {duration:.1f}s audio"
    return None, render_captions(session.captions, max_lines=None), "", status


def render_captions(captions: CaptionState, max_lines: int | None = 12) -> str:
    committed = captions.committed if max_lines is None else captions.committed[-max_lines:]
    rows = [_render_line(line) for line in committed]
    if captions.interim:
        rows.append(
            "<div class='caption-line interim'>"
            f"<div class='source'>{html.escape(captions.interim)}</div>"
            "<div class='target'>...</div>"
            "</div>"
        )
    if not rows:
        rows.append("<div class='empty-state'>No captions yet.</div>")
    return "<div class='caption-board'>" + "\n".join(rows) + "</div>"


def _render_line(line: Line) -> str:
    target = html.escape(line.target) if line.target else "..."
    return (
        "<div class='caption-line'>"
        f"<div class='line-id'>{line.id + 1:02d}</div>"
        "<div class='caption-copy'>"
        f"<div class='source'>{html.escape(line.source)}</div>"
        f"<div class='target'>{target}</div>"
        "</div>"
        "</div>"
    )


def reset_session() -> tuple[None, str, str, str]:
    return None, render_captions(CaptionState()), "", "reset"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Live Subtitles") as demo:
        session = gr.State(value=None)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=360):
                captions_html = gr.HTML(render_captions(CaptionState()), label="Captions")
                interim = gr.Textbox(label="Interim", interactive=False, lines=2)
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1, min_width=300):
                audio = gr.Audio(
                    label="Microphone",
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    show_label=True,
                )
                file_audio = gr.Audio(
                    label="Audio file",
                    sources=["upload"],
                    type="numpy",
                    streaming=False,
                    show_label=True,
                )
                run_file = gr.Button("Run File", variant="primary")
                with gr.Row():
                    source_lang = gr.Radio(["en", "fr"], value="fr", label="Source")
                    target_lang = gr.Radio(["en", "fr"], value="en", label="Target")
                model_size = gr.Dropdown(["tiny", "base", "small"], value="small", label="ASR")
                beam_size = gr.Slider(1, 5, value=1, step=1, label="Beam")
                with gr.Accordion("Input", open=False):
                    input_gain = gr.Slider(0.25, 8.0, value=1.0, step=0.25, label="Gain")
                    vad_threshold = gr.Slider(0.15, 0.7, value=0.5, step=0.05, label="VAD threshold")
                with gr.Accordion("Timing", open=False):
                    normal_pause = gr.Slider(0.1, 1.5, value=0.35, step=0.05, label="Pause")
                    long_phrase = gr.Slider(2.0, 8.0, value=4.0, step=0.5, label="Long phrase")
                    short_pause = gr.Slider(0.1, 0.8, value=0.2, step=0.05, label="Short pause")
                    max_phrase = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Max phrase")
                should_translate = gr.Checkbox(value=True, label="Translate")
                show_interim = gr.Checkbox(value=True, label="Interim ASR")
                reset = gr.Button("Reset", variant="secondary")

        stream_inputs = [
            audio,
            session,
            source_lang,
            target_lang,
            model_size,
            beam_size,
            normal_pause,
            long_phrase,
            short_pause,
            max_phrase,
            input_gain,
            vad_threshold,
            should_translate,
            show_interim,
        ]
        stream_outputs = [session, captions_html, interim, status]
        audio.stream(
            stream_captions,
            stream_inputs,
            stream_outputs,
            stream_every=STREAM_EVERY_SECONDS,
            time_limit=120,
            concurrency_limit=1,
            show_progress="hidden",
        )
        run_file.click(
            replay_file,
            [
                file_audio,
                source_lang,
                target_lang,
                model_size,
                beam_size,
                normal_pause,
                long_phrase,
                short_pause,
                max_phrase,
                input_gain,
                vad_threshold,
                should_translate,
            ],
            stream_outputs,
            concurrency_limit=1,
            show_progress="minimal",
        )
        reset.click(reset_session, outputs=stream_outputs, queue=False)
    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
