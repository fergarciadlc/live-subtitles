"""Caption state machine — build-order step 5 (the deterministic core).

The fork in the design (see the plan): ASR output streams live into an *interim*
source line; the VAD pause-point *commits* that line and triggers translation. This
keeps the slow MT step off the latency-critical path — committed lines get their
translation filled in afterwards, asynchronously, without blocking new speech.

This module is pure state — no audio, no model, no UI. It holds what to show and
enforces the interim→committed transition; the app reads it to render captions and
calls translate.py to fill in targets. Fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Line:
    """One committed source phrase and its (eventual) translation.

    `target` is None until translation completes — the app renders a placeholder
    in the gap. `id` is stable, so a translation that finishes out of order still
    lands on the right line.
    """

    id: int
    source: str
    target: str | None = None


class CaptionState:
    """Live captions: a growing list of committed lines plus one interim line.

    Usage mirrors the pipeline:
        state.update_interim(asr_text)   # repeatedly, as audio streams in
        line = state.commit()            # on a VAD pause; None if nothing pending
        if line: translate + state.set_translation(line.id, target)
    """

    def __init__(self) -> None:
        self._committed: list[Line] = []
        self._interim: str = ""
        self._next_id: int = 0

    @property
    def interim(self) -> str:
        """The live, not-yet-committed source line (empty between phrases)."""
        return self._interim

    @property
    def committed(self) -> tuple[Line, ...]:
        """Committed lines in order. A tuple so callers can't mutate the list."""
        return tuple(self._committed)

    def update_interim(self, text: str) -> None:
        """Replace the live source line with the latest ASR text (whitespace-trimmed)."""
        self._interim = text.strip()

    def commit(self) -> Line | None:
        """Promote the interim line to a committed line and return it for translation.

        Clears the interim either way. Returns None when there is nothing pending
        (a pause on silence, or a double-commit) so no empty line is ever created.
        """
        text = self._interim.strip()
        self._interim = ""
        if not text:
            return None
        line = Line(id=self._next_id, source=text)
        self._next_id += 1
        self._committed.append(line)
        return line

    def set_translation(self, line_id: int, target: str) -> None:
        """Attach a finished translation to a committed line by id."""
        for line in self._committed:
            if line.id == line_id:
                line.target = target.strip()
                return
        raise KeyError(f"no committed line with id {line_id}")
