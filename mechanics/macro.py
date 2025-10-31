"""Primitive building blocks for macro actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ACTION_SIZE = 8


def _ensure_controls(controls: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(controls), dtype=np.float32)
    if arr.shape != (ACTION_SIZE,):
        raise ValueError(f"Controls must have shape ({ACTION_SIZE},) – received {arr.shape}")
    return np.clip(arr, -1.0, 1.0)


@dataclass(frozen=True)
class ControlStep:
    """Single step of a macro with fixed controls for a duration in ticks."""

    duration: int
    controls: np.ndarray

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("ControlStep duration must be positive")
        object.__setattr__(self, "controls", _ensure_controls(self.controls))


class MacroAction:
    """Collection of timed control steps executed sequentially."""

    def __init__(self, name: str, steps: Sequence[ControlStep]) -> None:
        if not steps:
            raise ValueError("MacroAction must contain at least one ControlStep")
        self.name = name
        self.steps = tuple(steps)

    def instantiate(self) -> "MacroInstance":
        return MacroInstance(self)


class MacroInstance:
    """Stateful execution helper for :class:`MacroAction`."""

    def __init__(self, macro: MacroAction) -> None:
        self._macro = macro
        self._step_idx = 0
        self._ticks_remaining = macro.steps[0].duration
        self.finished = False

    @property
    def macro(self) -> MacroAction:
        return self._macro

    def step(self) -> np.ndarray:
        if self.finished:
            return self._macro.steps[-1].controls.copy()

        current_step = self._macro.steps[self._step_idx]
        controls = current_step.controls.copy()
        self._ticks_remaining -= 1

        if self._ticks_remaining <= 0:
            self._step_idx += 1
            if self._step_idx >= len(self._macro.steps):
                self.finished = True
            else:
                self._ticks_remaining = self._macro.steps[self._step_idx].duration

        return controls

