"""Utility helpers for generating controller arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

ACTION_SIZE = 8


def _controls(
    throttle: float = 0.0,
    steer: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    jump: float = 0.0,
    boost: float = 0.0,
    handbrake: float = 0.0,
) -> np.ndarray:
    arr = np.asarray([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)
    return np.clip(arr, -1.0, 1.0)


@dataclass(frozen=True)
class ControlPreset:
    """Named preset storing a ready-to-use controller array."""

    name: str
    values: np.ndarray

    def copy(self) -> np.ndarray:
        return self.values.copy()


class ControlLibrary:
    """Collection of reusable controller presets."""

    def __init__(self) -> None:
        self._presets: Dict[str, ControlPreset] = {
            preset.name: preset
            for preset in [
                ControlPreset("neutral", _controls()),
                ControlPreset("stabilise", _controls(throttle=0.3)),
                ControlPreset("drive_forward", _controls(throttle=1.0)),
                ControlPreset("drive_reverse", _controls(throttle=-0.8)),
                ControlPreset("boost_forward", _controls(throttle=1.0, boost=1.0)),
                ControlPreset("hard_left", _controls(throttle=1.0, steer=-1.0)),
                ControlPreset("hard_right", _controls(throttle=1.0, steer=1.0)),
                ControlPreset("powerslide_left", _controls(throttle=1.0, steer=-1.0, handbrake=1.0)),
                ControlPreset("powerslide_right", _controls(throttle=1.0, steer=1.0, handbrake=1.0)),
                ControlPreset("jump", _controls(jump=1.0)),
                ControlPreset("double_jump", _controls(jump=1.0, pitch=-0.3)),
            ]
        }

    def get(self, name: str) -> np.ndarray:
        preset = self._presets.get(name)
        if preset is None:
            raise KeyError(f"Unknown control preset: {name}")
        return preset.copy()

    def neutral(self) -> np.ndarray:
        return self.get("neutral")

    def stabilise(self) -> np.ndarray:
        return self.get("stabilise")

    def drive_forward(self) -> np.ndarray:
        return self.get("drive_forward")

    def drive_reverse(self) -> np.ndarray:
        return self.get("drive_reverse")

    def boost_forward(self) -> np.ndarray:
        return self.get("boost_forward")

    def hard_left(self) -> np.ndarray:
        return self.get("hard_left")

    def hard_right(self) -> np.ndarray:
        return self.get("hard_right")

    def powerslide_left(self) -> np.ndarray:
        return self.get("powerslide_left")

    def powerslide_right(self) -> np.ndarray:
        return self.get("powerslide_right")


def blend(base: np.ndarray, *, steer: float | None = None, throttle: float | None = None, boost: float | None = None,
          handbrake: float | None = None) -> np.ndarray:
    """Return a modified copy of ``base`` with selected fields overridden."""

    result = base.copy()
    if steer is not None:
        result[1] = float(np.clip(steer, -1.0, 1.0))
    if throttle is not None:
        result[0] = float(np.clip(throttle, -1.0, 1.0))
    if boost is not None:
        result[6] = float(np.clip(boost, 0.0, 1.0))
    if handbrake is not None:
        result[7] = float(np.clip(handbrake, 0.0, 1.0))
    return result


__all__ = ["ACTION_SIZE", "ControlLibrary", "blend"]
