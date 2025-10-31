"""Scripted mechanical routines exposed to the policy."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .macro import ControlStep, MacroAction


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
    return np.asarray([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)


def _macro(name: str, steps: Sequence[ControlStep]) -> MacroAction:
    return MacroAction(name, steps)


def fast_aerial_macro() -> MacroAction:
    steps = [
        ControlStep(4, _controls(throttle=1.0, jump=1.0, boost=1.0)),
        ControlStep(2, _controls(throttle=1.0, jump=0.0, boost=1.0)),
        ControlStep(4, _controls(throttle=1.0, pitch=-1.0, boost=1.0)),
        ControlStep(18, _controls(throttle=1.0, pitch=-0.8, boost=1.0)),
    ]
    return _macro("fast_aerial", steps)


def half_flip_macro() -> MacroAction:
    steps = [
        ControlStep(3, _controls(throttle=-1.0, jump=1.0, pitch=-1.0)),
        ControlStep(2, _controls(throttle=-1.0, jump=0.0, pitch=1.0)),
        ControlStep(1, _controls(jump=1.0, pitch=1.0)),
        ControlStep(6, _controls(roll=-1.0, yaw=-1.0)),
        ControlStep(8, _controls(throttle=1.0)),
    ]
    return _macro("half_flip", steps)


def power_shot_macro() -> MacroAction:
    steps = [
        ControlStep(6, _controls(throttle=1.0, steer=0.2)),
        ControlStep(2, _controls(throttle=1.0, jump=1.0, pitch=-0.4)),
        ControlStep(4, _controls(throttle=1.0, jump=0.0, pitch=-0.8)),
        ControlStep(2, _controls(throttle=1.0, jump=1.0, pitch=-1.0)),
        ControlStep(6, _controls(throttle=1.0, pitch=-1.0)),
    ]
    return _macro("power_shot", steps)


def speed_flip_kickoff_macro() -> MacroAction:
    steps = [
        ControlStep(8, _controls(throttle=1.0, boost=1.0, steer=-0.4)),
        ControlStep(1, _controls(throttle=1.0, jump=1.0, boost=1.0, steer=-0.4)),
        ControlStep(2, _controls(throttle=1.0, boost=1.0, yaw=-0.6)),
        ControlStep(1, _controls(throttle=1.0, jump=1.0, yaw=-1.0)),
        ControlStep(10, _controls(throttle=1.0, pitch=-0.9, yaw=-1.0)),
    ]
    return _macro("speed_flip_kickoff", steps)


def aerial_recovery_macro() -> MacroAction:
    steps = [
        ControlStep(4, _controls(throttle=0.7, pitch=0.4, yaw=0.4)),
        ControlStep(6, _controls(throttle=0.7, pitch=-0.6, roll=-0.4)),
        ControlStep(8, _controls(throttle=0.7, pitch=-0.2, yaw=0.0)),
    ]
    return _macro("aerial_recovery", steps)


def ground_dribble_macro() -> MacroAction:
    steps = [
        ControlStep(10, _controls(throttle=0.4, steer=0.0)),
        ControlStep(15, _controls(throttle=0.5, steer=0.2)),
        ControlStep(12, _controls(throttle=0.5, steer=-0.2)),
        ControlStep(6, _controls(throttle=0.8, steer=0.0)),
    ]
    return _macro("ground_dribble", steps)


def panic_clear_macro() -> MacroAction:
    steps = [
        ControlStep(2, _controls(throttle=1.0, jump=1.0, pitch=-1.0)),
        ControlStep(4, _controls(throttle=1.0, pitch=-1.0, boost=1.0)),
        ControlStep(4, _controls(throttle=1.0, pitch=-0.3)),
    ]
    return _macro("panic_clear", steps)

