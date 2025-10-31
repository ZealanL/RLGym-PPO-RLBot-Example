from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mechanics import MacroAction, MacroInstance, MechanicSupervisor, routines
from util.common_values import BLUE_GOAL_CENTER, BLUE_TEAM, ORANGE_GOAL_CENTER
from util.game_state import GameState
from util.player_data import PlayerData
from your_act import ControlLibrary, blend


@dataclass
class Decision:
    """Container describing the agent's next move."""

    controls: Optional[np.ndarray] = None
    macro: Optional[MacroAction] = None


class Agent:
    """Rule-based agent that mixes heuristics with scripted mechanics."""

    def __init__(self) -> None:
        self._controls = ControlLibrary()
        self._supervisor = MechanicSupervisor()
        self._macro_instance: Optional[MacroInstance] = None
        # Pre-build reusable macros so they can be triggered without delay.
        self._fast_aerial = routines.fast_aerial_macro()
        self._half_flip = routines.half_flip_macro()
        self._power_shot = routines.power_shot_macro()
        self._dribble = routines.ground_dribble_macro()

    # ------------------------------------------------------------------
    # Public API

    def act(self, context: Optional[Dict[str, Any]]) -> np.ndarray:
        if not isinstance(context, dict) or not context:
            return self._controls.neutral()

        state: GameState = context.get("state")
        player: PlayerData = context.get("player")
        if not isinstance(state, GameState) or not isinstance(player, PlayerData) or player.is_demoed:
            return self._controls.neutral()

        # Allow the rule-based supervisor to seize control when the match state
        # demands an urgent mechanic (kickoff, recovery, panic clear).
        supervisor_macro = self._supervisor.maybe_override(
            context, active_macro=self._macro_instance.macro if self._macro_instance else None
        )
        if supervisor_macro is not None:
            self._start_macro(supervisor_macro)

        if self._macro_instance is not None:
            return self._advance_macro()

        decision = self._evaluate_state(state, player)
        if decision.macro is not None:
            self._start_macro(decision.macro)
            return self._advance_macro()

        return decision.controls if decision.controls is not None else self._controls.neutral()

    # ------------------------------------------------------------------
    # Decision making

    def _evaluate_state(self, state: GameState, player: PlayerData) -> Decision:
        car = player.car_data
        ball = state.ball

        to_ball = ball.position - car.position
        flat_dist = float(np.linalg.norm(to_ball[:2]))
        height_diff = float(ball.position[2] - car.position[2])

        forward = car.forward()
        forward_flat = forward[:2]
        forward_norm = np.linalg.norm(forward_flat)
        if forward_norm > 1e-6:
            forward_flat /= forward_norm
        ball_flat = to_ball[:2]
        ball_norm = np.linalg.norm(ball_flat)
        if ball_norm > 1e-6:
            ball_flat /= ball_norm
        facing_ball = float(np.dot(forward_flat, ball_flat)) if ball_norm > 1e-6 else 0.0

        if self._should_fast_aerial(player, flat_dist, height_diff, facing_ball):
            return Decision(macro=self._fast_aerial)

        if self._should_half_flip(player, facing_ball, flat_dist):
            return Decision(macro=self._half_flip)

        if self._should_power_shot(flat_dist, height_diff, facing_ball):
            return Decision(macro=self._power_shot)

        if self._should_dribble(flat_dist, height_diff, facing_ball):
            return Decision(macro=self._dribble)

        target = self._choose_target(state, player)
        controls = self._drive_towards(player, target, boost_ok=flat_dist > 1800)
        return Decision(controls=controls)

    def _should_fast_aerial(self, player: PlayerData, flat_dist: float, height_diff: float, facing_ball: float) -> bool:
        return (
            player.on_ground
            and player.boost_amount > 0.45
            and height_diff > 220
            and flat_dist > 900
            and facing_ball > 0.4
        )

    def _should_half_flip(self, player: PlayerData, facing_ball: float, flat_dist: float) -> bool:
        return player.on_ground and facing_ball < -0.3 and flat_dist > 900 and player.has_flip

    def _should_power_shot(self, flat_dist: float, height_diff: float, facing_ball: float) -> bool:
        return flat_dist < 750 and height_diff < 200 and facing_ball > 0.6

    def _should_dribble(self, flat_dist: float, height_diff: float, facing_ball: float) -> bool:
        return flat_dist < 600 and height_diff < 150 and facing_ball > 0.4

    def _choose_target(self, state: GameState, player: PlayerData) -> np.ndarray:
        ball = state.ball
        car = player.car_data
        team = player.team_num

        if team == BLUE_TEAM:
            opponent_goal = np.asarray(ORANGE_GOAL_CENTER, dtype=np.float32)
        else:
            opponent_goal = np.asarray(BLUE_GOAL_CENTER, dtype=np.float32)

        to_goal = opponent_goal - ball.position
        to_goal_flat = to_goal[:2]
        distance_to_goal = np.linalg.norm(to_goal_flat)
        if distance_to_goal > 1e-6:
            to_goal_flat /= distance_to_goal

        approach_offset = to_goal_flat * 750.0
        approach = np.asarray([ball.position[0] - approach_offset[0], ball.position[1] - approach_offset[1], car.position[2]])

        # Encourage shadowing when the ball is deep in defence.
        defensive_line = -3500 if team == BLUE_TEAM else 3500
        if (team == BLUE_TEAM and ball.position[1] < defensive_line) or (
            team != BLUE_TEAM and ball.position[1] > defensive_line
        ):
            shadow_offset = to_goal_flat * -1200.0
            approach = np.asarray(
                [ball.position[0] + shadow_offset[0], ball.position[1] + shadow_offset[1], car.position[2]]
            )

        # Keep the target slightly ahead of the ball to avoid stopping on top of it.
        lead = to_goal_flat * -120.0
        approach[:2] += lead
        return approach

    # ------------------------------------------------------------------
    # Macro handling

    def _advance_macro(self) -> np.ndarray:
        if self._macro_instance is None:
            return self._controls.neutral()

        controls = self._macro_instance.step().copy()
        if self._macro_instance.finished:
            self._macro_instance = None
        return controls

    def _start_macro(self, macro: MacroAction) -> None:
        if macro is None:
            self._macro_instance = None
            return

        if (
            self._macro_instance is not None
            and not self._macro_instance.finished
            and self._macro_instance.macro is macro
        ):
            return

        try:
            self._macro_instance = macro.instantiate()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            print(f"Failed to start macro '{getattr(macro, 'name', '<unknown>')}': {exc}")
            self._macro_instance = None

    # ------------------------------------------------------------------
    # Low-level driving helpers

    def _drive_towards(self, player: PlayerData, target: np.ndarray, *, boost_ok: bool) -> np.ndarray:
        car = player.car_data
        to_target = target - car.position
        distance = float(np.linalg.norm(to_target[:2]))

        rotation = car.rotation_mtx()
        if rotation.shape != (3, 3):
            rotation = np.identity(3, dtype=np.float32)
        local = rotation.T @ to_target
        angle = math.atan2(local[1], local[0])

        steer = np.clip(angle * 2.0, -1.0, 1.0)
        throttle = 1.0 if local[0] > 0 else -1.0

        if distance < 900:
            throttle = np.clip(distance / 900, -1.0, 1.0) * (1.0 if throttle > 0 else -1.0)

        handbrake = 1.0 if abs(angle) > 1.8 and distance < 1600 else 0.0
        boost = 0.0
        if boost_ok and throttle > 0.6 and abs(angle) < 0.3 and player.boost_amount > 0.2:
            forward_speed = float(np.dot(car.linear_velocity, car.forward()))
            if forward_speed < 1900:
                boost = 1.0

        base = self._controls.drive_forward() if throttle >= 0 else self._controls.drive_reverse()
        return blend(base, steer=steer, throttle=throttle, boost=boost, handbrake=handbrake)
