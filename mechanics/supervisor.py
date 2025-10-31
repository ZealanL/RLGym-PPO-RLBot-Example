"""Rule-based supervisor which can override the PPO policy."""

from __future__ import annotations

from typing import Optional

import numpy as np

from util.common_values import BLUE_GOAL_CENTER, BLUE_TEAM, ORANGE_GOAL_CENTER

from .macro import MacroAction
from . import routines


class MechanicSupervisor:
    """Light-weight heuristics to trigger scripted mechanics when necessary."""

    def __init__(self) -> None:
        self._kickoff_macro = routines.speed_flip_kickoff_macro()
        self._recovery_macro = routines.aerial_recovery_macro()
        self._panic_macro = routines.panic_clear_macro()

    def maybe_override(
        self,
        context,
        *,
        active_macro=None,
    ) -> Optional[MacroAction]:
        if not context:
            return None

        state = context.get("state")
        player = context.get("player")
        if state is None or player is None:
            return None

        if active_macro is not None:
            return None

        if self._is_kickoff(state):
            return self._kickoff_macro

        if self._needs_recovery(player):
            return self._recovery_macro

        if self._own_goal_threat(state, player):
            return self._panic_macro

        return None

    # ------------------------------------------------------------------
    # Helpers

    def _is_kickoff(self, state) -> bool:
        ball = state.ball
        return (
            np.linalg.norm(ball.position[:2]) < 50
            and np.linalg.norm(ball.linear_velocity) < 5
        )

    def _needs_recovery(self, player) -> bool:
        car = player.car_data
        if player.on_ground:
            return False
        return abs(car.position[2]) < 300 and np.linalg.norm(car.linear_velocity) > 400

    def _own_goal_threat(self, state, player) -> bool:
        ball = state.ball
        goal = BLUE_GOAL_CENTER if player.team_num == BLUE_TEAM else ORANGE_GOAL_CENTER
        goal_vec = np.asarray(goal) - ball.position
        ball_vel = ball.linear_velocity
        toward_goal = np.dot(goal_vec[:2], ball_vel[:2]) < 0
        close = np.linalg.norm(goal_vec) < 2500
        return toward_goal and close

