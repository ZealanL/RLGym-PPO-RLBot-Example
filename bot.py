from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from agent import Agent
from util.game_state import GameState
from your_act import ACTION_SIZE


class WinYour1sBot(BaseAgent):
    """Heuristic Rocket League bot with scripted mechanics."""

    def __init__(self, name: str, team: int, index: int):
        super().__init__(name, team, index)
        self.agent = Agent()
        self.game_state: Optional[GameState] = None
        self.controls = SimpleControllerState()
        self.prev_time = 0.0

    def is_hot_reload_enabled(self) -> bool:
        return True

    def initialize_agent(self) -> None:
        field_info = self.get_field_info()
        self.game_state = GameState(field_info)
        self.controls = SimpleControllerState()
        self.prev_time = 0.0
        print("====================================")
        print("WinYour1s ready - Index:", self.index)
        print("Ensure your Rocket League FPS is 120, 240, or 360 for smooth control.")
        print("====================================")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.game_state is None:
            return SimpleControllerState()

        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = max(int(round(delta * 120)), 1)
        self.game_state.decode(packet, ticks_elapsed)

        if self.index >= len(self.game_state.players):
            return SimpleControllerState()

        player = self.game_state.players[self.index]
        context = {
            "state": self.game_state,
            "player": player,
            "delta_time": delta,
            "tick_skip": ticks_elapsed,
        }

        action = self.agent.act(context)
        self._apply_controls(action)
        return self.controls

    # ------------------------------------------------------------------
    # Helpers

    def _apply_controls(self, action: np.ndarray | Sequence[float]) -> None:
        normalized = self._normalize_action(action)

        self.controls.throttle = float(normalized[0])
        self.controls.steer = float(normalized[1])
        self.controls.pitch = float(normalized[2])
        self.controls.yaw = float(normalized[3])
        self.controls.roll = float(normalized[4])
        self.controls.jump = bool(normalized[5] > 0.5)
        self.controls.boost = bool(normalized[6] > 0.5)
        self.controls.handbrake = bool(normalized[7] > 0.5)

    def _normalize_action(self, action: np.ndarray | Sequence[float] | None) -> np.ndarray:
        if isinstance(action, np.ndarray):
            arr = np.asarray(action, dtype=np.float32).flatten()
        else:
            arr = np.asarray(action if action is not None else (), dtype=np.float32).flatten()

        if arr.size < ACTION_SIZE:
            padded = np.zeros(ACTION_SIZE, dtype=np.float32)
            padded[: arr.size] = arr
        else:
            padded = arr[:ACTION_SIZE]

        clipped = np.clip(padded, -1.0, 1.0)
        return clipped


# RLBot expects a ``Bot`` class at module scope.
Bot = WinYour1sBot
