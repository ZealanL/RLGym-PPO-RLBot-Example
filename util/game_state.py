"""Light-weight replication of the RLGym game state for heuristics."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket, PlayerInfo

from .physics_object import PhysicsObject
from .player_data import PlayerData

if TYPE_CHECKING:  # pragma: no cover - imported lazily for RLGym compatibility
    from rlgym.api.typing import AgentID
    from rlgym.rocket_league.api.car import Car as RLGymCar
    from rlgym.rocket_league.api.game_state import GameState as RLGymGameState
    from rlgym.rocket_league.api.physics_object import PhysicsObject as RLGymPhysics


class GameState:
    """Mirror of the RLGym ``GameState`` tailored for heuristic control."""

    def __init__(self, game_info: Optional[FieldInfoPacket] = None, *, boost_count: Optional[int] = None):
        self.blue_score = 0
        self.orange_score = 0
        self.players: List[PlayerData] = []
        self._on_ground_ticks = np.zeros(64, dtype=np.int32)

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        total_boosts = boost_count
        if total_boosts is None:
            if game_info is not None:
                total_boosts = int(getattr(game_info, "num_boosts", 34))
            else:
                total_boosts = 34

        self.boost_pads: np.ndarray = np.zeros(total_boosts, dtype=np.float32)
        self.inverted_boost_pads: np.ndarray = np.zeros_like(self.boost_pads, dtype=np.float32)
        self.last_touch: int | None = None

    def decode(self, packet: GameTickPacket, ticks_elapsed: int = 1) -> None:
        try:
            ticks = int(round(float(ticks_elapsed)))
        except (TypeError, ValueError):
            ticks = 1
        ticks = max(ticks, 1)

        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score

        boost_count = min(packet.num_boost, self.boost_pads.size)
        for i in range(boost_count):
            self.boost_pads[i] = float(packet.game_boosts[i].is_active)
        if boost_count < self.boost_pads.size:
            self.boost_pads[boost_count:] = 0
        self.inverted_boost_pads[:] = self.boost_pads[::-1]

        if packet.game_ball is not None:
            self.ball.decode_ball_data(packet.game_ball.physics)
            self.inverted_ball.invert(self.ball)

        self.players = []
        player_limit = min(packet.num_cars, len(self._on_ground_ticks))
        for i in range(player_limit):
            player_info = packet.game_cars[i]
            if player_info is None:
                continue

            player = self._decode_player(player_info, i, ticks)
            self.players.append(player)

            if player.ball_touched:
                self.last_touch = player.car_id

    def _decode_player(self, player_info: PlayerInfo, index: int, ticks_elapsed: int) -> PlayerData:
        player_data = PlayerData()

        physics = player_info.physics
        if physics is not None:
            player_data.car_data.decode_car_data(physics)
            player_data.inverted_car_data.invert(player_data.car_data)

        if player_info.has_wheel_contact:
            self._on_ground_ticks[index] = 0
        else:
            self._on_ground_ticks[index] += ticks_elapsed

        player_data.car_id = index
        player_data.team_num = player_info.team
        player_data.is_demoed = player_info.is_demolished
        player_data.on_ground = player_info.has_wheel_contact or self._on_ground_ticks[index] <= 6
        player_data.ball_touched = bool(player_info.ball_touched)
        player_data.has_flip = not player_info.double_jumped
        player_data.boost_amount = player_info.boost / 100

        return player_data

    # ------------------------------------------------------------------
    # RLGym compatibility helpers

    def update_from_rlgym(self, rl_state: "RLGymGameState", agent_order: Sequence["AgentID"] | None = None) -> None:
        """Populate the simplified state using an RLGym ``GameState`` instance."""

        if rl_state is None:
            return

        # Boost pad availability (0 = cooldown, 1 = ready)
        timers = getattr(rl_state, "boost_pad_timers", None)
        if timers is not None:
            self._ensure_boost_capacity(len(timers))
            np.less_equal(np.asarray(timers, dtype=np.float32), 0.0, out=self.boost_pads)
            self.boost_pads = self.boost_pads.astype(np.float32, copy=False)
            self.inverted_boost_pads[:] = self.boost_pads[::-1]

        if getattr(rl_state, "ball", None) is not None:
            self._copy_rlgym_physics(self.ball, rl_state.ball)
        if getattr(rl_state, "inverted_ball", None) is not None:
            self._copy_rlgym_physics(self.inverted_ball, rl_state.inverted_ball)
        elif getattr(rl_state, "ball", None) is not None:
            self.inverted_ball.invert(self.ball)

        car_items: Iterable[tuple["AgentID", "RLGymCar"]]
        if agent_order is None:
            car_items = rl_state.cars.items()
        else:
            car_items = ((agent, rl_state.cars[agent]) for agent in agent_order if agent in rl_state.cars)

        players: List[PlayerData] = []
        for idx, (_, car) in enumerate(car_items):
            player = PlayerData()
            player.car_id = idx
            player.team_num = int(getattr(car, "team_num", -1))
            player.is_demoed = bool(getattr(car, "is_demoed", False))
            player.on_ground = bool(getattr(car, "on_ground", False))
            player.ball_touched = bool(getattr(car, "ball_touches", 0))
            player.has_flip = bool(getattr(car, "has_flip", False))
            player.boost_amount = float(getattr(car, "boost_amount", 0.0)) / 100.0

            car_physics = getattr(car, "physics", None)
            if car_physics is not None:
                self._copy_rlgym_physics(player.car_data, car_physics)
            inverted_physics = getattr(car, "inverted_physics", None)
            if inverted_physics is not None:
                self._copy_rlgym_physics(player.inverted_car_data, inverted_physics)
            elif car_physics is not None:
                player.inverted_car_data.invert(player.car_data)

            players.append(player)

        self.players = players

    def _ensure_boost_capacity(self, count: int) -> None:
        if count <= 0:
            return
        if self.boost_pads.size != count:
            self.boost_pads = np.zeros(count, dtype=np.float32)
            self.inverted_boost_pads = np.zeros(count, dtype=np.float32)

    @staticmethod
    def _copy_rlgym_physics(target: PhysicsObject, source: "RLGymPhysics") -> None:
        target.position = np.asarray(source.position, dtype=np.float32)
        target.linear_velocity = np.asarray(source.linear_velocity, dtype=np.float32)
        target.angular_velocity = np.asarray(source.angular_velocity, dtype=np.float32)

        try:
            rotation_mtx = np.asarray(source.rotation_mtx, dtype=np.float32)
        except Exception:  # pragma: no cover - fall back to euler angles
            rotation_mtx = None
        if rotation_mtx is not None and rotation_mtx.shape == (3, 3):
            target._rotation_mtx = rotation_mtx
            target._has_computed_rot_mtx = True
        else:
            target._rotation_mtx = np.identity(3, dtype=np.float32)
            target._has_computed_rot_mtx = False

        try:
            euler = np.asarray(source.euler_angles, dtype=np.float32)
        except Exception:  # pragma: no cover - rely on rotation matrix later
            euler = None
        if euler is not None and euler.shape == (3,):
            target._euler_angles = euler
        elif target._has_computed_rot_mtx:
            # ``PhysicsObject.rotation_mtx`` will compute the Euler angles lazily.
            target._euler_angles = np.zeros(3, dtype=np.float32)
        else:
            target._euler_angles = np.zeros(3, dtype=np.float32)
