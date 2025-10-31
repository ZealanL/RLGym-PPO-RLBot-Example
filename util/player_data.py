"""Simplified player data mirror used by the heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .physics_object import PhysicsObject


@dataclass
class PlayerData:
    car_id: int = -1
    team_num: int = -1
    is_demoed: bool = False
    on_ground: bool = False
    ball_touched: bool = False
    has_flip: bool = False
    boost_amount: float = 0.0
    car_data: PhysicsObject = field(default_factory=PhysicsObject)
    inverted_car_data: PhysicsObject = field(default_factory=PhysicsObject)
