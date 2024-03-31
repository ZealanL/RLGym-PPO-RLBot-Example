import math
import numpy as np
from typing import Any, List

from util.game_state import GameState
from util.physics_object import PhysicsObject
from util.player_data import PlayerData

# IMPORTANT NOTICE:
# In some cases, the right row of player rotmats may actually be the left
# If your bot seems broken for no reason, try flipping right/left of rotmat

class YourOBS:
    def __init__(self):
        print("Implement your OBS here!")

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        print("Implement your OBS here!")
        return None