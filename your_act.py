import math
import numpy as np
import gym.spaces

# Action format
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

class YourActionParser:
    def __init__(self):
       print("Implement your action parser here!")
    
    def parse_action(self, action: int, state) -> np.ndarray:
        print("Implement your action parser here!")
        return None