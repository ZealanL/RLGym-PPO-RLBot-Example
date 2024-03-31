import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from your_act import SaturnAction
import your_obs
from discrete_policy import DiscreteFF

class Agent:
	def __init__(self):
		self.action_parser = SaturnAction()
		self.num_actions = len(self.action_parser._lookup_table)
		cur_dir = os.path.dirname(os.path.realpath(__file__))
		
		device = torch.device("cpu")
		self.policy = DiscreteFF(113, self.num_actions, [2048, 1024, 1024, 1024], device)
		self.policy.load_state_dict(torch.load(os.path.join(cur_dir, "PPO_POLICY.pt"), map_location=device))
		torch.set_num_threads(1)

	def act(self, state):
		with torch.no_grad():
			action_idx, probs = self.policy.get_action(state, True)
		
		return self.action_parser.parse_action([action_idx], None)[0]
