import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from agent import Agent
from your_obs import YourOBS

class RLGymPPOBot(BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		self.obs_builder = YourOBS()
		self.agent = Agent()
		self.tick_skip = your_tick_skip_here
		self.game_state: GameState = None
		self.controls = None
		self.action = None
		self.update_action = True
		self.ticks = 0
		self.prev_time = 0
		print('====================================')
		print('RLGym-PPO Bot Ready - Index:', index)
		print('Make sure your FPS is at 120, 240, or 360!')
		print('====================================')

	def is_hot_reload_enabled(self):
		return True

	def initialize_agent(self):
		# Initialize the rlgym GameState object now that the game is active and the info is available
		self.game_state = GameState(self.get_field_info())
		self.update_action = True
		self.ticks = self.tick_skip  # So we take an action the first tick
		self.prev_time = 0
		self.controls = SimpleControllerState()
		self.action = np.zeros(8)

	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
		cur_time = packet.game_info.seconds_elapsed
		delta = cur_time - self.prev_time
		self.prev_time = cur_time

		ticks_elapsed = round(delta * 120)
		self.ticks += ticks_elapsed
		self.game_state.decode(packet, ticks_elapsed)

		# We calculate the next action as soon as the prev action is sent
		# This gives you tick_skip ticks to do your forward pass
		if self.update_action:
			self.update_action = False

			player = self.game_state.players[self.index]
			
			obs = self.obs_builder.build_obs(player, self.game_state, self.action)
			self.action = self.agent.act(obs)

		if self.ticks >= self.tick_skip - 1:
			self.update_controls(self.action)

		if self.ticks >= self.tick_skip:
			self.ticks = 0
			self.update_action = True

		return self.controls

	def update_controls(self, action):
		self.controls.throttle = action[0]
		self.controls.steer = action[1]
		self.controls.pitch = action[2]
		self.controls.yaw = action[3]
		self.controls.roll = action[4]
		self.controls.jump = action[5] > 0
		self.controls.boost = action[6] > 0
		self.controls.handbrake = action[7] > 0
