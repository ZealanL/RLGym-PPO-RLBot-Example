"""RLGym compatibility helpers for the Phoenix heuristic bot."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np

from agent import Agent
from util.game_state import GameState
from your_act import ACTION_SIZE

from rlgym.api.rlgym import RLGym
from rlgym.api.typing import AgentID
from rlgym.api.config.action_parser import ActionParser
from rlgym.api.config.obs_builder import ObsBuilder
from rlgym.api.config.reward_function import RewardFunction
from rlgym.rocket_league.api.game_state import GameState as RLGymGameState
from rlgym.rocket_league.action_parsers.repeat_action import RepeatAction
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition
from rlgym.rocket_league.state_mutators.fixed_team_size_mutator import FixedTeamSizeMutator
from rlgym.rocket_league.state_mutators.kickoff_mutator import KickoffMutator
from rlgym.rocket_league.state_mutators.mutator_sequence import MutatorSequence
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine


class PhoenixObsBuilder(ObsBuilder[AgentID, np.ndarray, RLGymGameState, gym.spaces.Box]):
    """Minimal observation builder used purely to satisfy RLGym requirements."""

    def __init__(self) -> None:
        self._obs = np.zeros(1, dtype=np.float32)
        self._space = gym.spaces.Box(low=self._obs, high=self._obs, dtype=np.float32)

    def get_obs_space(self, agent: AgentID) -> gym.spaces.Box:
        return self._space

    def reset(self, agents: List[AgentID], initial_state: RLGymGameState, shared_info: Dict[str, np.ndarray]) -> None:
        # Observations are stateless for the heuristic controller.
        return

    def build_obs(
        self, agents: List[AgentID], state: RLGymGameState, shared_info: Dict[str, np.ndarray]
    ) -> Dict[AgentID, np.ndarray]:
        return {agent: self._obs.copy() for agent in agents}


class PhoenixZeroReward(RewardFunction[AgentID, RLGymGameState, float]):
    """Reward function that always returns zero (we are not training)."""

    def reset(self, agents: List[AgentID], initial_state: RLGymGameState, shared_info: Dict[str, np.ndarray]) -> None:
        return

    def get_rewards(
        self,
        agents: List[AgentID],
        state: RLGymGameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, np.ndarray],
    ) -> Dict[AgentID, float]:
        return {agent: 0.0 for agent in agents}


class PhoenixDirectActionParser(ActionParser[AgentID, np.ndarray, np.ndarray, RLGymGameState, gym.spaces.Box]):
    """Pass-through parser that accepts Phoenix controller arrays."""

    def __init__(self) -> None:
        self._low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self._space = gym.spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    def get_action_space(self, agent: AgentID) -> gym.spaces.Box:
        return self._space

    def reset(self, agents: List[AgentID], initial_state: RLGymGameState, shared_info: Dict[str, np.ndarray]) -> None:
        return

    def parse_actions(
        self, actions: Dict[AgentID, np.ndarray], state: RLGymGameState, shared_info: Dict[str, np.ndarray]
    ) -> Dict[AgentID, np.ndarray]:
        parsed: Dict[AgentID, np.ndarray] = {}
        for agent, raw in actions.items():
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size != ACTION_SIZE:
                    raise ValueError(f"Expected action of size {ACTION_SIZE}, received {arr.shape}.")
                arr = arr.reshape(1, ACTION_SIZE)
            elif arr.ndim == 2 and arr.shape[1] == ACTION_SIZE:
                pass
            else:  # pragma: no cover - defensive branch
                raise ValueError(f"Unexpected action shape {arr.shape}.")

            arr[:, :5] = np.clip(arr[:, :5], -1.0, 1.0)
            arr[:, 5:] = np.clip(arr[:, 5:], 0.0, 1.0)
            parsed[agent] = arr
        return parsed


class PhoenixGymController:
    """Bridges Phoenix heuristics to RLGym's RocketSim engine."""

    def __init__(self, tick_skip: int = 8) -> None:
        self.tick_skip = max(int(tick_skip), 1)
        self._agent = Agent()
        self._state: GameState | None = None
        self._agent_order: List[AgentID] = []
        self._last_tick = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks

    def handle_reset(self, rl_state: RLGymGameState) -> None:
        timers = getattr(rl_state, "boost_pad_timers", None)
        boost_count = len(timers) if timers is not None else 0
        self._state = GameState(boost_count=boost_count if boost_count > 0 else None)

        agent_ids: Iterable[AgentID] = rl_state.cars.keys()
        self._agent_order = sorted(agent_ids, key=lambda aid: str(aid))
        self._state.update_from_rlgym(rl_state, self._agent_order)
        self._last_tick = getattr(rl_state, "tick_count", 0)

    def build_action_dict(self, rl_state: RLGymGameState) -> Dict[AgentID, np.ndarray]:
        if self._state is None or any(agent not in rl_state.cars for agent in self._agent_order):
            self.handle_reset(rl_state)
        else:
            self._state.update_from_rlgym(rl_state, self._agent_order)

        assert self._state is not None  # for type checkers

        tick_now = getattr(rl_state, "tick_count", 0)
        tick_diff = max(tick_now - self._last_tick, self.tick_skip)
        delta_time = tick_diff / 120.0

        actions: Dict[AgentID, np.ndarray] = {}
        for idx, agent_id in enumerate(self._agent_order):
            if agent_id not in rl_state.cars or idx >= len(self._state.players):
                continue

            player = self._state.players[idx]
            context = {
                "state": self._state,
                "player": player,
                "delta_time": delta_time,
                "tick_skip": tick_diff,
            }
            controls = self._agent.act(context)
            actions[agent_id] = np.asarray(controls, dtype=np.float32)

        self._last_tick = tick_now
        return actions


def make_gym_environment(
    *,
    tick_skip: int = 8,
    blue_team_size: int = 1,
    orange_team_size: int = 1,
    auto_reset: bool = True,
) -> Tuple[RLGym, PhoenixGymController]:
    """Create an ``RLGym`` environment configured for the Phoenix bot."""

    base_parser = PhoenixDirectActionParser()
    action_parser = RepeatAction(base_parser, repeats=max(int(tick_skip), 1))
    obs_builder = PhoenixObsBuilder()
    reward_fn = PhoenixZeroReward()
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        transition_engine=RocketSimEngine(rlbot_delay=True),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(timeout_seconds=120.0),
    )

    controller = PhoenixGymController(tick_skip=tick_skip)
    if auto_reset:
        env.reset()
        controller.handle_reset(env.state)

    return env, controller


def run_episode(max_steps: int = 1800, tick_skip: int = 8) -> None:
    """Run a demonstration episode inside RocketSim using Phoenix heuristics."""

    env, controller = make_gym_environment(tick_skip=tick_skip, auto_reset=True)
    try:
        done = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        step = 0

        while step < max_steps and not any(done.values()) and not any(truncated.values()):
            actions = controller.build_action_dict(env.state)
            _, _, done, truncated = env.step(actions)
            step += 1

        if any(done.values()) or any(truncated.values()):
            env.reset()
            controller.handle_reset(env.state)
    finally:
        env.close()


__all__ = [
    "make_gym_environment",
    "run_episode",
    "PhoenixGymController",
    "PhoenixDirectActionParser",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    run_episode()

