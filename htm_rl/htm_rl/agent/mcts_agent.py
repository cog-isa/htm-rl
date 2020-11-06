from collections import deque
from itertools import islice
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from htm_rl.agent.mcts_actor_critic import MctsActorCritic
from htm_rl.agent.mcts_planner import MctsPlanner
from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import timed, trace
from htm_rl.envs.gridworld_map_generator import GridworldMapGenerator
from htm_rl.envs.gridworld_mdp import GridworldMdp
from htm_rl.envs.mdp import Mdp


class MctsAgent:
    memory: Memory
    planner: MctsPlanner
    _n_actions: int
    _planning_horizon: int

    def __init__(
            self, memory: Memory, planner: MctsPlanner,
            mcts_actor_critic: MctsActorCritic, n_actions
    ):
        self.memory = memory
        self.planner = planner
        self._mcts_actor_critic = mcts_actor_critic
        self._n_actions = n_actions

    @property
    def _planning_enabled(self):
        return self.planner.planning_horizon > 0

    def set_planning_horizon(self, planning_horizon: int):
        self.planner.planning_horizon = planning_horizon

    def reset(self):
        self.memory.tm.reset()
        self._mcts_actor_critic.reset()
        self._trace_mcts_stats()

    def make_step(self, state, reward, is_done, verbosity: int):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')

        # TODO remove max(0)
        encoded_state = self.memory.encoder.encode(Sa(state, None))
        self._mcts_actor_critic.add_step(encoded_state, max(reward, 0))

        action = self._make_action(state, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        self.memory.train(Sa(state, action), verbosity)
        if reward > 0:
            self.planner.add_goal(state)
        return action

    def _make_action(self, state, verbosity: int):
        if self._planning_enabled:
            current_sa = Sa(state, None)
            planned_actions, self._goal_state = self.planner.plan_actions(current_sa, verbosity)
        else:
            action = np.random.choice(self._n_actions)

        return action

    def _trace_mcts_stats(self):
        ac = self._mcts_actor_critic
        q = np.round(ac.cell_value / ac.cell_visited_count, 2)
        value_bit_buckets = (q[i: i + 8] for i in range(0, q.size, 8))
        q_str = '\n'.join(
            ' '.join(map(str, value_bits)) for value_bits in value_bit_buckets
        )
        trace(3, 1, q_str)
        trace(3, 1, '=' * 20)


class MctsAgentRunner:
    agent: MctsAgent
    env: Any
    n_episodes: int
    max_steps: int
    pretrain: int
    verbosity: int
    train_stats: RunStats
    name: str

    def __init__(self, agent, env, n_episodes, max_steps, pretrain, verbosity):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.pretrain = pretrain
        self.verbosity = verbosity
        self.train_stats = RunStats()
        self.name = '???'

    def run(self):
        trace(self.verbosity, 1, '============> RUN HTM AGENT')
        planning_horizon = self.agent.planner.planning_horizon
        if self.pretrain > 0:
            self.agent.set_planning_horizon(0)

        for ep in trange(self.n_episodes):
            if 0 < self.pretrain == ep:
                self.agent.set_planning_horizon(planning_horizon)

            (steps, reward), elapsed_time = self.run_episode()
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
        trace(self.verbosity, 1, '<============')

    def run_iterable(self, iterable):
        trace(self.verbosity, 1, '============> RUN HTM AGENT')
        planning_horizon = self.agent.planner.planning_horizon

        for global_ep, local_ep in iterable():
            if self.pretrain > 0 == local_ep:
                self.agent.set_planning_horizon(0)
            if 0 < self.pretrain == local_ep:
                self.agent.set_planning_horizon(planning_horizon)

            (steps, reward), elapsed_time = self.run_episode()
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
            yield
        trace(self.verbosity, 1, '<============')

    @timed
    def run_episode(self):
        self.agent.reset()
        state, reward, done = self.env.reset(), 0, False
        action = self.agent.make_step(state, reward, done, self.verbosity)

        step = 0
        total_reward = 0.
        while step < self.max_steps and not done:
            state, reward, done, info = self.env.step(action)
            action = self.agent.make_step(state, reward, done, self.verbosity)
            step += 1
            total_reward += reward

        return step, total_reward

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')
