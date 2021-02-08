from itertools import islice
from typing import Any

import numpy as np
from numpy.random._generator import Generator
from tqdm import trange

from htm_rl.agent.mcts_actor_critic import MctsActorCritic
from htm_rl.agent.mcts_planner import MctsPlanner
from htm_rl.agent.memory import Memory
from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.agents.ucb.ucb_agent import UcbAgent
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import timed, trace
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnv, BioGwLabEnvDynamics, BioGwLabEnvObservationWrapper
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator
from htm_rl.envs.gridworld_pomdp import GridworldPomdp


class UcbAgentRunner:
    agent: UcbAgent
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
        trace(self.verbosity, 1, '============> RUN MCTS AGENT')
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
        trace(self.verbosity, 1, '============> RUN MCTS AGENT')
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
        state, reward, done = self.env.reset(), 0, self.env.is_terminal()
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
