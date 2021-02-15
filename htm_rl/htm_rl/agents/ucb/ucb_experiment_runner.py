from typing import Any

import numpy as np
from numpy.random._generator import Generator
from tqdm import trange

from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.agents.ucb.ucb_agent import UcbAgent
from htm_rl.common.utils import trace
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnv, BioGwLabEnvDynamics, BioGwLabEnvObservationWrapper
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator


class UcbExperimentRunner:
    rnd: Generator
    env_generator: BioGwLabEnvGenerator
    n_episodes_all_fixed: int
    n_initial_states: int
    n_environments: int
    verbosity: int

    env: Any
    n_episodes: int
    max_steps: int
    pretrain: int
    verbosity: int
    train_stats: RunStats
    name: str

    def __init__(
            self, env_generator: BioGwLabEnvGenerator, n_episodes_all_fixed: int,
            n_initial_states: int, n_terminal_states: int,
            n_environments: int, verbosity: int, max_steps
    ):
        self.env_generator = env_generator
        self.n_episodes_all_fixed = n_episodes_all_fixed
        self.n_initial_states = n_initial_states
        self.n_terminal_states = n_terminal_states
        self.n_environments = n_environments
        self.rnd = np.random.default_rng(seed=env_generator.seed)
        self.verbosity = verbosity

        self.max_steps = max_steps
        self.verbosity = verbosity
        self.train_stats = RunStats()
        self.name = 'ucb'

    def run_experiment(self, agent_config):
        agent = UcbAgent(**agent_config)

        n_episodes = self.n_environments * self.n_terminal_states * self.n_initial_states
        n_episodes *= self.n_episodes_all_fixed

        zipped_iterator = lambda: zip(trange(n_episodes), self._traverse_episodes())
        for _ in self.run_iterable(zipped_iterator, agent):
            ...

    def run_iterable(self, iterable, agent):
        trace(self.verbosity, 1, '============> RUN UCB AGENT')

        for _, (env, _) in iterable():
            (steps, reward), elapsed_time = agent.run_episode(
                env, self.max_steps, self.verbosity
            )
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
            yield

        trace(self.verbosity, 1, '<============')

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')

    def _traverse_episodes(self):
        env: BioGwLabEnv
        dynamics = BioGwLabEnvDynamics()
        view_rect = (-2, 0), (2, 2)
        scent_rect = (-3, -2), (3, 4)

        for i in range(self.n_environments):
            seed = self.rnd.integers(100000)
            self.env_generator.seed = seed
            state = self.env_generator.generate()
            env = BioGwLabEnvObservationWrapper(state, dynamics, view_rect, scent_rect)
            for _ in range(self.n_initial_states):
                env.generate_initial_position()
                for local_episode in range(self.n_episodes_all_fixed):
                    yield env, local_episode

    def show_environments(self, n):
        # for env_map, _ in self.get_environment_maps(n):
        #     plt.imshow(env_map)
        #     plt.show(block=True)
        ...

    def get_environment_maps(self, n):
        # env: GridworldMdp
        # maps = []
        # for env in islice(self.env_generator, self.n_environments):
        #     for _ in range(self.n_terminal_states):
        #         env.set_random_terminal_state()
        #         # generate just one initial state instead of self.n_initial_states
        #         env.set_random_initial_state()
        #         env.reset()
        #         maps.append(env.get_representation(mode='img'))
        #         if len(maps) >= n:
        #             return maps
        # return maps
        ...
