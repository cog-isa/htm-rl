import numpy as np
from numpy.random._generator import Generator
from tqdm import trange

from htm_rl.agents.ucb.ucb_agent_runner import UcbAgentRunner
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnv, BioGwLabEnvDynamics, BioGwLabEnvObservationWrapper
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator


class UcbExperimentRunner:
    rnd: Generator
    env_generator: BioGwLabEnvGenerator
    n_episodes_all_fixed: int
    n_initial_states: int
    n_environments: int
    verbosity: int

    def __init__(
            self, env_generator: BioGwLabEnvGenerator, n_episodes_all_fixed: int,
            n_initial_states: int, n_terminal_states: int,
            n_environments: int, verbosity: int
    ):
        self.env_generator = env_generator
        self.n_episodes_all_fixed = n_episodes_all_fixed
        self.n_initial_states = n_initial_states
        self.n_terminal_states = n_terminal_states
        self.n_environments = n_environments
        self.rnd = np.random.default_rng(seed=env_generator.seed)
        self.verbosity = verbosity

    def run_experiment(self, agent_runner: UcbAgentRunner):
        n_episodes = self.n_environments * self.n_terminal_states * self.n_initial_states
        n_episodes *= self.n_episodes_all_fixed

        zipped_iterator = lambda: zip(trange(n_episodes), self._get_local_iterator(agent_runner))
        for _ in agent_runner.run_iterable(zipped_iterator):
            ...

    def _get_local_iterator(self, agent_runner: UcbAgentRunner):
        env: BioGwLabEnv
        dynamics = BioGwLabEnvDynamics()
        view_rect = (-2, 0), (2, 2)
        scent_rect = (-3, -2), (3, 4)

        for i in range(self.n_environments):
            seed = self.rnd.integers(100000)
            self.env_generator.seed = seed
            state = self.env_generator.generate()
            env = BioGwLabEnvObservationWrapper(state, dynamics, view_rect, scent_rect)
            agent_runner.env = env
            for _ in range(self.n_initial_states):
                env.generate_initial_position()
                for local_episode in range(self.n_episodes_all_fixed):
                    yield local_episode
