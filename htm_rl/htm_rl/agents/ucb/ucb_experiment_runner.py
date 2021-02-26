from tqdm import trange

from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.agents.ucb.ucb_agent import UcbAgent
from htm_rl.common.utils import trace
from htm_rl.envs.biogwlab.environment import BioGwLabEnvironment


class UcbExperimentRunner:
    n_episodes: int
    verbosity: int
    train_stats: RunStats
    name: str

    def __init__(self, n_episodes: int, verbosity: int):
        self.n_episodes = n_episodes
        self.verbosity = verbosity
        self.train_stats = RunStats()
        self.name = 'ucb'

    def run_experiment(self, env_config, agent_config):
        # view_rect = (-2, 0), (2, 2)
        # scent_rect = (-3, -2), (3, 4)
        env = BioGwLabEnvironment(**env_config)
        agent = UcbAgent(env, **agent_config)

        trace(self.verbosity, 1, '============> RUN UCB AGENT')
        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = agent.run_episode(env, self.verbosity)
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')

        trace(self.verbosity, 1, '<============')

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')

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
