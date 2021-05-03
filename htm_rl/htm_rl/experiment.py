from pathlib import Path
from typing import Dict

from tqdm import trange
import numpy as np

from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.plot_utils import store_environment_map
from htm_rl.common.utils import timed
from htm_rl.config import Config
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env
from htm_rl.envs.wrapper import unwrap
from htm_rl.envs.wrappers.recorders import HeatmapRecorder


class Experiment:
    n_episodes: int
    use_wandb: bool
    base_dir: Path
    project: str

    def __init__(self, n_episodes: int, project: str, base_dir: Path, use_wandb: bool = False):
        self.n_episodes = n_episodes
        self.base_dir = base_dir
        self.use_wandb = use_wandb
        self.project = project

    def run(
            self, env_seed: int, agent_seed: int,
            agent_config: Config, env_config: Config,
            run_results_processor: RunResultsProcessor = None, seed_ind=None,
    ):
        env = self.materialize_environment(env_seed, env_config)
        agent = self.materialize_agent(agent_seed, env, agent_config)
        train_stats = RunStats(agent.name)

        print(f'AGENT: {agent.name}     SEED: {env_seed} {agent_seed}')
        if run_results_processor is not None:
            store_environment_map(
                seed_ind, env.callmethod('render_rgb'),
                env_config.name, env_seed, run_results_processor.test_dir
            )
            # env = HeatmapRecorder(
            #     env, 25, run_results_processor.test_dir,
            #     f'{agent_config.name}_{env_seed}'
            # )

        # temporal dirty flag-based solution
        run = None
        if self.use_wandb:
            import wandb
            assert self.project is not None, 'Either set up `project` to the experiment, or turn off wandb'
            run = wandb.init(
                project=self.project, reinit=True, dir=self.base_dir
            )
            run.config.agent = agent_config.name
            run.config.env = env_config.name
            run.config.environment = {'seed': env_seed}
            run.config.seed = agent_seed

        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode(env, agent)
            train_stats.append_stats(steps, reward, elapsed_time)
            if self.use_wandb:
                run.log({
                    'steps': steps,
                    'reward': reward,
                    'elapsed_time': elapsed_time
                })

        return train_stats

    @timed
    def run_episode(self, env, agent):
        step = 0
        total_reward = 0.

        while True:
            reward, obs, first = env.observe()

            # from htm_rl.common.plot_utils import plot_grid_images
            # plot_grid_images(unwrap(env).render_rgb())

            if first and step > 0:
                break

            action = agent.act(reward, obs, first)
            env.act(action)

            step += 1
            total_reward += reward

        return step, total_reward

    @staticmethod
    def materialize_agent(seed: int, env: Env, config: Dict) -> Agent:
        agent_type = config['type']

        if agent_type == 'rnd':
            return RndAgent(seed=seed, env=env)
        elif agent_type == 'ucb':
            agent_config = config['agent']
            return UcbAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'svpn':
            agent_config = config['agent']
            return SvpnAgent(seed=seed, env=env, **agent_config)
        else:
            raise NameError(agent_type)

    @staticmethod
    def materialize_environment(seed: int, env_config: Dict) -> Env:
        return BioGwLabEnvironment(seed=seed, **env_config)
