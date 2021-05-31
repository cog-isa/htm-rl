from pathlib import Path

import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.utils import timed
from htm_rl.config import FileConfig
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env
from htm_rl.envs.wrappers.recorders import HeatmapRecorder


class Experiment:
    n_episodes: int
    print: dict
    wandb: dict
    base_dir: Path

    def __init__(self, n_episodes: int, print: dict, wandb: dict, base_dir: Path, **_):
        self.n_episodes = n_episodes
        self.print = print
        self.wandb = wandb
        self.base_dir = base_dir

    def run(self, config: FileConfig):
        env = self.materialize_environment(config)
        agent = self.materialize_agent(config, env)
        # if self.print['maps']:
        #     store_environment_map(
        #         seed_ind, env.callmethod('render_rgb'),
        #         env_config.name, env_seed, run_results_processor.test_dir
        #     )
        if self.print['heatmaps']:
            env = HeatmapRecorder(
                env, 1, self.base_dir,
                f'{config["agent"]}_{config["env_seed"]}'
            )

        train_stats = RunStats()
        wandb_run = self.init_wandb_run(agent, config) if self.wandb['enabled'] else None

        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode(env, agent)
            train_stats.append_stats(steps, reward, elapsed_time)
            if self.wandb['enabled']:
                wandb_run.log({
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
    def materialize_agent(config: FileConfig, env: Env) -> Agent:
        agent_name = config['agent']
        seed: int = config['agent_seed']
        agent_config: dict = config['agents'][agent_name]

        agent_type = agent_config['_type_']
        agent_config = filter_out_non_passable_items(agent_config)
        if agent_type == 'rnd':
            return RndAgent(seed=seed, env=env)
        elif agent_type == 'ucb':
            return UcbAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'svpn':
            return SvpnAgent(seed=seed, env=env, **agent_config)
        else:
            raise NameError(agent_type)

    @staticmethod
    def materialize_environment(config: FileConfig) -> Env:
        env_name = config['env']
        seed: int = config['env_seed']
        env_config: dict = config['envs'][env_name]

        env_type = env_config['_type_']
        env_config = filter_out_non_passable_items(env_config)
        if env_type == 'biogwlab':
            return BioGwLabEnvironment(seed=seed, **env_config)
        else:
            raise NameError(env_type)

    def init_wandb_run(self, agent, config):
        import wandb
        project = self.wandb['project']
        assert project is not None, 'Either set up `project` to the experiment, or turn off wandb'
        run = wandb.init(project=project, reinit=True, dir=self.base_dir)
        run.config.agent = {'name': config['agent'], 'type': agent.name, 'seed': config['agent_seed']}
        run.config.environment = {'name': config['env'], 'seed': config['env_seed']}
        return run


class RunStats:
    steps: list[int]
    rewards: list[float]
    times: list[float]

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.times = []

    def append_stats(self, steps, total_reward, elapsed_time):
        self.steps.append(steps)
        self.rewards.append(total_reward)
        self.times.append(elapsed_time)

    def print_results(self):
        avg_len = np.array(self.steps).mean()
        avg_reward = np.array(self.rewards).mean()
        avg_time = np.array(self.times).mean()
        elapsed = np.array(self.times).sum()
        print(
            f'AvgLen: {avg_len: .2f}  AvgReward: {avg_reward: .5f}  '
            f'AvgTime: {avg_time: .6f}  TotalTime: {elapsed: .6f}'
        )


def filter_out_non_passable_items(config: dict):
    """Filters out non-passable args started with '.' and '_'."""
    return {
        k: v
        for k, v in config.items()
        if not (k.startswith('.') or k.startswith('_'))
    }
