from pathlib import Path
from typing import Dict

from tqdm import trange

from htm_rl.agent.train_eval import RunStats
from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.utils import timed
from htm_rl.config import Config
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env


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

    def run(self, seed: int, agent_config: Config, env_config: Config):
        env = self.materialize_environment(seed, env_config)
        agent = self.materialize_agent(seed, env, agent_config)
        train_stats = RunStats(agent.name)

        print(f'AGENT: {agent.name}     SEED: {seed}')

        # temporal dirty flag-based solution
        run = None
        if self.use_wandb:
            import wandb
            run = wandb.init(
                project=self.project, reinit=True, dir=self.base_dir
            )
            run.config.agent = agent_config.name
            run.config.env = env_config.name
            run.config.seed = seed

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
        else:
            raise NameError(agent_type)

    @staticmethod
    def materialize_environment(seed: int, env_config: Dict) -> Env:
        return BioGwLabEnvironment(seed=seed, **env_config)
