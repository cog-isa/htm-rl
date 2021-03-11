from typing import Dict

from tqdm import trange

from htm_rl.agent.train_eval import RunStats
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.agent import Agent
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env


class Experiment:
    n_episodes: int

    def __init__(self, n_episodes: int):
        self.n_episodes = n_episodes

    def run(self, seed: int, agent_config: Dict, env_config: Dict):
        env = self.materialize_environment(seed, env_config)
        agent = self.materialize_agent(seed, env, agent_config)
        train_stats = RunStats(agent.name)

        print(f'AGENT: {agent.name}     SEED: {seed}')

        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode(env, agent)
            train_stats.append_stats(steps, reward, elapsed_time)

        return train_stats

    @timed
    def run_episode(self, env, agent):
        step = 0
        total_reward = 0.
        first = False

        while not (first and step > 0):
            reward, obs, first = env.observe()
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
