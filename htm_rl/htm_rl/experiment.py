import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.svpn.agent import SvpnAgent, ValueProvider, TDErrorProvider, AnomalyProvider, DreamingLengthProvider
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.utils import timed, wrap
from htm_rl.config import FileConfig
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.biogwlab.wrappers.recorders import AgentPositionProvider
from htm_rl.envs.env import Env, unwrap
from htm_rl.recorders import (
    AggregateRecorder, HeatmapRecorder, ValueMapRecorder, MapRecorder, DreamRecorder,
    BaseRecorder, AnomalyMapRecorder, TDErrorMapRecorder,
)


class Experiment:
    n_episodes: int
    print: dict
    wandb: dict

    def __init__(self, n_episodes: int, print: dict, wandb: dict, **_):
        self.n_episodes = n_episodes
        self.print = print
        self.wandb = wandb

    def run(self, config: FileConfig):
        env = self.materialize_environment(config)
        agent = self.materialize_agent(config, env)
        env_shape = unwrap(env).shape

        handlers = []
        if self.print['maps'] is not None:
            handlers.append(MapRecorder(config, self.n_episodes))

        if self.print['heatmaps'] is not None:
            env = AgentPositionProvider(env)
            handlers.extend([
                HeatmapRecorder(config, freq, env_shape)
                for freq in self.print['heatmaps']
            ])

        if self.print['debug'] is not None:
            env = AgentPositionProvider(env)
            agent = wrap(
                agent, ValueProvider, TDErrorProvider,
                AnomalyProvider, DreamingLengthProvider
            )
            for freq in self.print['debug']:
                aggregator = AggregateRecorder(config, freq)
                handlers.extend([
                    MapRecorder(config, freq, False, aggregator),
                    HeatmapRecorder(config, freq, env_shape, aggregator),
                    ValueMapRecorder(config, freq, env_shape, 'value', aggregator),
                    # ValueMapRecorder(config, freq, env_shape, 'value_exp', aggregator),
                    DreamRecorder(config, freq, env_shape, aggregator),
                    AnomalyMapRecorder(config, freq, env_shape, aggregator),
                    TDErrorMapRecorder(config, freq, env_shape, aggregator),
                    aggregator
                ])

        train_stats = RunStats()
        wandb_run = self.init_wandb_run(agent, config) if self.wandb['enabled'] else None

        for episode in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode(env, agent, handlers)
            self.handle_episode(env, agent, episode, handlers)
            train_stats.append_stats(steps, reward, elapsed_time)
            if self.wandb['enabled']:
                wandb_run.log({
                    'steps': steps,
                    'reward': reward,
                    'elapsed_time': elapsed_time
                })

        return train_stats

    @timed
    def run_episode(self, env: Env, agent: Agent, handlers: list):
        step = 0
        total_reward = 0.

        while True:
            reward, obs, first = env.observe()
            if first and step > 0:
                break

            action = agent.act(reward, obs, first)
            self.handle_step(env, agent, step, handlers)
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
        run = wandb.init(project=project, reinit=True, dir=config['base_dir'])
        run.config.agent = {'name': config['agent'], 'type': agent.name, 'seed': config['agent_seed']}
        run.config.environment = {'name': config['env'], 'seed': config['env_seed']}
        return run

    def handle_episode(
            self, env: Env, agent: Agent, episode: int, handlers: list[BaseRecorder]
    ):
        env_info = env.get_info()
        agent_info = agent.get_info()
        for handler in handlers:
            handler.handle_episode(
                env, agent, episode, env_info=env_info, agent_info=agent_info
            )

    def handle_step(
            self, env: Env, agent: Agent, step: int, handlers: list[BaseRecorder]
    ):
        env_info = env.get_info()
        agent_info = agent.get_info()
        for handler in handlers:
            handler.handle_step(
                env, agent, step, env_info=env_info, agent_info=agent_info
            )


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
