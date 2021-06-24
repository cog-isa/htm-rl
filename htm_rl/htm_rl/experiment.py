import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.debug import inject_debug_tools
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env, unwrap
from htm_rl.recorders import (
    BaseOutput,
)


class ProgressPoint:
    step: int
    episode: int

    def __init__(self):
        self.step = 0
        self.episode = 0

    @property
    def is_new_episode(self) -> bool:
        return self.step == 0

    def next_step(self):
        self.step += 1

    def start_new_episode(self):
        self.step = 0
        self.episode += 1


class Experiment:
    config: dict

    n_episodes: int
    print: dict
    wandb: dict

    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(self, config: dict):
        self.config = config

        self.n_episodes = config['n_episodes']
        self.print = config['print']
        self.wandb = config['wandb']

        debug_enabled = self.print['debug'] is not None

        self.env = self.materialize_environment(
            config['env'], config['env_seed'], config['envs'], debug_enabled
        )
        self.agent = self.materialize_agent(
            config['agent'], config['agent_seed'], config['agents'], self.env, debug_enabled
        )

        self.progress = ProgressPoint()

    def run(self):
        env_shape = unwrap(self.env).shape
        #
        # handlers = []
        # if self.print['maps'] is not None:
        #     handlers.append(MapRecorder(self.config, self.n_episodes))
        #
        # if self.print['heatmaps'] is not None:
        #     env = AgentPositionProvider(self.env)
        #     handlers.extend([
        #         HeatmapRecorder(config, freq, env_shape)
        #         for freq in self.print['heatmaps']
        #     ])
        #
        # if self.print['debug'] is not None:
        #     env = AgentPositionProvider(env)
        #     agent = wrap(
        #         agent,
        #         TDErrorProvider,
        #         AnomalyProvider, DreamingLengthProvider,
        #     )
        #     for freq in self.print['debug']:
        #         aggregator = ImageOutput(config, freq)
        #         handlers.extend([
        #             MapRecorder(config, freq, False, aggregator),
        #             HeatmapRecorder(config, freq, env_shape, aggregator),
        #             # ValueMapRecorder(config, freq, env_shape, 'value', aggregator),
        #             # ValueMapRecorder(config, freq, env_shape, 'value_exp', aggregator),
        #             DreamRecorder(config, freq, env_shape, aggregator),
        #             AnomalyMapRecorder(config, freq, env_shape, aggregator),
        #             TDErrorMapRecorder(config, freq, env_shape, aggregator),
        #             DreamingValueChangeProvider(config, freq, agent, env, aggregator),
        #             aggregator
        #         ])

        train_stats = RunStats()
        wandb_run = self.init_wandb_run()

        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode()
            train_stats.append_stats(steps, reward, elapsed_time)
            self.progress.start_new_episode()

            if wandb_run is not None:
                wandb_run.log({
                    'steps': steps,
                    'reward': reward,
                    'elapsed_time': elapsed_time
                })

        return train_stats

    @timed
    def run_episode(self):
        total_reward = 0.

        while True:
            reward, obs, first = self.env.observe()
            if first and self.progress.step > 0:
                break

            action = self.agent.act(reward, obs, first)
            self.env.act(action)

            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward

    @staticmethod
    def materialize_agent(
            name: str, seed: int, agent_configs: dict[str, dict], env: Env,
            debug_enabled: bool
    ) -> Agent:
        agent_config: dict = agent_configs[name]

        agent_type = agent_config['_type_']
        agent_config = filter_out_non_passable_items(agent_config)
        if agent_type == 'rnd':
            return RndAgent(seed=seed, env=env)
        elif agent_type == 'ucb':
            return UcbAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'svpn':
            ctor = inject_debug_tools(SvpnAgent, inject_or_not=debug_enabled)
            return ctor(seed=seed, env=env, **agent_config)
        else:
            raise NameError(agent_type)

    @staticmethod
    def materialize_environment(
            name: str, seed: int, env_configs: dict[str, dict], debug_enabled: bool
    ) -> Env:
        env_config: dict = env_configs[name]

        env_type = env_config['_type_']
        env_config = filter_out_non_passable_items(env_config)
        if env_type == 'biogwlab':
            ctor = inject_debug_tools(BioGwLabEnvironment, inject_or_not=debug_enabled)
            return ctor(seed=seed, **env_config)
        else:
            raise NameError(env_type)

    def init_wandb_run(self):
        if not self.wandb['enabled']:
            return None

        import wandb
        project = self.wandb['project']
        assert project is not None, 'Wandb project name, set by `wandb.project` config field, is missing.'
        run = wandb.init(project=project, reinit=True, dir=self.config['base_dir'])
        run.config.agent = {
            'name': self.config['agent'],
            'type': self.agent.name,
            'seed': self.config['agent_seed']
        }
        run.config.environment = {
            'name': self.config['env'],
            'seed': self.config['env_seed']
        }
        return run

    def handle_episode(
            self, env: Env, agent: Agent, episode: int, handlers: list[BaseOutput]
    ):
        env_info = env.get_info()
        agent_info = agent.get_info()
        for handler in handlers:
            handler.handle_episode(
                env, agent, episode, env_info=env_info, agent_info=agent_info
            )

    def handle_step(
            self, env: Env, agent: Agent, step: int, handlers: list[BaseOutput]
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
