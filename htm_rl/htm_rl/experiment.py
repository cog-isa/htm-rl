from pathlib import Path
from typing import Optional, Any

import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.svpn.agent import SvpnAgent, ValueRecorder, TDErrorRecorder, AnomalyRecorder
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.plot_utils import store_environment_map, plot_grid_images
from htm_rl.common.utils import timed, ensure_list
from htm_rl.config import FileConfig
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.biogwlab.wrappers.recorders import AgentPositionRecorder
from htm_rl.envs.env import Env, unwrap


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

        if self.print['maps'] is not None:
            store_environment_map(
                0, env.callmethod('render_rgb'),
                config['env'], config['env_seed'], config['results_dir']
            )
        handlers = []
        if self.print['heatmaps'] is not None:
            env = AgentPositionRecorder(env)
            heatmap_recorders = [
                HeatmapRecorder(config, freq, env_shape)
                for freq in self.print['heatmaps']
            ]
            handlers.extend(heatmap_recorders)

        if self.print['debug'] is not None:
            env = AgentPositionRecorder(env)
            agent = ValueRecorder(agent)
            agent = TDErrorRecorder(agent)
            agent = AnomalyRecorder(agent)
            for freq in self.print['debug']:
                aggregator = AggregateRecorder(config, freq)
                handlers.extend([
                    MapRecorder(config, freq, aggregator),
                    HeatmapRecorder(config, freq, env_shape, aggregator),
                    ValueMapRecorder(config, freq, env_shape, aggregator),
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

    def handle_episode(self, env: Env, agent: Agent, episode: int, handlers: list):
        for handler in handlers:
            handler.handle_episode(env, agent, episode, )

    def handle_step(self, env: Env, agent: Agent, step: int, handlers: list):
        for handler in handlers:
            handler.handle_step(env, agent, step, )


class BaseRecorder:
    config: FileConfig
    save_dir: Path
    aggregator: Optional[Any]

    def __init__(self, config: FileConfig, aggregator=None):
        self.config = config
        self.save_dir = self.config['results_dir']
        self.aggregator = aggregator

    def handle_step(self, env: Env, agent: Agent, step: int):
        raise NotImplementedError

    def handle_episode(self, env: Env, agent: Agent, episode: int):
        raise NotImplementedError


class AggregateRecorder(BaseRecorder):
    images: list[np.ndarray]
    titles: list[str]
    name_str: str

    def __init__(self, config: FileConfig, frequency):
        super().__init__(config)

        self.name_str = f'debug_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.images = []
        self.titles = []

    def handle_step(self, env: Env, agent: Agent, step: int):
        self._flush_images(step + 1)

    def handle_episode(self, env: Env, agent: Agent, episode: int):
        self._flush_images(episode + 1)

    def handle_img(self, image: np.ndarray, title: str = None):
        self.images.append(image.copy())
        if title is not None:
            self.titles.append(title)

    def _flush_images(self, ind: int):
        if not self.images:
            return
        save_path = self.save_dir.joinpath(f'{self.name_str}_{ind}.svg')
        plot_grid_images(
            images=self.images, titles=self.titles,
            show=False, save_path=save_path
        )
        self.images.clear()
        self.titles.clear()


class HeatmapRecorder(BaseRecorder):
    heatmap: np.ndarray
    frequency: int
    name_str: str

    def __init__(self, config: FileConfig, frequency: int, shape, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'heatmap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.heatmap = np.zeros(shape, dtype=np.float)

    def handle_step(self, env: Env, agent: Agent, step: int):
        position: tuple[int, int] = env.get_info()['agent_position']
        self.heatmap[position] += 1.

    def handle_episode(self, env: Env, agent: Agent, episode: int):
        if (episode + 1) % self.frequency == 0:
            self._flush_heatmap(episode + 1)

    def _flush_heatmap(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            self.aggregator.handle_img(self.heatmap, plot_title)
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            plot_grid_images(
                images=self.heatmap, titles=plot_title,
                show=False, save_path=save_path
            )
        # clear heatmap
        self.heatmap.fill(0.)


class ValueMapRecorder(BaseRecorder):
    fill_value: float = -1.

    value_map: np.ndarray
    frequency: int
    name_str: str
    last_value: Optional[float]

    def __init__(self, config: FileConfig, frequency: int, shape, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'valuemap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.value_map = np.full(shape, self.fill_value, dtype=np.float)
        self.last_value = None

    def handle_step(self, env: Env, agent: Agent, step: int):
        position: tuple[int, int] = env.get_info()['agent_position']
        if self.last_value is not None:
            self.value_map[position] = self.last_value

        self.last_value = agent.get_info()['value']

    def handle_episode(self, env: Env, agent: Agent, episode: int):
        if (episode + 1) % self.frequency == 0:
            self._flush_map(episode + 1)

    def _flush_map(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            self.aggregator.handle_img(self.value_map, plot_title)
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            plot_grid_images(
                images=self.value_map, titles=plot_title,
                show=False, save_path=save_path
            )
        # clear heatmap
        self.value_map.fill(self.fill_value)


class MapRecorder(BaseRecorder):
    env_map: Optional[np.ndarray]
    frequency: int
    name_str: str
    titles: list[str]

    def __init__(self, config: FileConfig, frequency: int, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'map_{config["env"]}_{config["env_seed"]}'
        self.titles = [
            f'{config["env"]}, seed={config["env_seed"]}',
            'agent observation'
        ]
        self.env_map = None

    def handle_step(self, env: Env, agent: Agent, step: int):
        if self.env_map is None:
            self.env_map = ensure_list(env.callmethod('render_rgb'))

    def handle_episode(self, env: Env, agent: Agent, episode: int):
        if (episode + 1) % self.frequency == 0:
            self._flush_map()

    def _flush_map(self):
        if self.env_map is None:
            return

        if self.aggregator is not None:
            for i in range(len(self.env_map)):
                self.aggregator.handle_img(self.env_map[i], self.titles[i])
        else:
            save_path = self.save_dir.joinpath(f'map_{self.name_str}.svg')
            plot_grid_images(self.env_map, self.titles, show=False, save_path=save_path)


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
