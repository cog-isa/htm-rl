from pathlib import Path
from typing import Optional, Any

import numpy as np

from htm_rl.envs.biogwlab.agent import Agent as EnvAgent
from htm_rl.agents.agent import Agent, unwrap as agent_unwrap
from htm_rl.envs.env import unwrap as env_unwrap
from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.ucb.sparse_value_network import update_slice_lin_sum, update_slice_exp_sum
from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.common.utils import ensure_list
from htm_rl.config import FileConfig
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import EntityType
from htm_rl.envs.env import Env


class BaseRecorder:
    config: FileConfig
    save_dir: Path
    aggregator: Optional[Any]

    def __init__(self, config: FileConfig, aggregator=None):
        self.config = config
        self.save_dir = self.config['results_dir']
        self.aggregator = aggregator

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        raise NotImplementedError

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
        raise NotImplementedError

    @staticmethod
    def env_info(env_info: dict, env: Env) -> dict:
        if env_info is None:
            return env.get_info()
        return env_info

    @staticmethod
    def agent_info(agent_info: dict, agent: Agent) -> dict:
        if agent_info is None:
            return agent.get_info()
        return agent_info


class AggregateRecorder(BaseRecorder):
    images: list[np.ndarray]
    titles: list[str]
    name_str: str

    def __init__(self, config: FileConfig, frequency):
        super().__init__(config)

        self.name_str = f'debug_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.images = []
        self.titles = []

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        self._flush_images(step + 1)

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
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

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        env_info = self.env_info(env_info, env)
        position: tuple[int, int] = env_info['agent_position']
        self.heatmap[position] += 1.

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
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


class AnomalyMapRecorder(BaseRecorder):
    anomaly_map: np.ndarray
    frequency: int
    name_str: str

    def __init__(self, config: FileConfig, frequency: int, shape, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'anomaly_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.anomaly_map = np.ones(shape, dtype=np.float)

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        env_info = self.env_info(env_info, env)
        agent_info = self.agent_info(agent_info, agent)

        position: tuple[int, int] = env_info['agent_position']
        self.anomaly_map[position] = agent_info['anomaly']

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
        if (episode + 1) % self.frequency == 0:
            self._flush_map(episode + 1)

    def _flush_map(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            self.aggregator.handle_img(self.anomaly_map, plot_title)
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            plot_grid_images(
                images=self.anomaly_map, titles=plot_title,
                show=False, save_path=save_path
            )

        # self.anomaly_map.fill(1.)


class DreamRecorder(BaseRecorder):
    dream_heatmap: np.ndarray
    frequency: int
    name_str: str

    def __init__(self, config: FileConfig, frequency: int, shape, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'dream_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.dream_heatmap = np.zeros(shape, dtype=np.float)

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        env_info = self.env_info(env_info, env)
        agent_info = self.agent_info(agent_info, agent)

        position: tuple[int, int] = env_info['agent_position']
        dream_length: Optional[int] = agent_info['dream_length']
        if dream_length is not None:
            self.dream_heatmap[position] += dream_length

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
        if (episode + 1) % self.frequency == 0:
            self._flush_heatmap(episode + 1)

    def _flush_heatmap(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            self.aggregator.handle_img(self.dream_heatmap, plot_title)
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            plot_grid_images(
                images=self.dream_heatmap, titles=plot_title,
                show=False, save_path=save_path
            )
        # clear heatmap
        self.dream_heatmap.fill(0.)


class ValueMapRecorder(BaseRecorder):
    fill_value: float = -1.

    value_map: np.ndarray
    frequency: int
    name_str: str
    value_key: str
    last_value: Optional[float]

    def __init__(self, config: FileConfig, frequency: int, shape, key='value', aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'valuemap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.value_map = np.full(shape, self.fill_value, dtype=np.float)
        self.value_key = key
        self.last_value = None

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        env_info = self.env_info(env_info, env)
        agent_info = self.agent_info(agent_info, agent)

        position: tuple[int, int] = env_info['agent_position']
        if self.last_value is not None:
            self.value_map[position] = self.last_value

        self.last_value = agent_info[self.value_key]

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
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


# noinspection PyProtectedMember
class DreamingValueChangeProvider(BaseRecorder):
    fill_value: float = 0.

    frequency: int
    name_str: str

    agent: SvpnAgent
    env: Env
    value_maps: list[tuple[np.ndarray, tuple[int, int]]]

    def __init__(
            self, config: FileConfig, frequency: int,
            agent: SvpnAgent, env: Env, aggregator=None
    ):
        super().__init__(config, aggregator)
        self.agent = agent_unwrap(agent)
        self.env = env_unwrap(env)
        self.value_maps = []

        self.agent.set_breakpoint('_reset_dreaming', self.snapshot_value_map)
        # self.agent.set_breakpoint('_wake', self.snapshot_value_map)

        self.frequency = frequency
        self.name_str = f'valuemap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        pass

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
        if (episode + 1) % self.frequency == 0:
            if not self.value_maps:
                self.value_maps.append(self.construct_value_map())
            self._flush_maps(episode + 1)

    def _flush_maps(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            for value_map, agent_pos in self.value_maps:
                self.aggregator.handle_img(value_map, f'{plot_title}_{agent_pos}')
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            value_maps, plot_titles = [], []
            for value_map, agent_pos in self.value_maps:
                value_maps.append(value_map)
                plot_titles.append(f'{plot_title}_{agent_pos}')

            plot_grid_images(images=value_maps, titles=plot_title, show=False, save_path=save_path)

        self.value_maps.clear()

    def snapshot_value_map(self, agent, func, *args, **kwargs):
        value_map, agent_pos = self.construct_value_map()
        if self.value_maps:
            for vm, p in self.value_maps:
                if p == agent_pos:
                    value_map -= vm
                    break

        self.value_maps.append((value_map, agent_pos))
        func(*args, **kwargs)

    def construct_value_map(self):
        env: Environment = env_unwrap(self.env)
        value_map = np.full(env.shape, self.fill_value, dtype=np.float)
        env_agent: EnvAgent = env.entities['agent']
        agent_state = env_agent.position, env_agent.view_direction
        obstacles_mask = env.aggregated_mask[EntityType.Obstacle]

        for i in range(env.shape[0]):
            for j in range(env.shape[1]):
                pos = (i, j)
                if obstacles_mask[pos]:
                    continue
                env_agent.position = pos
                obs = env.render()
                s = self.agent.state_sp.compute(obs, learn=False)
                actions_sa_sdr = self.agent._encode_actions(s, learn=False)
                greedy_action, action_values = self.agent.sqvn.choose(actions_sa_sdr, greedy=True)
                value_map[pos] = action_values[greedy_action]

        env_agent.position, env_agent.view_direction = agent_state
        return value_map, env_agent.position


class TDErrorMapRecorder(BaseRecorder):
    fill_value: float = 0.

    td_error_map: np.ndarray
    frequency: int
    name_str: str
    last_td_error: Optional[float]

    def __init__(self, config: FileConfig, frequency: int, shape, aggregator=None):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'tderrormap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}_{frequency}'
        self.td_error_map = np.full(shape, self.fill_value, dtype=np.float)
        self.last_td_error = None

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        env_info = self.env_info(env_info, env)
        agent_info = self.agent_info(agent_info, agent)

        position: tuple[int, int] = env_info['agent_position']
        if self.last_td_error is not None:
            update_slice_exp_sum(self.td_error_map, position, .7, self.last_td_error)

        self.last_td_error = agent_info['td_error']

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
        if (episode + 1) % self.frequency == 0:
            self._flush_map(episode + 1)

    def _flush_map(self, episode):
        plot_title = f'{self.name_str}_{episode}'
        if self.aggregator is not None:
            self.aggregator.handle_img(self.td_error_map, plot_title)
        else:
            save_path = self.save_dir.joinpath(f'{plot_title}.svg')
            plot_grid_images(
                images=self.td_error_map, titles=plot_title,
                show=False, save_path=save_path
            )
        self.td_error_map.fill(0.)


class MapRecorder(BaseRecorder):
    env_map: Optional[np.ndarray]
    frequency: int
    name_str: str
    titles: list[str]
    include_observation: bool

    def __init__(
            self, config: FileConfig, frequency: int, include_observation=False,
            aggregator=None
    ):
        super().__init__(config, aggregator)
        self.frequency = frequency
        self.name_str = f'map_{config["env"]}_{config["env_seed"]}'
        self.titles = [
            f'{config["env"]}, seed={config["env_seed"]}',
            'agent observation'
        ]
        self.include_observation = False
        self.env_map = None

    def handle_step(
            self, env: Env, agent: Agent, step: int,
            env_info: dict = None, agent_info: dict = None
    ):
        if self.env_map is None:
            self.env_map = ensure_list(env.callmethod('render_rgb'))
            if len(self.env_map) > 1 and not self.include_observation:
                self.env_map = self.env_map[:1]

    def handle_episode(
            self, env: Env, agent: Agent, episode: int,
            env_info: dict = None, agent_info: dict = None
    ):
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