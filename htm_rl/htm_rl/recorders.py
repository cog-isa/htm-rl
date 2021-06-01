from pathlib import Path
from typing import Optional, Any

import numpy as np

from htm_rl.agents.agent import Agent
from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.common.utils import ensure_list
from htm_rl.config import FileConfig
from htm_rl.envs.env import Env


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