from typing import List, Any, Tuple

import numpy as np

from htm_rl.common.plot_utils import store_heatmap
from htm_rl.common.sdr import SparseSdr
from htm_rl.envs.env import Env
from htm_rl.envs.wrapper import Wrapper


class EpisodeLengthRecorder(Wrapper):
    episode_lengths: List[int]

    _current_length: int

    def __init__(self, env):
        super(EpisodeLengthRecorder, self).__init__(env)
        self._current_length = 0
        self.episode_lengths = []

    def act(self, action: Any) -> None:
        super(EpisodeLengthRecorder, self).act(action)
        self._current_length += 1

        _, _, first = super(EpisodeLengthRecorder, self).observe()
        if first:
            self.episode_lengths.append(self._current_length)
            self._current_length = 0


class HeatmapRecorder(Wrapper):
    heatmap: np.ndarray
    heatmap_ind: int
    current_episode: int
    n_episodes: int
    test_dir: str
    name_str: str
    acted: bool

    def __init__(self, env: Wrapper, n_episodes, test_dir, name_str):
        super(HeatmapRecorder, self).__init__(env.env)
        self.heatmap = np.zeros(env.env.shape, dtype=np.float)
        self.heatmap_ind = 0
        self.current_episode = 0
        self.n_episodes = n_episodes
        self.test_dir = test_dir
        self.name_str = name_str
        self.acted = True

    def act(self, action: Any) -> None:
        super(HeatmapRecorder, self).act(action)
        self.acted = True

    def observe(self) -> Tuple[float, SparseSdr, bool]:
        r, obs, first = super(HeatmapRecorder, self).observe()

        agent_pos = self.env.agent.position
        self.heatmap[agent_pos] += 1.

        if first and self.acted:
            self.current_episode += 1
            if self.current_episode == self.n_episodes:
                self._flush_heatmap()

        self.acted = False
        return r, obs, first

    def _flush_heatmap(self):
        self.heatmap_ind += self.n_episodes
        store_heatmap(
            self.heatmap_ind, self.heatmap,
            self.name_str, self.test_dir
        )
        self.current_episode = 0
        self.heatmap.fill(0.)


