import numpy as np

from htm_rl.agents.svpn.debug.providers import AgentStateProvider
from htm_rl.experiment import Experiment


class HeatmapTracker(AgentStateProvider):
    fill_value: float = 0.

    heatmap: np.ndarray
    frequency: int
    name_str: str

    def __init__(self, experiment: Experiment):
        super(HeatmapTracker, self).__init__(experiment)

        config = self.experiment.config
        self.name_str = f'heatmap_{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}'
        env_shape = self.env.shape
        self.heatmap = np.full(env_shape, self.fill_value, dtype=np.float)

        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent, act, *args, **kwargs):
        position, view_direction = self.state
        self.heatmap[position] += 1
        act(*args, **kwargs)

    def reset(self):
        self.heatmap.fill(self.fill_value)

    @property
    def title(self) -> str:
        return self.filename

    @property
    def filename(self) -> str:
        episode = self.progress.episode
        filename = f'{self.name_str}_{episode}'
        if not self.progress.is_new_episode:
            step = self.progress.step
            filename = f'{filename}_{step}'
        return filename