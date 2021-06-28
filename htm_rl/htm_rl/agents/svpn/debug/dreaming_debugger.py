from htm_rl.agents.svpn.agent import SvpnAgent
from htm_rl.agents.svpn.debug.debugger import Debugger
from htm_rl.agents.svpn.debug.providers import ValueMapProvider, EnvMapProvider
from htm_rl.agents.svpn.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.envs.biogwlab.environment import Environment

from htm_rl.experiment import Experiment
from htm_rl.recorders import ImageOutput


class DreamingDebugger(Debugger):
    env: Environment
    agent: SvpnAgent

    waking: bool

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self.waking = False
        self.output_renderer = ImageOutput(experiment.config)
        self.env_map_provider = EnvMapProvider(experiment)
        self.trajectory_tracker = TrajectoryTracker(experiment)
        self.value_map_provider = ValueMapProvider(experiment)

        self.progress.set_breakpoint('end_episode', self.on_end_episode)
        self.agent.set_breakpoint('_wake', self.on_wake)
        self.agent.set_breakpoint('_reset_dreaming', self.on_reset_dreaming)

    def on_reset_dreaming(self, agent, func, *args, **kwargs):
        if self.waking:
            # waking from dreaming
            pass

        # starting new rollout

    def on_wake(self, agent, func, *args, **kwargs):
        self.waking = True
        func(*args, **kwargs)
        self.waking = False

    def on_end_episode(self, agent, func, *args, **kwargs):
        if self.output_renderer.is_empty:
            self._add_env_map()
            self._add_value_map(greedy=True)
            self._add_value_map(greedy=False)
            self._add_trajectory()
            self.output_renderer.flush(
                f'end_episode_{self._default_config_identifier}_{self._default_progress_identifier}'
            )

        func(*args, **kwargs)

    def _add_trajectory(self):
        self.output_renderer.handle_img(
            self.trajectory_tracker.heatmap, self.trajectory_tracker.title, with_value_text=True
        )
        self.trajectory_tracker.reset()

    def _add_value_map(self, greedy):
        q_map = self.value_map_provider.get_q_value_map(greedy)
        q_map *= 10
        v_map = self.value_map_provider.get_value_map(q_map)
        title = 'V-func' if greedy else 'V-func-ucb'
        self.output_renderer.handle_img(v_map, title, with_value_text=True)

        q_map_render = self.value_map_provider.get_q_value_map_for_rendering(q_map)
        title = 'Q-func' if greedy else 'Q-func-ucb'
        self.output_renderer.handle_img(q_map_render, title, with_value_text=False)

    def _add_env_map(self):
        env_maps = self.env_map_provider.maps
        env_map_titles = self.env_map_provider.titles
        for i in range(1):
            self.output_renderer.handle_img(env_maps[i], env_map_titles[i])
