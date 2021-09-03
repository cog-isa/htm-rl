from htm_rl.agents.q.debug.q_map_provider import QMapProvider
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.agents.qmb.debug.anomaly_tracker import AnomalyTracker
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.rnd.debug.env_map_provider import EnvMapProvider
from htm_rl.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario


class ModelDebugger(Debugger):
    env: Environment
    agent: QModelBasedAgent

    def __init__(self, experiment: Scenario, images: bool):
        super().__init__(experiment)

        self.output_renderer = ImageOutput(experiment.config)
        self.env_map_provider = EnvMapProvider(experiment)
        self.trajectory_tracker = TrajectoryTracker(experiment)
        self.q_map_provider = QMapProvider(experiment)
        self.anomaly_tracker = AnomalyTracker(experiment)
        self.images = images

        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode)

    # noinspection PyUnusedLocal
    def on_end_episode(self, agent, func, *args, **kwargs):
        if self.output_renderer.is_empty and self.images:
            self._add_env_map()
            self._add_value_maps(q=True, v=True)
            self._add_anomaly()
            self._add_trajectory()
            self.output_renderer.flush(
                f'end_episode_{self._default_config_identifier}_{self._default_progress_identifier}'
            )

        func(*args, **kwargs)

    def _add_anomaly(self):
        an_tr = self.anomaly_tracker
        self.output_renderer.handle_img(
            an_tr.heatmap, an_tr.title, with_value_text=True
        )
        an_tr.reset()

    def _add_trajectory(self):
        self.output_renderer.handle_img(
            self.trajectory_tracker.heatmap, self.trajectory_tracker.title, with_value_text=True
        )
        self.trajectory_tracker.reset()

    # noinspection PyPep8Naming
    def _add_value_maps(self, q: bool, v: bool):
        self.q_map_provider.precompute()
        Q = self.q_map_provider.Q
        V = self.q_map_provider.V(Q)
        if v:
            self.output_renderer.handle_img(V, 'V', with_value_text=True)
        if q:
            Q_render = self.q_map_provider.reshape_q_for_rendering(Q)
            self.output_renderer.handle_img(Q_render, 'Q', with_value_text=False)

    def _add_env_map(self):
        env_maps = self.env_map_provider.maps
        env_map_titles = self.env_map_provider.titles
        for i in range(1):
            self.output_renderer.handle_img(env_maps[i], env_map_titles[i])
