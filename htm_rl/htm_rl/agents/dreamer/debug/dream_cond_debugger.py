from typing import Optional

from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.dreamer.debug.dreaming_point_test_tracker import DreamingPointTestTracker
from htm_rl.agents.q.debug.q_map_provider import QMapProvider
from htm_rl.agents.qmb.debug.anomaly_tracker import AnomalyTracker
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.rnd.debug.env_map_provider import EnvMapProvider
from htm_rl.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.common.debug import inject_debug_tools
from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.env import unwrap as env_unwrap
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.utils import ProgressPoint


class DreamingConditionsDebugger(Debugger):
    env: Environment
    agent: DreamerAgent

    dreaming_test_started: bool
    output_data: dict

    # noinspection PyUnresolvedReferences
    def __init__(self, scenario, images: bool):
        super(DreamingConditionsDebugger, self).__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        self.env_map_provider = EnvMapProvider(scenario)
        self.trajectory_tracker = TrajectoryTracker(scenario)
        self.q_map_provider = QMapProvider(scenario)
        self.anomaly_tracker = AnomalyTracker(scenario)
        self.dreaming_point_test_tracker: Optional[DreamingPointTestTracker] = None
        self.images = images
        self.output_data = {}
        self.dreaming_test_started = False

        # record stats during baseline pre-run
        self.progress.set_breakpoint('end_episode', self.on_end_episode_no_dreaming)

        inject_debug_tools(self.scenario)
        # track transition to the 2nd part of the scenario
        self.scenario.set_breakpoint('init_run', self.on_init_run)

    # noinspection PyUnusedLocal
    def on_end_episode_dreaming(self, pp: ProgressPoint, func, *args, **kwargs):
        cols_per_row = 3
        is_train = self.scenario.mode == 'train'
        if self.images:
            output_train = self.output_data[pp.episode, True]
            if is_train and self.dreaming_test_started:
                self.output_renderer.restore(**output_train)
                self._add_dreaming_point_test()
                output_train = self.output_renderer.flush(
                    filename=None, save_path=output_train['save_path']
                )
            plot_grid_images(show=False, cols_per_row=cols_per_row, **output_train)

            output_eval = self.output_data[pp.episode, False]
            plot_grid_images(show=False, cols_per_row=cols_per_row, **output_eval)

        func(*args, **kwargs)

    # noinspection PyUnusedLocal
    def on_end_episode_no_dreaming(self, pp, func, *args, **kwargs):
        is_train = self.scenario.mode == 'train'
        train_eval_mark = 'A' if is_train else 'Z'

        if self.output_renderer.is_empty and self.images:
            self._add_env_map()
            self._add_value_maps(q=True, v=True)
            self._add_anomaly()
            self._add_trajectory()
            output_filename = self._get_output_filename(pp, train_eval_mark)
            output = self.output_renderer.flush(output_filename)
            self.output_data[pp.episode, is_train] = output

        func(*args, **kwargs)

    # noinspection PyUnusedLocal,PyUnresolvedReferences
    def on_init_run(self, scenario, func, *args, **kwargs):
        func(*args, **kwargs)
        self.env = env_unwrap(scenario.env)
        self.progress = scenario.progress
        # agent is invalid from now on
        self.agent = None

        inject_debug_tools(self.env)
        inject_debug_tools(self.progress)

        # track transition to the test
        self.scenario.set_breakpoint('init_dreaming_test', self.on_init_dreaming_test)
        # track progress
        self.scenario.progress.set_breakpoint('end_episode', self.on_end_episode_dreaming)

    # noinspection PyUnusedLocal,PyUnresolvedReferences
    def on_init_dreaming_test(self, scenario, func, *args, **kwargs):
        func(*args, **kwargs)
        inject_debug_tools(scenario.force_dreaming_point)

        # record single dreaming position test
        self.dreaming_point_test_tracker = DreamingPointTestTracker(scenario)
        # track progress
        self.scenario.progress.unset_breakpoint('end_episode', self.on_end_episode_dreaming)
        self.scenario.force_dreaming_point.set_breakpoint('end_episode', self.on_end_episode_dreaming)
        self.dreaming_test_started = True

    def _add_dreaming_point_test(self):
        self.output_renderer.handle_img(
            self.dreaming_point_test_tracker.dreaming_test_results,
            self.dreaming_point_test_tracker.title,
            with_value_text=True
        )
        self.dreaming_point_test_tracker.reset()

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

    def _get_output_filename(self, pp: ProgressPoint, train_eval_mark):
        config_id = self._default_config_identifier
        pp_id = f'{pp.episode}_{train_eval_mark}_{pp.step}'
        return f'end_episode_{config_id}_{pp_id}'
