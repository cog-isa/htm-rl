from typing import Optional

from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.dreamer.debug.dreaming_point_test_tracker import DreamingPointTestTracker
from htm_rl.agents.q.debug.q_map_provider import QMapProvider
from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
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

    def __init__(self, scenario, images: bool):
        super(DreamingConditionsDebugger, self).__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        self.env_map_provider = EnvMapProvider(scenario)
        self.trajectory_tracker = TrajectoryTracker(scenario)
        self.state_encoding_provider = StateEncodingProvider(scenario)
        self.q_map_provider = QMapProvider(scenario, self.state_encoding_provider)
        self.anomaly_tracker = AnomalyTracker(scenario, self.state_encoding_provider)
        self.dreaming_point_test_tracker: Optional[DreamingPointTestTracker] = None
        self.images = images
        self.output_data = {}
        self.dreaming_test_started = False

        # record stats during baseline pre-run
        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode_no_dreaming)

        inject_debug_tools(self.scenario)
        # track transition to the 2nd part of the scenario
        # noinspection PyUnresolvedReferences
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
    def on_end_episode_no_dreaming(self, pp: ProgressPoint, func, *args, **kwargs):
        is_train = self.scenario.mode == 'train'
        train_eval_mark = 'A' if is_train else 'Z'

        if self.images:
            self.state_encoding_provider.encoding_scheme = None
            self.env_map_provider.print_map(self.output_renderer)
            self.q_map_provider.print_maps(self.output_renderer, q=True, v=True)
            self.anomaly_tracker.print_map(self.output_renderer)
            self.trajectory_tracker.print_map(self.output_renderer)

            output_filename = self.get_episode_filename(pp, self.train_eval_mark)
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
