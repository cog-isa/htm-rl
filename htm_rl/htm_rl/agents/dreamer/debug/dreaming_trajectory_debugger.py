from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.dreamer.debug.dreaming_heatmap_tracker import DreamingHeatmapTracker
from htm_rl.agents.dreamer.debug.dreaming_trajectory_tracker import DreamingTrajectoryTracker
from htm_rl.agents.dreamer.dreaming_double import DreamingDouble
from htm_rl.agents.q.debug.q_map_provider import QMapProvider
from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
from htm_rl.agents.qmb.debug.anomaly_map_provider import AnomalyMapProvider
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.rnd.debug.env_map_provider import EnvMapProvider
from htm_rl.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.common.debug import inject_debug_tools
from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario
from htm_rl.scenarios.utils import ProgressPoint


class DreamingTrajectoryDebugger(Debugger):
    env: Environment
    agent: DreamerAgent

    def __init__(self, scenario: Scenario, images: bool):
        super().__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        self.env_map_provider = EnvMapProvider(scenario)
        self.trajectory_tracker = TrajectoryTracker(scenario)
        self.state_encoding_provider = StateEncodingProvider(scenario)
        self.q_map_provider = QMapProvider(scenario, self.state_encoding_provider)
        self.anomaly_map_provider = AnomalyMapProvider(scenario, self.state_encoding_provider)
        self.dreaming_heatmap_tracker = DreamingHeatmapTracker(scenario)
        self.dreaming_trajectory_tracker = DreamingTrajectoryTracker(scenario)
        self.print_images = images

        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode)

        dreamer = self.agent.dreamer
        inject_debug_tools(dreamer)
        # noinspection PyUnresolvedReferences
        self.agent.dreamer.set_breakpoint('_on_new_rollout', self.on_new_rollout)
        # noinspection PyUnresolvedReferences
        self.agent.dreamer.set_breakpoint('_wake', self.on_new_rollout)

    def on_new_rollout(self, dreaming_double: DreamingDouble, func, *args, **kwargs):
        func(*args, **kwargs)
        if self.print_images:
            self.dreaming_trajectory_tracker.print_map(self.output_renderer)

    # noinspection PyUnusedLocal
    def on_end_episode(self, pp: ProgressPoint, func, *args, **kwargs):
        if self.print_images:
            self.state_encoding_provider.reset()
            self.env_map_provider.print_map(self.output_renderer, n=1)
            self.q_map_provider.print_maps(self.output_renderer, q=True, v=True)
            self.anomaly_map_provider.print_map(self.output_renderer)
            self.trajectory_tracker.print_map(self.output_renderer)
            self.dreaming_heatmap_tracker.print_map(self.output_renderer)

            output_filename = self.get_episode_filename(pp, self.train_eval_mark)
            output = self.output_renderer.flush(output_filename)
            plot_grid_images(show=False, cols_per_row=3, **output)

        func(*args, **kwargs)
