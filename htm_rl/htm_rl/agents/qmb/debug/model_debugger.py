from htm_rl.agents.q.debug.q_map_provider import QMapProvider
from htm_rl.agents.q.debug.state_encoding_provider import StateEncodingProvider
from htm_rl.agents.qmb.agent import QModelBasedAgent
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


class ModelDebugger(Debugger):
    env: Environment
    agent: QModelBasedAgent

    def __init__(self, scenario: Scenario, print_images: bool):
        super().__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        self.env_map_provider = EnvMapProvider(scenario)
        self.trajectory_tracker = TrajectoryTracker(scenario)
        self.state_encoding_provider = StateEncodingProvider(scenario)
        self.q_map_provider = QMapProvider(scenario, self.state_encoding_provider)
        self.anomaly_map_provider = AnomalyMapProvider(scenario, self.state_encoding_provider)
        self.print_images = print_images

        inject_debug_tools(self.progress)
        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode)

    # noinspection PyUnusedLocal
    def on_end_episode(self, pp: ProgressPoint, func, *args, **kwargs):
        if self.print_images and self.agent.train:
            output_filename = self.get_episode_filename(pp, self.train_eval_mark)

            self.state_encoding_provider.reset()
            self.env_map_provider.print_map(self.output_renderer, n=1)
            self.q_map_provider.print_maps(self.output_renderer, q=True, v=True)
            self.anomaly_map_provider.print_map(self.output_renderer)
            self.trajectory_tracker.print_map(self.output_renderer)

            output = self.output_renderer.flush(output_filename)
            plot_grid_images(show=False, cols_per_row=3, **output)

        if not self.agent.train:
            self.trajectory_tracker.reset()

        func(*args, **kwargs)
