from hima.agents.q.debug.q_map_provider import QMapProvider
from hima.agents.q.debug.state_encoding_provider import StateEncodingProvider
from hima.agents.qmb.agent import QModelBasedAgent
from hima.agents.qmb.debug.anomaly_map_provider import AnomalyMapProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.agents.rnd.debug.env_map_provider import EnvMapProvider
from hima.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from hima.common.debug import inject_debug_tools
from hima.common.plot_utils import plot_grid_images
from hima.envs.biogwlab.environment import Environment
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario
from hima.scenarios.utils import ProgressPoint


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
