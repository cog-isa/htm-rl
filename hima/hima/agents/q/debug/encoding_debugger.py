from hima.agents.q.agent import QAgent
from hima.agents.q.debug.encoding_tracker import EncodingTracker
from hima.agents.rnd.debug.debugger import Debugger
from hima.agents.rnd.debug.env_map_provider import EnvMapProvider
from hima.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from hima.common.plot_utils import plot_grid_images
from hima.envs.biogwlab.environment import Environment
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario
from hima.scenarios.utils import ProgressPoint


class EncodingDebugger(Debugger):
    env: Environment
    agent: QAgent

    def __init__(self, scenario: Scenario, images: bool):
        super().__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        # self.env_map_provider = EnvMapProvider(scenario)
        # self.trajectory_tracker = TrajectoryTracker(scenario)
        self.encoding_tracker = EncodingTracker(scenario)
        self.print_images = images

        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode)

    # noinspection PyUnusedLocal
    def on_end_episode(self, pp: ProgressPoint, func, *args, **kwargs):
        if self.print_images and self.agent.train and pp.episode + 1 == self.scenario.n_episodes:
            output_filename = self.get_episode_filename(pp, self.train_eval_mark)
            # self.env_map_provider.print_map(self.output_renderer, n=1)
            # self.trajectory_tracker.print_map(self.output_renderer)
            self.encoding_tracker.print_plot(self.output_renderer)

            output = self.output_renderer.flush(filename=output_filename)
            plot_grid_images(show=False, cols_per_row=3, **output)

        func(*args, **kwargs)
