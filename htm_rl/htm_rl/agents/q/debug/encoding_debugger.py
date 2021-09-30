from htm_rl.agents.q.agent import QAgent
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.rnd.debug.env_map_provider import EnvMapProvider
from htm_rl.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.debug_output import ImageOutput
from htm_rl.scenarios.standard.scenario import Scenario
from htm_rl.scenarios.utils import ProgressPoint


class EncodingDebugger(Debugger):
    env: Environment
    agent: QAgent

    def __init__(self, scenario: Scenario, images: bool):
        super().__init__(scenario)

        self.output_renderer = ImageOutput(scenario.config)
        self.env_map_provider = EnvMapProvider(scenario)
        self.trajectory_tracker = TrajectoryTracker(scenario)
        self.print_images = images

        # noinspection PyUnresolvedReferences
        self.progress.set_breakpoint('end_episode', self.on_end_episode)
        # noinspection PyUnresolvedReferences
        self.agent.set_breakpoint('act', self.on_act)

    def on_act(self, agent: QAgent, act, *args, **kwargs):
        if self.print_images:
            self.dreaming_trajectory_tracker.print_map(self.output_renderer)
        return act(*args, **kwargs)

    # noinspection PyUnusedLocal
    def on_end_episode(self, pp: ProgressPoint, func, *args, **kwargs):
        if self.print_images:
            output_filename = self.get_episode_filename(pp, self.train_eval_mark)
            self.env_map_provider.print_map(self.output_renderer, n=1)
            self.trajectory_tracker.print_map(self.output_renderer)

            output = self.output_renderer.flush(filename=output_filename)
            plot_grid_images(show=False, cols_per_row=3, **output)

        func(*args, **kwargs)
