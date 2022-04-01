import numpy as np
from numpy import ma

from hima.agents.dreamer.agent import DreamerAgent
from hima.agents.dreamer.debug.dreaming_trajectory_tracker import DreamingTrajectoryTracker
from hima.agents.q.debug.state_encoding_provider import StateEncodingProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.common.debug import remove_debug_tools_from_obj
from hima.envs.biogwlab.environment import Environment
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario


class DreamingPointTestTracker(Debugger):

    env: Environment
    agent: DreamerAgent
    fill_value: float = 0.
    name_prefix = 'dreaming_test'
    dreaming_test_results: ma.MaskedArray

    # noinspection PyMissingConstructor
    def __init__(
            self, scenario: Scenario, state_encoding_provider: StateEncodingProvider,
            print_images: bool
    ):
        super(DreamingPointTestTracker, self).__init__(scenario)
        self.dreaming_test_results = ma.masked_all(
            self.env.shape, dtype=np.int
        )
        self.dreaming_trajectory_tracker = DreamingTrajectoryTracker(
            scenario, state_encoding_provider
        )
        self.renderer = None
        if print_images:
            self.renderer = ImageOutput(self.scenario.config)
        # noinspection PyUnresolvedReferences
        self.scenario.set_breakpoint('save_checkpoint', self.on_save_checkpoint)
        # noinspection PyUnresolvedReferences
        self.scenario.set_breakpoint('restore_checkpoint', self.on_restore_checkpoint)

    # noinspection PyUnusedLocal
    def on_save_checkpoint(self, scenario, save_checkpoint, *args, **kwargs):
        remove_debug_tools_from_obj(self.agent.dreamer)
        save_checkpoint(*args, **kwargs)
        self.dreaming_trajectory_tracker.inject_debug_tools_to_dreamer()

    def on_restore_checkpoint(self, scenario, restore_checkpoint, *args, **kwargs):
        restore_checkpoint(*args, **kwargs)
        position, result = scenario.test_results_map[scenario.force_dreaming_point]
        res = -int(result)
        if res > 0:
            res *= 100
            if self.renderer is not None:
                self.dreaming_trajectory_tracker.print_map(self.renderer, min_traj=1)

        self.dreaming_trajectory_tracker.reset()

        if not self.dreaming_test_results.mask[position]:
            # keep max value on a map
            res = ma.max((self.dreaming_test_results[position], res))
        self.dreaming_test_results[position] = res
        self.dreaming_trajectory_tracker.inject_debug_tools_to_dreamer()

    def reset(self):
        self.dreaming_test_results.mask[:] = True

    @property
    def title(self) -> str:
        return self.name_prefix

    def print_map(self, renderer: ImageOutput):
        renderer.handle_img(
            self.dreaming_test_results, self.title, with_value_text=True
        )
        self.reset()
        if self.renderer is not None and not self.renderer.is_empty:
            renderer.restore(**self.renderer.flush())
