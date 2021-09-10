import numpy as np
from numpy import ma

from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.envs.env import unwrap as env_unwrap
from htm_rl.scenarios.standard.scenario import Scenario


class DreamingPointTestTracker(Debugger):
    fill_value: float = 0.
    name_prefix = 'dreaming_test'
    dreaming_test_results: ma.MaskedArray

    # noinspection PyUnresolvedReferences,PyMissingConstructor
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.env = env_unwrap(scenario.env)
        self.progress = scenario.progress

        self.dreaming_test_results = ma.masked_all(
            self.env.shape, dtype=np.int
        )
        self.scenario.set_breakpoint('restore_checkpoint', self.on_restore_checkpoint)

    def on_restore_checkpoint(self, scenario, restore_checkpoint, *args, **kwargs):
        restore_checkpoint(*args, **kwargs)
        position, result = scenario.test_results_map[scenario.force_dreaming_point]
        res = -int(result)
        if res > 0:
            res *= 100

        if not self.dreaming_test_results.mask[position]:
            # keep max value on a map
            res = ma.max((self.dreaming_test_results[position], res))
        self.dreaming_test_results[position] = res

    def reset(self):
        self.dreaming_test_results.mask[:] = True

    @property
    def title(self) -> str:
        return self.name_prefix

    @property
    def filename(self) -> str:
        return f'{self.name_prefix}_{self._default_config_identifier}_{self._default_progress_identifier}'
