from typing import Dict, Tuple

from htm_rl.envs.biogwlab.environment_state import EnvironmentState


class BioGwLabEnvironment:
    output_sdr_size: int
    state: EnvironmentState

    def __init__(self, shape_xy: Tuple[int, int], seed: int, **environment):
        state = EnvironmentState(
            shape_xy=shape_xy, seed=seed
        )
        state.set_actions(environment['actions'])
        state.set_action_costs(**environment['action_costs'])
        state.set_areas(**environment.get('areas', dict()))
        state.set_obstacles(**environment['obstacles'])
        state.set_food(**environment['food'])
        state.set_regenerator(**environment['regenerator'])

        state.generate_areas()
        state.generate_obstacles()
        state.generate_food()

        state.set_rendering(**environment['rendering'])
        state.spawn_agent()

        self.state = state

    def observe(self):
        return self.state.observe()

    def act(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        if self.state.is_terminal():
            self.state.reset()
            return

        self.state.act(action)

    @property
    def n_actions(self):
        return len(self.state.actions)

    @property
    def output_sdr_size(self):
        return self.state.output_sdr_size
