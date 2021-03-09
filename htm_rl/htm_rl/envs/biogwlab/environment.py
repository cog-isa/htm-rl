from typing import Dict, Tuple

from htm_rl.envs.biogwlab.areas import add_areas
from htm_rl.envs.biogwlab.areas_generator import AreasGenerator
from htm_rl.envs.biogwlab.environment_state import EnvironmentState


registrar = {
    'areas': add_areas,
    'areas_generator': AreasGenerator,
}


class BioGwLabEnvironment:
    output_sdr_size: int
    state: EnvironmentState

    def __init__(self, shape_xy: Tuple[int, int], seed: int, **modules):
        state = EnvironmentState(
            shape_xy=shape_xy, seed=seed
        )

        supported_modules = {'areas'}
        for module_name, module in modules.items():
            if module_name not in supported_modules:
                continue

            module_type = module_name
            if module_type not in registrar:
                module_type = module['_type_']
                module.remove('_type_')

            module = registrar[module_type](env=state, **module)
            state.add_module(module_name, module)

        print(state.modules)
        print(state.handlers)
        return

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
