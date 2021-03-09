from typing import Tuple

from htm_rl.envs.biogwlab.areas_generator import AreasGenerator
from htm_rl.envs.biogwlab.environment_state import EnvironmentState
from htm_rl.envs.biogwlab.food import add_food
from htm_rl.envs.biogwlab.obstacles_generator import ObstaclesGenerator
from htm_rl.envs.biogwlab.renderer import Renderer

registrar = {
    'areas': AreasGenerator,
    'obstacles': ObstaclesGenerator,
    'food': add_food,
    'rendering': Renderer,
}


class BioGwLabEnvironment:
    output_sdr_size: int
    state: EnvironmentState

    def __init__(
            self, shape_xy: Tuple[int, int], seed: int,
            action_costs, regenerator, actions=None,
            **modules
    ):
        state = EnvironmentState(
            shape_xy=shape_xy, seed=seed
        )
        state.set_actions(actions)
        state.set_action_costs(**action_costs)
        state.set_regenerator(**regenerator)

        BioGwLabEnvironment.add_module(state, modules, 'areas')
        BioGwLabEnvironment.add_module(state, modules, 'obstacles')
        BioGwLabEnvironment.add_module(state, modules, 'food')
        state.add_agent()

        # print(state.modules)
        # print(state.handlers)
        state.reset()

        BioGwLabEnvironment.add_module(state, modules, 'rendering')

        self.state = state
        sdr = state.render()
        # print(sdr)

    @staticmethod
    def add_module(env, modules, name):
        config = modules[name]
        module_type = name
        if module_type not in registrar:
            module_type = config['_type_']
            config.pop('_type_')

        module = registrar[module_type](env=env, **config)
        env.add_module(name, module)

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
