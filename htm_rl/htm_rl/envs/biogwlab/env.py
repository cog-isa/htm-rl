from typing import Tuple, Dict

from htm_rl.envs.biogwlab.areas import Areas
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.episode_terminator import EpisodeTerminator
from htm_rl.envs.biogwlab.food import add_food
from htm_rl.envs.biogwlab.obstacles import Obstacles
from htm_rl.envs.biogwlab.renderer import Renderer
from htm_rl.envs.wrapper import Wrapper

registry = {
    'areas': Areas,
    'obstacles': Obstacles,
    'food': add_food,
    'rendering': Renderer,
    'terminate': EpisodeTerminator,
}


class BioGwLabEnvironment(Wrapper):
    output_sdr_size: int

    def __init__(
            self, shape_xy: Tuple[int, int], seed: int,
            action_costs, actions=None,
            **modules
    ):
        env = Environment(
            shape_xy=shape_xy, seed=seed
        )
        env.set_actions(actions)
        env.set_action_costs(**action_costs)
        # env.set_regenerator(**regenerator)

        add_module(env, modules, 'areas')
        add_module(env, modules, 'obstacles')
        add_module(env, modules, 'food')
        env.add_agent()

        env.reset()
        add_module(env, modules, 'rendering')
        add_module(env, modules, 'terminate')

        super(BioGwLabEnvironment, self).__init__(env)


def add_module(env, modules: Dict, name):
    config = modules.get(name, dict())

    module_type = name
    if module_type not in registry:
        module_type = config['_type_']
        config.pop('_type_')

    module = registry[module_type](env=env, **config)
    env.add_module(name, module)
