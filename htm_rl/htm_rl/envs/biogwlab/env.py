from typing import Tuple, Dict, Any

from htm_rl.envs.biogwlab.agent import Agent
from htm_rl.envs.biogwlab.area import MultiAreaGenerator
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.episode_terminator import EpisodeTerminator
from htm_rl.envs.biogwlab.food import Food
from htm_rl.envs.biogwlab.modules.actions_cost import ActionsCost
from htm_rl.envs.biogwlab.obstacle import Obstacle, BorderObstacle
from htm_rl.envs.biogwlab.regenerator import Regenerator
from htm_rl.envs.biogwlab.renderer import Renderer
from htm_rl.envs.wrapper import Wrapper

registry = {
    'areas': MultiAreaGenerator,
    'obstacle': Obstacle,
    'food': Food,
    'rendering': Renderer,
    'terminate': EpisodeTerminator,
    'regenerate': Regenerator,
    'actions_cost': ActionsCost,

    Agent.family: Agent,
    BorderObstacle.family: BorderObstacle,
}


class BioGwLabEnvironment(Wrapper):
    output_sdr_size: int

    def __init__(
            self, shape_xy: Tuple[int, int], seed: int, actions=None,
            **modules
    ):
        env = Environment(
            shape_xy=shape_xy, seed=seed, actions=actions
        )

        default_modules = [
            Agent.family, BorderObstacle.family, 'regenerate'
        ]
        for module_name in default_modules:
            append_module(module_name, modules)

        for module_name, module_config in modules.items():
            add_module(env, module_name, module_config)

        # add_module(env, modules, 'terminate')

        env.reset()
        # add_module(env, modules, 'rendering')

        super(BioGwLabEnvironment, self).__init__(env)


def append_module(name: str, modules: Dict[str, Any]):
    if name not in modules:
        modules[name] = {}


def add_module(env, name: str, config: Dict):
    module_type = name
    if module_type not in registry:
        if '_type_' not in config:
            raise KeyError({'name': name, 'config': config})
        module_type = config['_type_']
        config.pop('_type_')

    module = registry[module_type](env=env, name=name, **config)
    env.add_module(module)
