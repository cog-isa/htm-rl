from typing import Any

from htm_rl.envs.biogwlab.agent import Agent
from htm_rl.envs.biogwlab.area import MultiAreaGenerator
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.modules.episode_terminator import EpisodeTerminator
from htm_rl.envs.biogwlab.food import Food
from htm_rl.envs.biogwlab.modules.actions_cost import ActionsCost
from htm_rl.envs.biogwlab.obstacle import Obstacle, BorderObstacle
from htm_rl.envs.biogwlab.modules.regenerator import Regenerator
from htm_rl.envs.biogwlab.renderer import Renderer
from htm_rl.envs.env import Wrapper

registry = {
    'areas': MultiAreaGenerator,
    'rendering': Renderer,
    'terminate': EpisodeTerminator,
    'regenerate': Regenerator,
    'actions_cost': ActionsCost,

    Obstacle.family: Obstacle,
    BorderObstacle.family: BorderObstacle,
    Food.family: Food,
    Agent.family: Agent,
}


class BioGwLabEnvironment(Wrapper):
    def __init__(
            self, shape_xy: tuple[int, int], seed: int,
            rendering: dict = None, actions=None,
            **modules
    ):
        env = Environment(
            shape_xy=shape_xy, seed=seed, actions=actions,
            rendering=rendering
        )

        default_modules = [
            Agent.family, 'regenerate', 'terminate'
        ]
        for module_name in default_modules:
            append_module(module_name, modules)

        for module_name, module_config in modules.items():
            add_module(env, module_name, module_config)

        env.reset()
        super(BioGwLabEnvironment, self).__init__(env)


def append_module(name: str, modules: dict[str, Any]):
    if name not in modules or modules[name] is None:
        modules[name] = {}


def add_module(env, name: str, config: dict):
    module_type = name
    if module_type not in registry:
        if '_type_' not in config:
            raise KeyError({'name': name, 'config': config})
        module_type = config['_type_']
        config.pop('_type_')

    assert config is not None, f'Config for "{name}" is None'
    module = registry[module_type](env=env, name=name, **config)
    env.add_module(module)
