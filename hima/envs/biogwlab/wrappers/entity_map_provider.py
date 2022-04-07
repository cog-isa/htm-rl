import numpy as np

from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import EntityType
from hima.envs.env import Wrapper


class EntityMapProvider(Wrapper):
    root_env: Environment

    entities: dict[EntityType, np.array]

    def __init__(self, entities: dict[EntityType, np.array], env):
        super().__init__(env)
        self.entities = entities

    def get_info(self) -> dict:
        info = super(EntityMapProvider, self).get_info()

        mask_map = dict()
        for entity, entity_flag in self.entities.items():
            mask_map[entity] = self.root_env.aggregated_mask[entity_flag]

        info['map'] = mask_map
        return info