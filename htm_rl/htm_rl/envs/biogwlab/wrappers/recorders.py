import numpy as np

from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import EntityType
from htm_rl.envs.env import Wrapper


class AgentPositionRecorder(Wrapper):
    root_env: Environment

    def get_info(self) -> dict:
        info = super(AgentPositionRecorder, self).get_info()
        info['agent_position'] = self.root_env.agent.position
        return info


class EntityMapRecorder(Wrapper):
    root_env: Environment

    entities: dict[EntityType, np.array]

    def __init__(self, entities: dict[EntityType, np.array], env):
        super().__init__(env)
        self.entities = entities

    def get_info(self) -> dict:
        info = super(EntityMapRecorder, self).get_info()

        mask_map = dict()
        for entity, entity_flag in self.entities.items():
            mask_map[entity] = self.root_env.aggregated_mask[entity_flag]

        info['map'] = mask_map
        return info
