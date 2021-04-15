from typing import Dict, List, Tuple

import numpy as np

from htm_rl.envs.biogwlab.module import Entity, EntityType


class CachedFlagDict(dict):
    _base_dict: Dict[str, Entity]

    def __init__(self, base_dict: Dict[str, Entity]):
        super().__init__()
        self._base_dict = base_dict

    def flush(self):
        self.clear()

    def __getitem__(self, flag: EntityType):
        if True or flag not in self:
            self[flag] = self._find_items(flag)
        return super(CachedFlagDict, self).__getitem__(flag)

    def _find_items(self, flag: EntityType):
        return [
            entity
            for entity in self._base_dict.values()
            if entity.type & flag
        ]


class CachedEntityAggregation(dict):
    _shape: Tuple[int, int]
    _flag_dict: Dict[EntityType, List[Entity]]

    def __init__(
            self, flag_dict: Dict[EntityType, List[Entity]],
            shape: Tuple[int, int]
    ):
        super().__init__()
        self._flag_dict = flag_dict
        self._shape = shape

    def flush(self):
        self.clear()
        self._flag_dict.clear()

    def __getitem__(self, flag: EntityType):
        if True or flag not in self:
            self[flag] = self._aggregate_entities(flag)
        return super(CachedEntityAggregation, self).__getitem__(flag)

    def _aggregate_entities(self, flag: EntityType):
        mask = np.zeros(self._shape, dtype=np.bool)
        for entity in self._flag_dict[flag]:
            if entity.initialized:
                entity.append_mask(mask)
        return mask


