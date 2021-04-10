from enum import Flag
from typing import Any, Dict, List, Tuple

import numpy as np

from htm_rl.envs.biogwlab.module import Entity


class CachedFlagDict(dict):
    _base_dict: Dict[str, Entity]

    def __init__(self, base_dict: Dict[str, Entity]):
        super().__init__()
        self._base_dict = base_dict

    def flush(self):
        self.clear()

    def __getitem__(self, flag: Flag):
        if flag not in self:
            self[flag] = self._find_items(flag)
        return self[flag]

    def _find_items(self, flag: Flag):
        return [
            entity
            for entity in self._base_dict.values()
            if entity.type & flag
        ]


class CachedEntityAggregation(dict):
    _shape: Tuple[int, int]
    _flag_dict: Dict[Flag, np.ndarray]

    def __init__(self, entities: Dict[str, Entity], shape: Tuple[int, int]):
        super().__init__()
        self._flag_dict = CachedFlagDict(entities)
        self._shape = shape

    def flush(self):
        self.clear()
        self._flag_dict.clear()

    def __getitem__(self, flag: Flag):
        if flag not in self:
            self[flag] = self._aggregate_entities(flag)
        return self[flag]

    def _aggregate_entities(self, flag: Flag):
        mask = np.zeros(self._shape, dtype=np.uint8)
        for entity in self._flag_dict[flag]:
            entity.append_mask(mask)
        return mask


