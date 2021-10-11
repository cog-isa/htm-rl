from typing import Union

import numpy as np

from htm_rl.common.utils import ensure_list
from htm_rl.envs.biogwlab.module import Entity, EntityType

TEntityKey = Union[str, EntityType]
TEntityValue = Union[Entity, list[Entity]]
TEntityDict = dict[TEntityKey, TEntityValue]


class EntityDict(TEntityDict):
    def __getitem__(self, key: TEntityKey) -> TEntityValue:
        if isinstance(key, str):
            return super(EntityDict, self).__getitem__(key)
        elif isinstance(key, EntityType):
            return self._filter_by_flag(key)
        else:
            raise KeyError()

    def __setitem__(self, key: str, value: Entity):
        # assert only single items are added and only by str key
        assert isinstance(key, str) and isinstance(value, Entity)
        super(EntityDict, self).__setitem__(key, value)

    def _filter_by_flag(self, flag: EntityType) -> list[Entity]:
        return [
            entity
            for entity in self.values()
            if entity.type & flag
        ]


class EntityMaskAggregation(dict[TEntityKey, np.ndarray]):
    _shape: tuple[int, int]
    _entities: TEntityDict

    def __init__(self, entities: TEntityDict, shape: tuple[int, int]):
        super().__init__()
        self._entities = entities
        self._shape = shape

    def __getitem__(self, key: TEntityKey) -> np.ndarray:
        # one or many entities could be queried
        entities = ensure_list(self._entities[key])

        # aggregate mask
        mask = np.zeros(self._shape, dtype=np.bool)
        for entity in entities:
            # IMPORTANT: only initialized entities are "valid" !
            if entity.initialized:
                entity.append_mask(mask)

        return mask
