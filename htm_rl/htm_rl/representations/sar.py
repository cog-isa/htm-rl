from typing import List, Optional, NamedTuple, TypeVar, Generic
from dataclasses import dataclass, astuple

from utils import isnone


TS = TypeVar('TS')
TA = TypeVar('TA')
TR = TypeVar('TR')


@dataclass(frozen=True)
class SarRelatedComposition(Generic[TS, TA, TR]):
    """
    Represents a composition of objects related to (state, action, reward).
    """
    __slots__ = ['state', 'action', 'reward']
    state: TS
    action: TA
    reward: TR

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    def __getitem__(self, item: int):
        return astuple(self)[item]


@dataclass(frozen=True)
class BaseSar(SarRelatedComposition[
    Optional[TS],
    Optional[TA],
    Optional[TR]
]):
    """
    Represents a composition of (state, action, reward) aka sar. Every part is optional, e.g. (s, a, None)
    """
    pass

Superposition = List[int]
