from typing import List, Optional, NamedTuple, TypeVar, Generic
from dataclasses import dataclass, astuple

from utils import isnone


_S = TypeVar('S')
_A = TypeVar('A')
_R = TypeVar('R')


@dataclass(frozen=True)
class SarRelatedComposition(Generic[_S, _A, _R]):
    """
    Represents a composition of objects related to (state, action, reward).
    """
    __slots__ = ['state', 'action', 'reward']
    state: _S
    action: _A
    reward: _R

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    def __getitem__(self, item: int):
        return astuple(self)[item]


@dataclass(frozen=True)
class BaseSar(SarRelatedComposition[
    Optional[_S],
    Optional[_A],
    Optional[_R]
]):
    """
    Represents a composition of (state, action, reward) aka sar. Every part is optional, e.g. (s, a, None)
    """
    pass

Superposition = List[int]
