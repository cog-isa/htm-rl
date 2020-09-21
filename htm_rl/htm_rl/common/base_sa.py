from dataclasses import dataclass, astuple
from typing import Optional, TypeVar, Generic, List

T = TypeVar('T')
TS = TypeVar('TS')
TA = TypeVar('TA')


@dataclass(frozen=True)
class SaRelatedComposition(Generic[TS, TA]):
    """
    Represents a composition of objects related to (state, action).
    """
    __slots__ = ['state', 'action']
    state: TS
    action: TA

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    def __getitem__(self, item: int):
        return astuple(self)[item]


@dataclass(frozen=True)
class Sa(SaRelatedComposition[
    Optional[TS],
    Optional[TA],
]):
    """
    Represents a composition of (state, action) aka sa.
    Every part is optional, e.g. (s, None)
    """
    pass


Superposition = List[T]
SaSuperposition = SaRelatedComposition[Superposition, Superposition]
