from typing import List, Optional, NamedTuple, TypeVar, Generic
from dataclasses import dataclass, astuple

from utils import isnone


State = TypeVar('State')
ActionReward = TypeVar('ActionReward')


@dataclass(frozen=True)
class Sar(Generic[State, ActionReward]):
    """
    Represents a combination of state, action, reward aka (s, a, r).

    Every part is optional, e.g. (s, a, None)
    """

    #
    __slots__ = ['state', 'action', 'reward']
    state: Optional[State]
    action: Optional[ActionReward]
    reward: Optional[ActionReward]

    # is needed for tuple unpacking
    def __iter__(self):
        yield from astuple(self)

    def __getitem__(self, item: int):
        return astuple(self)[item]
