from collections import namedtuple, defaultdict, deque
from functools import partial
from typing import Deque

import numpy as np
import torch

from htm_rl.common.utils import isnone


class DequeStorage(defaultdict):
    def __init__(self, max_len):
        super().__init__(partial(deque, maxlen=max_len))

    def feed(self, data) -> None:
        for k, v in data.items():
            self[k].append(v)

    @property
    def _one_deque(self) -> Deque:
        return next(iter(self.values()))

    @property
    def sub_size(self):
        return len(self._one_deque)

    @property
    def is_full(self):
        def contains_full_deque():
            one_deque = self._one_deque
            return len(one_deque) == one_deque.maxlen

        return len(self) > 0 and contains_full_deque()

    def extract(self, keys=None, indices=None):
        def fetch_items(key):
            dq = self[key]
            if indices is None:
                res = list(dq)
            else:
                res = [dq[ix] for ix in indices]
            return res

        keys = isnone(keys, self.keys())
        result = []

        for k in keys:
            a = fetch_items(k)
            if not isinstance(a[0], torch.Tensor):
                if isinstance(a[0], np.ndarray):
                    a = np.vstack(a)
                t = torch.Tensor(a)
            elif a[0].dim() == 0:
                t = torch.stack(a)
            else:
                t = torch.cat(a, dim=0)
            result.append(t)

        Entry = namedtuple('Entry', keys)
        return Entry(*result)

    def reset(self) -> None:
        for d in self.values():
            d.clear()

    def pop_one_left(self) -> None:
        for d in self.values():
            d: Deque
            d.popleft()
