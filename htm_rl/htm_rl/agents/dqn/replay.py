import random
from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'mask'])
PrioritizedTransition = namedtuple('Transition',
                                   ['state', 'action', 'reward', 'next_state', 'mask', 'sampling_prob', 'idx'])


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return idx, self.tree[idx], data_idx


class Storage:
    def __init__(self, memory_size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['state', 'action', 'reward', 'mask',
                       'v', 'q', 'pi', 'log_pi', 'entropy',
                       'advantage', 'ret', 'q_a', 'log_pi_a',
                       'mean', 'next_state']
        self.keys = keys
        self.memory_size = memory_size
        self.reset()

    def feed(self, data):
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.memory_size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
        self.pos = 0
        self._size = 0

    def extract(self, keys):
        data = [getattr(self, k)[:self.memory_size] for k in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))


class UniformReplay(Storage):
    TransitionCLS = Transition

    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        super(UniformReplay, self).__init__(memory_size, keys)
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount = discount
        self.history_length = history_length
        self.pos = 0
        self._size = 0

    def compute_valid_indices(self):
        indices = []
        indices.extend(list(range(self.history_length - 1, self.pos - self.n_step)))
        indices.extend(list(range(self.pos + self.history_length - 1, self.size() - self.n_step)))
        return np.asarray(indices)

    def feed(self, data):
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            storage = getattr(self, k)
            pos = self.pos
            size = self.size()
            if pos >= len(storage):
                storage.append(v)
                size += 1
            else:
                storage[self.pos] = v
            pos = (pos + 1) % self.memory_size
        self.pos = pos
        self._size = size

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert self.size() > 0

        sampled_data = []
        while len(sampled_data) < batch_size:
            transition = self.construct_transition(np.random.randint(0, self.size()))
            if transition is not None:
                sampled_data.append(transition)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return Transition(*sampled_data)

    def valid_index(self, index):
        if index - self.history_length + 1 >= 0 and index + self.n_step < self.pos:
            return True
        if index - self.history_length + 1 >= self.pos and index + self.n_step < self.size():
            return True
        return False

    def construct_transition(self, index):
        if not self.valid_index(index):
            return None
        s_start = index - self.history_length + 1
        s_end = index
        if s_start < 0:
            raise RuntimeError('Invalid index')
        next_s_start = s_start + self.n_step
        next_s_end = s_end + self.n_step
        if s_end < self.pos <= next_s_end:
            raise RuntimeError('Invalid index')

        state = [self.state[i] for i in range(s_start, s_end + 1)]
        next_state = [self.state[i] for i in range(next_s_start, next_s_end + 1)]
        action = self.action[s_end]
        reward = [self.reward[i] for i in range(s_end, s_end + self.n_step)]
        mask = [self.mask[i] for i in range(s_end, s_end + self.n_step)]
        if self.history_length == 1:
            # eliminate the extra dimension if no frame stack
            state = state[0]
            next_state = next_state[0]
        state = np.array(state)
        next_state = np.array(next_state)
        cum_r = 0
        cum_mask = 1
        for i in reversed(np.arange(self.n_step)):
            cum_r = reward[i] + mask[i] * self.discount * cum_r
            cum_mask = cum_mask and mask[i]
        return Transition(state=state, action=action, reward=cum_r, next_state=next_state, mask=cum_mask)

    def size(self):
        return self._size

    def full(self):
        return self._size == self.memory_size

    def update_priorities(self, info):
        raise NotImplementedError


class PrioritizedReplay(UniformReplay):
    TransitionCLS = PrioritizedTransition

    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        super(PrioritizedReplay, self).__init__(memory_size, batch_size, n_step, discount, history_length, keys)
        self.tree = SumTree(memory_size)
        self.max_priority = 1

    def feed(self, data):
        super().feed(data)
        self.tree.add(self.max_priority, None)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        segment = self.tree.total() / batch_size

        sampled_data = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)
            transition = super().construct_transition(data_index)
            if transition is None:
                continue
            sampled_data.append(PrioritizedTransition(
                *transition,
                sampling_prob=p / self.tree.total(),
                idx=idx,
            ))

        while len(sampled_data) < batch_size:
            # This should rarely happen
            sampled_data.append(random.choice(sampled_data))

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = PrioritizedTransition(*sampled_data)
        return sampled_data

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
