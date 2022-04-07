from typing import Optional, Iterator

import numpy as np
from numpy.random import Generator

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.scenarios.utils import which_type


class Policy:
    id: int
    _policy: np.ndarray

    def __init__(self, id: int, policy, seed=None):
        self.id = id
        self._policy = policy

    def __iter__(self) -> Iterator[tuple[SparseSdr, SparseSdr]]:
        return iter(self._policy)

    def shuffle(self) -> None:
        ...


class SyntheticGenerator:
    n_states: int
    n_actions: int

    policy_similarity: float
    _rng: Generator

    def __init__(
            self, config: dict,
            n_states: int, n_actions: int,
            state_encoder: str, action_encoder: str,
            policy_similarity: float, seed: int
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_encoder = resolve_encoder(config, state_encoder, 'state_encoders')
        self.action_encoder = resolve_encoder(config, action_encoder, 'action_encoders')

        self.policy_similarity = policy_similarity
        self._rng = np.random.default_rng(seed)

    def generate_policies(self, n_policies) -> list[Policy]:
        n_states, n_actions = self.n_states, self.n_actions
        rng = self._rng

        base_policy = rng.integers(0, high=n_actions, size=(1, n_states))
        policies = base_policy.repeat(n_policies, axis=0)

        n_states_to_change = int(n_states * (1 - self.policy_similarity))
        # to-change indices
        indices = np.vstack([
            rng.choice(n_states, n_states_to_change, replace=False)
            for _ in range(n_policies - 1)
        ])

        # re-sample actions — from reduced action space (note n_actions-1)
        new_actions = rng.integers(0, n_actions - 1, (n_policies - 1, n_states_to_change))
        old_actions = policies[0][indices]
        # that's how we exclude origin action: |0|1|2| -> |0|.|2|3| — action 1 is excluded
        mask = new_actions >= old_actions
        new_actions[mask] += 1

        # replace origin actions for specified state indices with new actions
        np.put_along_axis(policies[1:], indices, new_actions, axis=1)

        states_encoding = [self.state_encoder.encode(s) for s in range(n_states)]
        action_encoding = [self.action_encoder.encode(a) for a in range(n_actions)]

        encoded_policies = []
        for i_policy in range(n_policies):
            policy = []
            for state in range(n_states):
                action = policies[i_policy, state]
                s = states_encoding[state]
                a = action_encoding[action]
                policy.append((s, a))

            encoded_policies.append(Policy(id=i_policy, policy=policy))

        return encoded_policies


class PolicySelector:
    n_policies: int
    regime: str

    seed: Optional[int]

    def __init__(self, n_policies: int, regime: str, seed: int = None):
        self.n_policies = n_policies
        self.regime = regime
        self.seed = seed

    def __iter__(self):
        if self.regime == 'ordered':
            return range(self.n_policies)
        elif self.regime == 'random':
            assert self.seed is not None, 'seed is expected for random selector'

            rng = np.random.default_rng(self.seed)
            return iter(rng.permutation(self.n_policies))
        else:
            raise KeyError(f'{self.regime} is not supported')


def resolve_data_generator(config: dict):
    seed = config['seed']
    generator_type, generator_config = which_type(config['generator'], extract=True)

    if generator_type == 'synthetic':
        return SyntheticGenerator(config, seed=seed, **generator_config)
    else:
        raise KeyError(f'{generator_type} is not supported')


def resolve_encoder(config: dict, key, registry_key: str):
    registry = config[registry_key]
    encoder_type, encoder_config = which_type(registry[key], extract=True)

    if encoder_type == 'int_bucket':
        return IntBucketEncoder(**encoder_config)
    else:
        raise KeyError(f'{encoder_type} is not supported')


def generate_data(n, n_actions, n_states, randomness=1.0, seed=0):
    raw_data = list()
    np.random.seed(seed)
    seed_seq = np.random.randint(0, n_actions, n_states)
    raw_data.append(seed_seq.copy())
    n_replace = int(n_states * randomness)
    for i in range(1, n):
        new_seq = np.random.randint(0, n_actions, n_states)
        if randomness == 1.0:
            raw_data.append(new_seq)
        else:
            indices = np.random.randint(0, n_states, n_replace)
            seed_seq[indices] = new_seq[indices]
            raw_data.append(seed_seq.copy())
    data = [list(zip(range(n_states), x)) for x in raw_data]
    return raw_data, data
