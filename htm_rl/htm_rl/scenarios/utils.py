from ast import literal_eval
from typing import Iterable, Union

from htm_rl.common.utils import ensure_list
from htm_rl.scenarios.config import Config


class ProgressPoint:
    step: int
    episode: int

    def __init__(self, pp: 'ProgressPoint' = None):
        if pp is not None:
            self.step = pp.step
            self.episode = pp.episode
        else:
            self.step = 0
            self.episode = 0

    @property
    def is_new_episode(self) -> bool:
        return self.step == 0

    def next_step(self):
        self.step += 1

    def end_episode(self, increase_episode: bool = True):
        self.step = 0
        if increase_episode:
            self.episode += 1

    def as_tuple(self):
        return self.episode, self.step

    def __eq__(self, other):
        if not isinstance(other, ProgressPoint):
            return False
        if self.episode != other.episode:
            return False
        return self.step == other.step

    def __str__(self):
        return f'{self.episode}.{self.step}'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.as_tuple())


def filtered(d: dict, keys_to_remove: Iterable[str], depth: int) -> dict:
    """
    Returns a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


def filtered_by_name_convention(config: dict, depth: int) -> dict:
    """
    Recursively filters out non-passable args started with '.' and '_'.

    The `depth == 1` means filtering `config` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.

    For example, it filters out `_type_` and `.xyz` dict keys,
    which by our convention in yaml configs denote the type name
    and a local variable correspondingly.
    """
    if not isinstance(config, dict) or depth <= 0:
        return config

    return {
        k: filtered_by_name_convention(v, depth - 1)
        for k, v in config.items()
        if not (k.startswith('.') or k.startswith('_'))
    }


def parse_str(val):
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not Boolean Value!')

    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(val)
        except ValueError:
            pass
    return val


def which_type(config: dict, extract: bool = False) -> Union[str, tuple[str, dict]]:
    key = '_type_'
    t = config.get(key, None)
    if extract:
        filtered_config = filtered(config, keys_to_remove={key}, depth=1)
        return t, filtered_config

    return t


def add_overwrite_attributes(config: Config, overwrites: list[str]):
    i = 0
    while i < len(overwrites):
        key = overwrites[i]
        if not key.startswith('--'):
            raise ValueError(
                f'Config attribute name started with `--` is expected, got `{key}`!'
            )
        key = key[2:]
        value = overwrites[i+1]
        if value.startswith('--'):
            raise ValueError(
                f'Config attribute value is expected, got `{value}`, which looks like an attribute name!'
            )
        value = parse_str(value)
        config[key] = value
        i += 2


def get_filtered_names_for(config: Config, key: str) -> list[str]:
    filter_key = f'{key}_filter'
    names = config[key].keys()
    names_filter = config[filter_key]

    if names_filter is None:
        return names

    names_filter = set(ensure_list(names_filter))
    return [
        name
        for name in names
        if name in names_filter
    ]
