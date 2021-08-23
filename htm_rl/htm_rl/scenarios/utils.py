from ast import literal_eval

from htm_rl.common.utils import ensure_list
from htm_rl.scenarios.config import Config


class ProgressPoint:
    step: int
    episode: int

    def __init__(self):
        self.step = 0
        self.episode = 0

    @property
    def is_new_episode(self) -> bool:
        return self.step == 0

    def next_step(self):
        self.step += 1

    def end_episode(self):
        self.step = 0
        self.episode += 1


def filter_out_non_passable_items(config: dict, depth: int):
    """Recursively filters out non-passable args started with '.' and '_'."""
    if not isinstance(config, dict) or depth <= 0:
        return config

    return {
        k: filter_out_non_passable_items(v, depth - 1)
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
