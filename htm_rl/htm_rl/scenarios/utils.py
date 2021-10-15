from ast import literal_eval

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


def add_overwrite_attributes(
        config: Config, overwrites: list[str], prefix: str = None, ignore: str = None
):
    i = 0
    step = None
    while i < len(overwrites):
        key = overwrites[i]
        if not key.startswith('--'):
            raise ValueError(
                f'Config attribute name started with `--` is expected, got `{key}`!'
            )

        # remove starting `--`
        key = key[2:]

        # resolve params format if not yet
        if step is None:
            if '=' in key:
                # step = 1 for `--key=value` wandb format
                step = 1
            else:
                # step = 2 for `--key value` for default arg.parse format
                step = 2

        if step == 1:
            key, value = key.split('=')
        else:
            value = overwrites[i+1]
            if value.startswith('--'):
                raise ValueError(
                    f'Config attribute value is expected, got `{value}`, '
                    f'which looks like an attribute name!'
                )

        # materialize value from string
        value = parse_str(value)

        if prefix is not None:
            if key.startswith(prefix):
                key = key[len(prefix):]
            else:
                i += step
                continue
        elif ignore is not None:
            if key.startswith(ignore):
                i += step
                continue

        config[key] = value
        i += step
    return i


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
