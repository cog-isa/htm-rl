from ast import literal_eval
from typing import Iterable, Union


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
