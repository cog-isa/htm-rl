import dataclasses
import inspect
from functools import wraps
from itertools import chain
from timeit import default_timer as timer
from typing import Sequence, Any, Iterable


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def range_reverse(arr: Sequence[Any]) -> Iterable[int]:
    """Returns range iterator for reverse `arr` traversal"""
    return range(len(arr) - 1, -1, -1)


def trace(verbose: bool, str_to_print: str = ''):
    """Prints string only if `verbose` is True."""
    if verbose:
        print(str_to_print)


def project_to_type_fields(config_type, config):
    projection = {
        field.name: config[field.name]
        for field in dataclasses.fields(config_type)
        if field.name in config
    }
    return projection


def project_to_method_params(func, config):
    argspec = inspect.getfullargspec(func)
    args = chain(argspec.args, argspec.kwonlyargs)

    projection = {
        arg_name: config[arg_name]
        for arg_name in args
        if arg_name in config
    }
    return projection


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return wrap
