from functools import wraps
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


def trace2(accepted_verbosity: int, verbosity: int, str_to_print: str = ''):
    """
    Prints string only if `verbosity` is not greater than `accepted_verbosity`
    """
    if verbosity <= accepted_verbosity:
        print(str_to_print)


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return wrap
