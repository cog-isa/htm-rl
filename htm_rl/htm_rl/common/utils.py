from functools import wraps
from timeit import default_timer as timer
from typing import Sequence, Any, Iterable, Union, Optional


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def range_reverse(arr: Sequence[Any]) -> Iterable[int]:
    """Returns range iterator for reverse `arr` traversal"""
    return range(len(arr) - 1, -1, -1)


def trace(verbosity: int, req_level: int, str_to_print: str = ''):
    """
    Prints string only if `req_level` is not greater than `verbosity`

    :param verbosity: accepted level of verbosity defined by the caller
    :param req_level: required level of verbosity defined by the callee
    :param str_to_print: string to print
    """
    if req_level <= verbosity:
        print(str_to_print)


def clip(x, high):
    if x >= high:
        return high - 1
    if x < 0:
        return 0
    return x


def ensure_list(arr: Optional[Union[Any, list[Any]]]) -> Optional[list[Any]]:
    if arr is not None and not isinstance(arr, list):
        arr = [arr]
    return arr


def safe_ith(arr: Optional[list], ind: int, default: Any = None) -> Optional[Any]:
    if arr is not None:
        return arr[ind]
    return default


def wrap(obj, *wrappers):
    for wrapper in wrappers:
        obj = wrapper(obj)
    return obj


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return wrap
