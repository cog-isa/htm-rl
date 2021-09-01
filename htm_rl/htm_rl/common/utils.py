from functools import wraps
from timeit import default_timer as timer
from typing import Any, Union, Optional

import numpy as np


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def trace(verbosity: int, req_level: int, str_to_print: str = ''):
    """
    Prints string only if `req_level` is not greater than `verbosity`

    :param verbosity: accepted level of verbosity defined by the caller
    :param req_level: required level of verbosity defined by the callee
    :param str_to_print: string to print
    """
    if req_level <= verbosity:
        print(str_to_print)


def ensure_list(arr: Optional[Union[Any, list[Any]]]) -> Optional[list[Any]]:
    """Wraps single value to list or return list as it is."""
    if arr is not None and not isinstance(arr, list):
        arr = [arr]
    return arr


def safe_ith(arr: Optional[list], ind: int, default: Any = None) -> Optional[Any]:
    """Performs safe index access. If array is None, returns default."""
    if arr is not None:
        return arr[ind]
    return default


def wrap(obj, *wrappers):
    """Sequentially wraps object."""
    for wrapper in wrappers:
        obj = wrapper(obj)
    return obj


def timed(f):
    """Wraps function with the timer that returns tuple: result, elapsed_time."""
    @wraps(f)
    def _wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return _wrap


def exp_sum(ema, decay, val):
    """Returns new exponential moving average (EMA) adding next value."""
    return ema * decay + val


def lin_sum(x, lr, y):
    """Returns linear sum."""
    return x + lr * (y - x)


def update_slice_exp_sum(s, ind, decay, val):
    """Updates EMA for specified slice."""
    s[ind] *= decay
    s[ind] += val


def update_slice_lin_sum(s, ind, lr, val):
    """Updates slice value estimate with specified learning rate."""
    s[ind] = (1 - lr) * s[ind] + lr * val


def update_exp_trace(traces, tr, decay, val=1.):
    """Updates exponential trace."""
    traces *= decay
    traces[tr] += val


def exp_decay(factor_decay_tuple):
    """Returns new tuple with exponentially decayed first value."""
    factor, decay = factor_decay_tuple
    return factor * decay, decay


def modify_factor_tuple(factor_decay_tuple, alpha):
    """Returns new tuple with the first value multiplied by specified factor."""
    factor, decay = factor_decay_tuple
    return factor * alpha, decay


def softmax(x):
    """Computes softmax values for a vector `x`."""
    e_x = np.exp(x - np.max(x, axis=-1))
    return e_x / e_x.sum(axis=-1)
