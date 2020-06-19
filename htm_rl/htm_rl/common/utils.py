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
