from typing import Sequence, Any, Iterable

import numpy as np

from htm_rl.representations.sdr import SparseSdr, DenseSdr


def isnone(x, default):
    return x if x is not None else default


def range_reverse(arr: Sequence[Any]) -> Iterable[Any]:
    return range(len(arr) - 1, -1, -1)


def sparse_to_dense(indices: SparseSdr, total_size: int) -> DenseSdr:
    dense_vector = np.zeros(total_size, dtype=np.int8)
    dense_vector[indices] = 1
    return dense_vector


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.nonzero(dense_vector)[0].tolist()

