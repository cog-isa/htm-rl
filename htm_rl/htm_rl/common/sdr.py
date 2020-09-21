from typing import Iterable, Union, Sequence

import numpy as np

SparseSdr = Union[Iterable[int], Sequence[int]]
DenseSdr = np.ndarray


def sparse_to_dense(indices: SparseSdr, total_size: int) -> DenseSdr:
    """Converts SDR from sparse representation to dense."""
    dense_vector = np.zeros(total_size, dtype=np.int8)
    dense_vector[indices] = 1
    return dense_vector


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.nonzero(dense_vector)[0].tolist()
