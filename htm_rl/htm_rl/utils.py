from typing import List, Tuple

import numpy as np


def isnone(x, default):
    return x if x is not None else default


def sparse_to_dense(indices: List[int], total_size: int) -> np.ndarray:
    dense_vector = np.zeros(total_size, dtype=np.int8)
    dense_vector[indices] = 1
    return dense_vector


def dense_to_sparse(dense_vector: np.ndarray) -> List[int]:
    return np.nonzero(dense_vector)[0].tolist()

