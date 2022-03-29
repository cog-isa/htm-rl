import numpy as np


def symmetric_diff_sz(set1: np.ndarray, set2: np.ndarray) -> int:
    return np.setdiff1d(set1, set2).size + np.setdiff1d(set2, set1).size


def symmetric_error(_output, _target):
    if _output.size + _target.size == 0:
        return 0
    return symmetric_diff_sz(_output, _target) / np.union1d(_output, _target).size


def representations_intersection_1(dense1, dense2):
    if np.count_nonzero(dense1) == 0:
        return 1
    return np.count_nonzero(dense1 * dense2) / np.count_nonzero(dense1)


def row_similarity(policy_1, policy_2):
    counter = 0
    for index in range(len(policy_1)):
        if policy_1[index] == policy_2[index]:
            counter += 1
    return counter / len(policy_1)


def representation_similarity(representation_1, representation_2):
    overlap = (representation_1 * representation_2).nonzero()[0].size
    union = (representation_1 | representation_2).nonzero()[0].size

    return overlap / union
