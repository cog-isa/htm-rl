from typing import Tuple, List

import numpy as np

from mdp_planner import TemporalMemory, DataMultiEncoder, DataEncoder, HtmAgent, SarEncoder
from tests import test_all


def encode_sequence(encoder: DataMultiEncoder, sequence: List[int]) -> List[np.ndarray]:
    return [encoder.encode_dense(x) for x in SarEncoder.decode_arr(sequence)]


if __name__ == '__main__':
    test_all()

    encoder = DataMultiEncoder((
        DataEncoder('state', 6, 8, 6),
        DataEncoder('action', 3, 4, 3),
        DataEncoder('reward', 2, 3, 2),
    ))

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=2,
        activation_threshold=12, learning_threshold=9,
        initial_permanence=.5, connected_permanence=.5
    )
    agent = HtmAgent(tm, encoder)

    raw_sequences = [
        [1, 12, 21, 130],
        [2, 41, 51, 130]
    ]
    train_samples = [encode_sequence(encoder, sequence) for sequence in raw_sequences]
    n_train_samples = len(train_samples)

    for _ in range(n_train_samples * 40):
        train_sample = train_samples[np.random.choice(n_train_samples)]
        agent.train_cycle(train_sample, print_enabled=False, learn=True)

    start_from = train_samples[0][0]
    print(raw_sequences[0])
    agent.predict_cycle(start_from, 3, print_enabled=True, reset_enabled=True)
