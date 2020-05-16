from typing import Tuple, List

import numpy as np

from mdp_planner import TemporalMemory, DataMultiEncoder, DataEncoder, HtmAgent, SarEncoder
from tests import test_all


def encode_sequence(encoder: DataMultiEncoder, sequence: List[int]) -> List[np.ndarray]:
    return [encoder.encode_dense(x) for x in SarEncoder.decode_arr(sequence)]


if __name__ == '__main__':
    test_all()

    encoder = DataMultiEncoder((
        DataEncoder('state', 6, 10, 8),
        DataEncoder('action', 3, 5, 4),
        DataEncoder('reward', 2, 5, 4),
    ))

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=2,
        activation_threshold=16, learning_threshold=12,
        initial_permanence=.5, connected_permanence=.5
    )
    agent = HtmAgent(tm, encoder)

    sar_sequences = [
        [10, 120, 210, 301],
        [20, 410, 510, 301],
        [10, 120, 220, 410, 510, 301]
    ]
    train_samples = [encode_sequence(encoder, sequence) for sequence in sar_sequences]
    n_train_samples = len(train_samples)

    for _ in range(n_train_samples * 40):
        train_sample = train_samples[np.random.choice(n_train_samples)]
        agent.train_cycle(train_sample, print_enabled=False, learn=True)


    start_value = encoder.encode_dense_state_all_actions(0)
    # start_value = train_samples[0][0]

    # agent.predict_cycle(start_value, 3, print_enabled=True, reset_enabled=True)
    agent.plan_to_value(start_value, 4, print_enabled=True, reset_enabled=True)
