import numpy as np

from agent import Agent
from planner import Planner
from representations.int_sdr_encoder import IntSdrEncoder
from representations.sar import Sar, str_from_sar_superposition
from representations.sar_sdr_encoder import SarSdrEncoder, SarSdrEncodersNT
from representations.sar_value import decode_sar_value
from temporal_memory import TemporalMemory
# from tests import test_all

if __name__ == '__main__':
    # test_all()

    sar_encoders = SarSdrEncodersNT(
        s=IntSdrEncoder('state', 6, 10, 8),
        a=IntSdrEncoder('action', 3, 5, 4),
        r=IntSdrEncoder('reward', 2, 5, 4),
    )
    encoder = SarSdrEncoder(sar_encoders)

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=1,
        activation_threshold=16, learning_threshold=12,
        initial_permanence=.5, connected_permanence=.5
    )
    agent = Agent(tm, encoder)

    sar_value_sequences = [
        [10, 120, 210, 301],
        [20, 410, 510, 301],
        # [10, 120, 220, 410, 510, 301],
        # [20, 410]
    ]
    train_samples = [
        [encoder.encode_sparse(x) for x in decode_sar_value(sequence)]
        for sequence in sar_value_sequences
    ]
    n_train_samples = len(train_samples)

    for _ in range(n_train_samples * 40):
        train_sample = train_samples[np.random.choice(n_train_samples)]
        agent.train_cycle(train_sample, print_enabled=False, reset_enabled=True)

    initial_sar = Sar(0, -1, 0)
    initial_proximal_input = encoder.encode_sparse(initial_sar)
    print(str_from_sar_superposition(encoder.decode_sparse(initial_proximal_input)))
    # start_value = train_samples[0][0]

    # agent.predict_cycle(initial_proximal_input, 3, print_enabled=True, reset_enabled=True)
    # agent.plan_to_value(start_value, 6, print_enabled=True, reset_enabled=True)

    print(agent.tm.n_columns)
    planner = Planner(agent, 6, print_enabled=True)
    planner.plan_actions(initial_sar)
