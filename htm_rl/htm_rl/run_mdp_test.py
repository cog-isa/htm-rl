import numpy as np

from htm_rl.agent import Agent
from htm_rl.mdp_agent.sar import SarSuperpositionFormatter, Sar
from htm_rl.mdp_agent.sar_sdr_encoder import SarSdrEncoder
from htm_rl.mdp_agent.sar_value import decode_sar_value
from htm_rl.planner import Planner
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.temporal_memory import TemporalMemory


if __name__ == '__main__':
    encoder = SarSdrEncoder((
        IntSdrEncoder('state', 6, 10, 8),
        IntSdrEncoder('action', 3, 5, 4),
        IntSdrEncoder('reward', 2, 5, 4),
    ))

    activation_threshold = encoder.activation_threshold
    learning_threshold = int(0.66*activation_threshold)
    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=2,
        activation_threshold=activation_threshold, learning_threshold=learning_threshold,
        initial_permanence=.5, connected_permanence=.5
    )
    agent = Agent(tm, encoder, SarSuperpositionFormatter.format)

    sar_value_sequences = [
        [10, 120, 210, 301],
        [20, 410, 510, 301],
        [10, 120, 220, 410, 510, 301],
        [20, 410]
    ]
    train_samples = [
        [encoder.encode(x) for x in decode_sar_value(sequence)]
        for sequence in sar_value_sequences
    ]
    n_train_samples = len(train_samples)

    for _ in range(n_train_samples * 40):
        train_sample = train_samples[np.random.choice(n_train_samples)]
        agent.train_cycle(train_sample, print_enabled=False, reset_enabled=True)

    initial_sar = Sar(0, -1, 0)
    initial_proximal_input = encoder.encode(initial_sar)
    # start_value = train_samples[0][0]

    agent.predict_cycle(initial_proximal_input, 3, print_enabled=True, reset_enabled=True)

    planner = Planner(agent, 6, print_enabled=True)
    planner.plan_actions(initial_sar)
