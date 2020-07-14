import random

import numpy as np

from htm_rl.agent.agent import Agent
from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.common.base_sar import SarRelatedComposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.common.utils import trace
from htm_rl.envs.mdp import GridworldMdpGenerator, Mdp, SarSuperpositionFormatter
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


def make_mdp_passage(cell_gonality, path, seed, clockwise_action: bool = False):
    initial_state = (0, 0)
    current_cell = 0
    cell_transitions = []
    for direction in path:
        cell_transitions.append((current_cell, direction, current_cell + 1))
        current_cell += 1

    generator = GridworldMdpGenerator(cell_gonality)
    return generator.generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way(start_direction, seed, clockwise_action: bool = False):
    """
    ######
    #01###
    #234##
    ##567#
    ###89#
    ######
    """
    initial_state = (0, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 2),  # c0 . c2
        (1, 3, 3),  # c1 . c3
        (2, 0, 3),  # c2 > c3
        (3, 0, 4),  # c3 > c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 6),  # c4 > c6
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
        (6, 3, 8),  # c6 . c8
        (7, 3, 9),  # c7 . c9
        (8, 0, 9),  # c8 > c9
    ]

    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way_v0(start_direction, seed, clockwise_action: bool = False):
    """
    #####
    #012#
    #3#4#
    #567#
    #####
    """
    initial_state = (0, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 3),  # c0 . c3
        (1, 0, 2),  # c1 > c2
        (2, 3, 4),  # c2 . c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 7),  # c4 . c7
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
    ]
    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_sar_encoder(
        n_unique_states, n_unique_actions, n_unique_rewards=2,
        bits_per_state_value=8, bits_per_action_value=8, bits_per_reward_value=8,
        trace_format: str = 'short'
):
    # shortcuts
    bps, bpa, bpr = bits_per_state_value, bits_per_action_value, bits_per_reward_value

    state_encoder = IntSdrEncoder('state', n_unique_states, bps, bps - 1, trace_format)
    action_encoder = IntSdrEncoder('action', n_unique_actions, bpa, bpa - 1, trace_format)
    reward_encoder = IntSdrEncoder('reward', n_unique_rewards, bpr, bpr - 1, trace_format)

    return SarSdrEncoder(
        encoders=SarRelatedComposition(state_encoder, action_encoder, reward_encoder)
    )


def make_tm(encoder: SarSdrEncoder, cells_per_column, seed, verbose: bool) -> TemporalMemory:
    n_columns = encoder.total_bits
    activation_threshold = encoder.activation_threshold
    learning_threshold = int(0.85*activation_threshold)
    max_new_synapses_count = encoder.value_bits
    max_synapses_per_segment = encoder.value_bits

    action_act_threshold = encoder._encoders.action.activation_threshold
    reward_act_threshold = encoder._encoders.reward.activation_threshold
    assert action_act_threshold + reward_act_threshold < learning_threshold

    trace(
        verbose,
        f'Cells: {n_columns}x{cells_per_column}; activation: {activation_threshold}; learn: {learning_threshold}'
    )

    return TemporalMemory(
        n_columns=n_columns,
        cells_per_column=cells_per_column,
        activation_threshold=activation_threshold,
        learning_threshold=learning_threshold,
        initial_permanence=.5,
        connected_permanence=.4,
        predictedSegmentDecrement=.0001,
        permanenceIncrement=.1,
        permanenceDecrement=.05,
        maxNewSynapseCount=max_new_synapses_count,
        maxSynapsesPerSegment=max_synapses_per_segment,
        seed=seed,
    )


def make_agent(
        env, cells_per_column, planning_horizon, use_cooldown, seed: int, trace_format: str, verbose: bool
):
    n_states = env.n_states
    n_actions = env.n_actions
    trace(verbose, f'States: {n_states}')

    encoder = make_sar_encoder(n_states, n_actions, trace_format=trace_format)

    sdr_formatter = encoder.format
    sar_formatter = SarSuperpositionFormatter.format

    tm = make_tm(encoder, cells_per_column, seed, verbose=verbose)
    memory = Memory(tm, encoder, sdr_formatter, sar_formatter, collect_anomalies=True)
    planner = Planner(memory, planning_horizon)
    agent = Agent(memory, planner, n_actions, use_cooldown)
    return agent