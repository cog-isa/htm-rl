import random

import numpy as np

from htm_rl.agent.agent import Agent
from htm_rl.agent.planner import Planner
from htm_rl.common.base_sar import Sar, SarRelatedComposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.envs.mdp import GridworldMdpGenerator, SarSuperpositionFormatter, Mdp
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


"""
Here you can test agent+planner on a simple handcrafted MDP envs.
"""


def render_env(env, render):
    if not render:
        return
    env.render()


def print_debug_sar(sar, encoder, sar_formatter):
    indices = encoder.encode(sar)
    print(encoder.format_sar_superposition(indices))
    sar_superposition = encoder.decode(indices)
    print(sar_formatter.format_sar_superposition(sar_superposition))


def train_for(n_steps, observation, reward, print_enabled):
    reward_reached = 0
    done = False
    for step in range(n_steps + 1):
        action = np.random.choice(env.n_actions)

        render_env(env, render)
        agent.train(Sar(observation, action, reward), print_enabled)

        if step == n_steps:
            break
        next_observation, reward, done, info = env.step(action)
        observation = next_observation
        if reward > 0:
            reward = 1
            reward_reached += 1

        if done:
            action = 0 # any next action
            render_env(env, render)
            agent.train(Sar(observation, action, reward), print_enabled)
            break

    if done and reward != 1:
        agent.tm.reset()

    if print_enabled:
        print()
    return reward_reached


if __name__ == '__main__':
    random.seed(1337)
    np.random.seed(1337)

    env = GridworldMdpGenerator(4).generate_mdp(
        Mdp,
        initial_state=(0, 2),   # c0 >
        # initial_state=None,     # random
        cell_transitions=[
            (0, 0, 1),      # c0 > c1
            # (1, 1, 2),      # c1 ^ c2
            # (2, 0, 3),      # c2 > c3
        ],
        # cell_transitions=[
        #     (0, 0, 1),      # c0 > c1
        #     (0, 3, 2),      # c0 . c2
        #     (1, 3, 3),      # c1 > c4
        #     (2, 0, 3),      # c2 > c4
        #     (3, 0, 4),      # c2 > c4
        #     (3, 3, 5),      # c2 > c4
        #     (4, 3, 6),      # c2 > c4
        #     (5, 0, 6),      # c2 > c4
        #     (6, 3, 7),      # c2 > c4
        #
        # ],
        # allow_clockwise_action=True
    )

    observation, reward, done = env.reset(), 0, False
    n_states = env.n_states
    n_actions = env.n_actions
    print(f'States: {n_states}')

    bs, ba, br = 10, 10, 10
    sar_encoders = SarRelatedComposition(
        IntSdrEncoder('state', n_states, bs, bs - 1, 'short'),
        IntSdrEncoder('action', n_actions, ba, ba-1, 'short'),
        IntSdrEncoder('reward', 2, br, br-1, 'short')
    )
    encoder = SarSdrEncoder(sar_encoders)

    sdr_formatter = encoder
    sar_formatter = SarSuperpositionFormatter

    activation_threshold = encoder.activation_threshold
    learning_threshold = int(0.85*activation_threshold)
    max_synapses_per_segment = encoder.value_bits
    # max_synapses_per_segment = int(2 * encoder.value_bits)
    print('Bits, act, learn:', encoder.total_bits, activation_threshold, learning_threshold)

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=1,
        activation_threshold=activation_threshold, learning_threshold=learning_threshold,
        initial_permanence=.5, connected_permanence=.4,
        maxNewSynapseCount=encoder.value_bits,
        predictedSegmentDecrement=.0001,
        permanenceIncrement=.1,
        permanenceDecrement=.05,
        maxSynapsesPerSegment=max_synapses_per_segment,
        # maxSegmentsPerCell=4,
    )
    agent = Agent(tm, encoder, sdr_formatter.format, sar_formatter.format)

    render, pause = False, .1
    reward_reached = 0
    for _ in range(30):
        reward_reached += train_for(50, observation, reward, False)
        tm.reset()
        observation = env.reset()
        reward = 0
        a_ind = 0
    print(f'Reward reached: {reward_reached}')

    # train_for(10, observation, reward, 10, True)

    # initial_proximal_input = encoder.encode(Sar(0, 0, 0))
    # agent.predict_cycle(initial_proximal_input, 4, True)

    observation = env.reset()
    initial_sar = Sar(observation, 1, 0)
    planner = Planner(agent, n_states)
    planner.plan_actions(initial_sar, True)

    # agent.tm.printParameters()

    # plt.figure(figsize=(12, 6))
    # anomalies = agent.anomalies
    # xs = np.arange(len(anomalies))
    # plt.plot(xs, anomalies)
    # plt.show()
