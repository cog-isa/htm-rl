import random

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.agent import Agent
from htm_rl.envs.mdp import generate_gridworld_mdp
from htm_rl.mdp_agent.sar import Sar, SarSuperpositionFormatter
from htm_rl.mdp_agent.sar_sdr_encoder import SarSdrEncoder
from htm_rl.planner import Planner
from htm_rl.representations.int_sdr_encoder import IntSdrEncoderShortFormat as IntSdrEncoder
from htm_rl.representations.temporal_memory import TemporalMemory


def render_env(env, render):
    if not render:
        return
    env.render()


def print_debug_sar(sar, encoder, sar_formatter):
    indices = encoder.encode(sar)
    print(encoder.format(indices))
    sar_superposition = encoder.decode(indices)
    print(sar_formatter.format(sar_superposition))


def train_for(n_steps, observation, reward, print_enabled):
    reward_reached = 0
    done = False
    for step in range(n_steps + 1):
        action = np.random.choice(2)

        render_env(env, render)
        sar = Sar(observation, action, reward)
        proximal_input = encoder.encode(sar)
        agent.train_one_step(proximal_input, print_enabled)

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
            sar = Sar(observation, action, reward)
            proximal_input = encoder.encode(sar)
            agent.train_one_step(proximal_input, print_enabled)
            break

    if done and reward != 1:
        agent.tm.reset()

    if print_enabled:
        print()
    return reward_reached


if __name__ == '__main__':
    random.seed(1337)
    np.random.seed(1337)

    env = generate_gridworld_mdp(
        initial_state=(0, 0),   # c0 ^
        cell_transitions=[
            (0, 0, 1),      # c0 > c1
            (1, 1, 2),      # c1 ^ c2
            (2, 0, 3),      # c2 > c3
        ],
        # cell_transitions=[
        #     (0, 0, 1),      # c0 > c1
        #     (0, 3, 2),      # c0 . c2
        #     (1, 3, 4),      # c1 > c4
        #     (2, 0, 4),      # c2 > c4
        # ],
        # add_clockwise_action=True
    )

    observation, reward, done = env.reset(), 0, False
    n = env.n_states
    print(f'States: {n}')

    bs, ba, br = 10, 10, 10
    encoder = SarSdrEncoder((
        IntSdrEncoder('state', n, bs, bs-1),
        IntSdrEncoder('action', 2, ba, ba-1),
        IntSdrEncoder('reward', 2, br, br-1)
    ))
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
    agent = Agent(tm, encoder, sar_formatter.format)

    render, pause = False, .1
    reward_reached = 0
    for _ in range(20):
        reward_reached += train_for(40, observation, reward, False)
        tm.reset()
        observation = env.reset()
        reward = 0
        a_ind = 0
    print(f'Reward reached: {reward_reached}')

    # train_for(10, observation, reward, 10, True)

    initial_sar = Sar(observation, 1, 0)
    initial_proximal_input = encoder.encode(Sar(3, 1, 0))
    agent.predict_cycle(initial_proximal_input, 1, True)

    planner = Planner(agent, n, print_enabled=True)
    planner.plan_actions(initial_sar)

    # agent.tm.printParameters()

    # plt.figure(figsize=(12, 6))
    # anomalies = agent.anomalies
    # xs = np.arange(len(anomalies))
    # plt.plot(xs, anomalies)
    # plt.show()
