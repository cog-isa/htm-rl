import random

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.agent import Agent
from htm_rl.gridworld_agent.list_sdr_encoder import ListSdrEncoder, Dim2d
from htm_rl.gridworld_agent.minigrid import make_minigrid
from htm_rl.mdp_agent.sar import Sar, SarSuperpositionFormatter
from htm_rl.mdp_agent.sar_sdr_encoder import SarSdrEncoder
from htm_rl.planner import Planner
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.temporal_memory import TemporalMemory


def render_env(env, render: bool, pause: float = None):
    if not render:
        return

    plt.imshow(env.render('rgb_array'))
    if pause is None or pause < .01:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(.1)


def print_debug_sar(sar, encoder, sar_formatter):
    indices = encoder.encode(sar)
    print(encoder.format(indices))
    sar_superposition = encoder.decode(indices)
    print(sar_formatter.format(sar_superposition))


def get_all_observations(actions, env, render, pause):
    observation = env.reset()
    observations = [observation]
    for action in actions:
        render_env(env, render, pause)

        next_observation, reward, done, info = env.step(action)
        observation = next_observation
        observations.append(observation)

    return observations


def get_observations_mapping(observations):
    observations = np.array(observations)
    observations = np.unique(observations, axis=0)
    return observations


def find_index(observations, observation):
    mask = (observations == observation).all(axis=1)
    return np.argwhere(mask)[0][0]


def train_for(n_steps, observation, reward, a_ind, observations_mapping, print_enabled):
    reward_reached = 0
    done = False
    for _ in range(n_steps):
        if np.random.rand() < 2:
            action = np.random.choice(3)
        else:
            action = actions[a_ind % len(actions)]
        a_ind += 1

        render_env(env, render, pause)
        encoded_obs = find_index(observations_mapping, observation)
        sar = Sar(encoded_obs, action, reward)
        proximal_input = encoder.encode(sar)
        agent.train_one_step(proximal_input, print_enabled)
        # print_debug_sar(sar, encoder, sar_formatter)

        next_observation, reward, done, info = env.step(action)
        if reward > 0:
            reward = 1
            reward_reached += 1
        # obs_sdr = encode_data(merge_data(obs, action, reward))

        observation = next_observation

        if done:
            action = 2 # any next action
            render_env(env, render, pause)
            encoded_obs = find_index(observations_mapping, observation)
            sar = Sar(encoded_obs, action, reward)
            proximal_input = encoder.encode(sar)
            agent.train_one_step(proximal_input, print_enabled)
            # print_debug_sar(sar, encoder, sar_formatter)
            break

    if done and reward != 1:
        agent.tm.reset()

    return reward_reached


if __name__ == '__main__':
    # plt.figure(figsize=(2.5, 2.5))
    random.seed(1337)
    np.random.seed(1337)

    size, view_size = 5, 4
    env = make_minigrid(size, view_size)
    n_dims = Dim2d(view_size, view_size)

    # k = (size - 2 - 1) // 2 + 1
    # actions, a_ind = ([2, 2, 1, 0] * k + [1, 2, 1] + [2, 0, 1, 2] * k + [0, 2, 0])*k, 0

    k = size - 2
    actions, a_ind = (([0] * 4 + [2]) * k + [1, 2, 1] + ([0] * 4 + [2]) * k + [0, 2, 0]) * ((k - 1)//2 + 1), 0
    observation, reward, done = env.reset(), 0, False

    all_observations = get_all_observations(actions, env, False, .2)
    all_observations = get_observations_mapping(all_observations)
    n = all_observations.shape[0]
    print(n)

    bs, ba, br = 6, 10, 10
    encoder = SarSdrEncoder((
        IntSdrEncoder('state', n, bs, bs-1),
        IntSdrEncoder('action', 3, ba, ba-1),
        IntSdrEncoder('reward', 2, br, br-1)
    ))
    sar_formatter = SarSuperpositionFormatter

    activation_threshold = encoder.activation_threshold
    learning_threshold = int(0.9*activation_threshold)
    print(encoder.total_bits, activation_threshold, learning_threshold)

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=8,
        activation_threshold=activation_threshold, learning_threshold=learning_threshold,
        initial_permanence=.5, connected_permanence=.25,
        maxNewSynapseCount=int(1.1 * encoder.value_bits),
        predictedSegmentDecrement=.0001,
        permanenceIncrement=.1,
        permanenceDecrement=.1,
    )
    agent = Agent(tm, encoder, sar_formatter.format)

    render, pause = False, .1
    reward_reached = 0
    for _ in range(50):
        reward_reached += train_for(40, observation, reward, 0, all_observations, False)
        tm.reset()
        observation = env.reset()
        reward = 0
        a_ind = 0
    print(f'Reward reached: {reward_reached}')

    # train_for(10, observation, reward, 10, True)

    encoded_obs = find_index(all_observations, observation)
    initial_sar = Sar(encoded_obs, 2, 0)
    initial_proximal_input = encoder.encode(initial_sar)
    # agent.predict_cycle(initial_proximal_input, 20, True)

    planner = Planner(agent, 5, print_enabled=True)
    planner.plan_actions(initial_sar)

    agent.tm.printParameters()

    plt.figure(figsize=(12, 6))
    anomalies = agent.anomalies
    xs = np.arange(len(anomalies))
    plt.plot(xs, anomalies)
    plt.show()
