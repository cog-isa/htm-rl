import random

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.agent import Agent
from htm_rl.experimental.list_sdr_encoder import Dim2d
from htm_rl.envs.minigrid import make_minigrid
from htm_rl.mdp.sar import Sar, SarSuperpositionFormatter
from htm_rl.mdp.sar_sdr_encoder import SarSdrEncoder
from htm_rl.agent.planner import Planner
from htm_rl.common.int_sdr_encoder import IntSdrEncoderShortFormat as IntSdrEncoder
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!
It's outdated and doesn't work RN as for now minigrid is not a main target for testing.

Don't delete as it contains some useful code. It will be updated at some point in future.
!!!!!!!!!!!!!!!!!!!!!!!!!!!


Here you can test agent+planner on easy MDP variations of minigrid envs.
"""


def render_env(env, render: bool, pause: float = None):
    if not render:
        return

    plt.imshow(env.render('rgb_array'))
    if pause is None or pause < .01:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(pause)


def print_debug_sar(sar, encoder, sar_formatter):
    indices = encoder.encode(sar)
    print(encoder.format_sar_superposition(indices))
    sar_superposition = encoder.decode(indices)
    print(sar_formatter.format_sar_superposition(sar_superposition))


def print_proximal_input(proximal_input, encoder, print_enabled):
    if print_enabled:
        print(encoder.format_sar_superposition(proximal_input))

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


def train_for(n_steps, observation, reward, a_ind, observations_mapping, rnd_eps, print_enabled):
    if print_enabled:
        print('=> stepping')
    reward_reached = 0
    done = False
    prev_obs, prev_action = None, None
    for _ in range(n_steps + 1):
        if np.random.rand() < rnd_eps:
            action = np.random.choice(2) + 1
        else:
            action = actions[a_ind % len(actions)]
        a_ind += 1

        encoded_obs = find_index(observations_mapping, observation)
        sar = Sar(encoded_obs, action, reward)
        proximal_input = encoder.encode(sar)
        if prev_obs is not None and (prev_obs, prev_action) not in mdp:
            mdp[(prev_obs, prev_action)] = encoded_obs

        # print(encoded_obs, action, reward)
        # print(format_minigrid_observation(observation))
        agent.train_one_step(proximal_input, print_enabled)
        if print_enabled:
            print(f'{encoded_obs} <-', [(s1, a) for (s1, a), s2 in mdp.items() if s2 == encoded_obs])
            print(f'{encoded_obs} ->', [(a, s2) for (s1, a), s2 in mdp.items() if s1 == encoded_obs])
            print()
        render_env(env, render, pause)

        next_observation, reward, done, info = env.step(action)
        if reward > 0:
            reward = 1
            reward_reached += 1
        # obs_sdr = encode_data(merge_data(obs, action, reward))

        observation = next_observation
        prev_obs, prev_action = encoded_obs, action

        if done:
            action = 2 # any next action
            render_env(env, render, pause)
            encoded_obs = find_index(observations_mapping, observation)
            sar = Sar(encoded_obs, action, reward)
            proximal_input = encoder.encode(sar)
            if prev_obs is not None and (prev_obs, prev_action) not in mdp:
                mdp[(prev_obs, prev_action)] = encoded_obs

            # print(encoded_obs)
            # print(format_minigrid_observation(observation))
            agent.train_one_step(proximal_input, print_enabled)
            if print_enabled:
                print(f'{encoded_obs} <-', [(s1, a) for (s1, a), s2 in mdp.items() if s2 == encoded_obs])
                print(f'{encoded_obs} ->', [(a, s2) for (s1, a), s2 in mdp.items() if s1 == encoded_obs])
            # print('YAY')
            render_env(env, render, pause)
            break

    if done and reward != 1:
        agent.tm.reset()

    if print_enabled:
        print('<= stepping')

    return reward_reached


if __name__ == '__main__':
    # plt.figure(figsize=(2.5, 2.5))
    random.seed(1337)
    np.random.seed(1337)

    size, view_size = 7, None
    env = make_minigrid(size, view_size)
    n_dims = Dim2d(view_size, view_size)

    # k = (size - 2 - 1) // 2 + 1
    # actions, a_ind = ([2, 2, 1, 0] * k + [1, 2, 1] + [2, 0, 1, 2] * k + [0, 2, 0])*k, 0

    k = size - 2
    actions, a_ind = (([0] * 4 + [2]) * k + [1, 2, 1] + ([0] * 4 + [2]) * k + [0, 2, 0]) * ((k - 1)//2 + 1), 0
    observation, reward, done = env.reset(), 0, False

    all_observations = get_all_observations(actions, env, False, .2)
    all_observations = get_observations_mapping(all_observations)
    mdp = dict()
    n = all_observations.shape[0]
    print(n)

    bs, ba, br = 10, 10, 10
    # bs, ba, br = 5, 5, 5
    encoder = SarSdrEncoder((
        IntSdrEncoder('state', n, bs, bs-1),
        IntSdrEncoder('action', 3, ba, ba-1),
        IntSdrEncoder('reward', 2, br, br-1)
    ))
    sar_formatter = SarSuperpositionFormatter

    activation_threshold = encoder.activation_threshold
    learning_threshold = int(0.85*activation_threshold)
    print(encoder.total_bits, activation_threshold, learning_threshold)

    tm = TemporalMemory(
        n_columns=encoder.total_bits,
        cells_per_column=4,
        activation_threshold=activation_threshold, learning_threshold=learning_threshold,
        initial_permanence=.5, connected_permanence=.4,
        maxNewSynapseCount=encoder.value_bits,
        predictedSegmentDecrement=.0001,
        permanenceIncrement=.1,
        permanenceDecrement=.05,
        maxSynapsesPerSegment=encoder.value_bits,
    )
    agent = Agent(tm, encoder, sar_formatter.format_sar_superposition)

    encoded_obs = find_index(all_observations, observation)
    initial_sar = Sar(encoded_obs, 2, 0)
    initial_proximal_input = encoder.encode(initial_sar)

    observation = env.reset()
    encoded_obs = find_index(all_observations, observation)

    render, pause, print_enabled = False, -.3, False
    rnd_eps = 2.
    reward_reached = 0
    for _ in range(20):
        rew = train_for(1000, observation, reward, 0, all_observations, rnd_eps, print_enabled)
        reward_reached += rew
        tm.reset()
        if print_enabled:
            print()
        # agent.predict_cycle(initial_proximal_input, 1, True)
        tm.reset()
        observation = env.reset()
        reward = 0
        a_ind = 0
        if print_enabled:
            print('='*20)
    print(f'Reward reached: {reward_reached}')

    # print((initial_sar.state, initial_sar.action), mdp[(initial_sar.state, initial_sar.action)])
    # print(format_minigrid_observation(all_observations[9]))
    # print(format_minigrid_observation(all_observations[0]))

    # print((27, 1), mdp[(27, 1)])
    # print(format_minigrid_observation(all_observations[27]))
    # print(format_minigrid_observation(all_observations[10]))
    # print(format_minigrid_observation(all_observations[9]))
    #
    # print(format_minigrid_observation(all_observations[9]), 9)
    # for i in [1, 4, 17]:
    #     obs = all_observations[i]
    #     print(format_minigrid_observation(obs), i)
    #
    # render_env(env, True, None)
    planner = Planner(agent, size * 3, print_enabled=True)
    planner.plan_actions(initial_sar)
    #
    # agent.tm.printParameters()
    #
    # plt.figure(figsize=(12, 6))
    # anomalies = agent.anomalies
    # xs = np.arange(len(anomalies))
    # plt.plot(xs, anomalies)
    # plt.show()

    # print(format_minigrid_observation(all_observations[14]))
    # print(format_minigrid_observation(all_observations[20]))
    # for i, obs in enumerate(all_observations):
    #     print(format_minigrid_observation(obs), i)

