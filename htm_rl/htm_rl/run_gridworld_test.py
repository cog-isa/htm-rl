import random
from time import sleep

import gym
import gym_minigrid as minigrid
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM


def plot_prediction_accuracy(xs, ys, title):
    plt.ylim([-0.1, 1.1])
    plt.plot(xs, ys)
    plt.xlabel("Timestep")
    plt.ylabel("Prediction Accuracy")
    plt.title(title)
    plt.show()


def extract_observation_data(raw_obs):
    x = raw_obs[:, :, 0].copy()

    # make data is categorical on [0, 2] range
    x[x == 8] = 0

    # take 5x4 observation
    x = x[1:-1, -4:]

    x[x.shape[1] // 2, -1] = 3
    r = '\n'.join(
        ''.join('X' if c == 0 else '#' if c == 2 else '^' if c == 3 else '-' for c in row)
            for row in x.T
    )
    print(r)

    x[x.shape[1] // 2, -1] = 1

    return x


def merge_data(raw_obs, action, reward):
    # all data scalars are categorical on [0,2]
    # so we can just concat it together
    obs = extract_observation_data(raw_obs)
    obs = obs.ravel()
    obs = np.concatenate((obs, [action, reward]))
    return obs


def encode_data(raw, active_bits=3):
    s = SDR((raw.size, active_bits))

    for i in range(active_bits):
        s.dense[raw == i, i] = 1

    s.dense = s.dense
    return s


tm = TM(
    columnDimensions=(5 * 4 + 2, 3),
    cellsPerColumn=8,
    minThreshold=5,
    activationThreshold=5,
    initialPermanence=0.5,
    connectedPermanence=0.5,
)

env = gym.make('MiniGrid-Empty-6x6-v0')
env = minigrid.wrappers.ImgObsWrapper(env)
random.seed(1337)
np.random.seed(1337)

# render = False
render = True

k = 3
actions, a_ind = [2, 0, 1, 1, 0] * k + [1, 2, 1] + [2, 0, 1, 1, 0] * k + [0, 2, 0], 0
obs = env.reset()
ys = []

state_chars = {
    0: 'X',
    1: '-',
    2: '#',
    3: '^'
}
action_chars = {
    0: '<',
    1: '>',
    2: '^'
}
for _ in range(20):
    # action = np.random.choice(3)
    action = actions[a_ind % len(actions)]
    print(action_chars[action])
    a_ind += 1

    next_obs, reward, done, info = env.step(action)

    obs_sdr = encode_data(merge_data(obs, action, reward))
    tm.compute(obs_sdr, learn=True)
    ys.append(1 - tm.anomaly)

    if done:
        a_ind = 0
        next_obs = env.reset()
    #         tm.reset()

    if render:
        # plt.close()
        plt.imshow(env.render('rgb_array'))
        plt.show(block=False)
        plt.pause(.1)

    obs = next_obs

plt.pause(1.)
plt.close()
env.render(close=True)
# env.close()

xs = np.arange(len(ys))
ys = np.array(ys)

# plt.plot(xs, ys)
# plt.show()

# if __name__ == '__main__':
#     sar_encoders = SarSdrEncodersNT(
#         s=IntSdrEncoder('state', 6, 10, 8),
#         a=IntSdrEncoder('action', 3, 5, 4),
#         r=IntSdrEncoder('reward', 2, 5, 4),
#     )
#     encoder = SarSdrEncoder(sar_encoders)
#
#     tm = TemporalMemory(
#         n_columns=encoder.total_bits,
#         cells_per_column=2,
#         activation_threshold=16, learning_threshold=12,
#         initial_permanence=.5, connected_permanence=.5
#     )
#     agent = Agent(tm, encoder)
#
#     sar_value_sequences = [
#         [10, 120, 210, 301],
#         [20, 410, 510, 301],
#         [10, 120, 220, 410, 510, 301],
#         [20, 410]
#     ]
#     train_samples = [
#         [encoder.encode_sparse(x) for x in decode_sar_value(sequence)]
#         for sequence in sar_value_sequences
#     ]
#     n_train_samples = len(train_samples)
#
#     for _ in range(n_train_samples * 40):
#         train_sample = train_samples[np.random.choice(n_train_samples)]
#         agent.train_cycle(train_sample, print_enabled=False, reset_enabled=True)
#
#     initial_sar = Sar(0, -1, 0)
#     initial_proximal_input = encoder.encode_sparse(initial_sar)
#     # start_value = train_samples[0][0]
#
#     # agent.predict_cycle(initial_proximal_input, 3, print_enabled=True, reset_enabled=True)
#     # agent.plan_to_value(start_value, 6, print_enabled=True, reset_enabled=True)
#
#     planner = Planner(agent, 6, print_enabled=True)
#     planner.plan_actions(initial_sar)
