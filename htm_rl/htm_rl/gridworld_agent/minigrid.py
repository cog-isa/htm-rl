from math import sqrt

import gym
import gym_minigrid as minigrid


class MiniGridObservationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, view_size: int):
        super().__init__(env)
        self.view_size = view_size

    def observation(self, observation):
        # new_size, orig_size = self.view_size, observation.shape[0]
        # left = (orig_size - new_size) // 2
        # right = left + new_size
        # observation = observation[left:right, -new_size:]

        x = observation[:, :, 0].T.ravel().copy()
        # make data is categorical on [0, 2] range
        x[x == 8] = 0
        return x


def make_minigrid(size: int, view_size: int):
    assert size in {5, 6, 8, 16}
    env = gym.make(f'MiniGrid-Empty-{size}x{size}-v0')

    # observation_space = gym.spaces.Box(
    #     low=0,
    #     high=255,
    #     shape=(view_size, view_size, 3),
    #     dtype='uint8'
    # )
    #
    # env.observation_space = gym.spaces.Dict({
    #     'image': observation_space
    # })
    # env.agent_view_size = view_size

    env = minigrid.wrappers.ImgObsWrapper(env)
    env = minigrid.wrappers.ViewSizeWrapper(env, view_size)
    env = MiniGridObservationWrapper(env, view_size)
    return env

def format_minigrid_observation(observation):
    state_chars = ['X', '-', '#', '^']
    n = int(sqrt(len(observation)))
    return '\n'.join(
        ''.join(state_chars[x] for x in observation[i:][:n])
        for i in range(0, n*n, n)
    )