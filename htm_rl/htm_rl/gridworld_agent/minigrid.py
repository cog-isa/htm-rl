import gym
import gym_minigrid as minigrid


class MiniGridObservationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        x = observation[:, :, 0].T.ravel().copy()
        # make data is categorical on [0, 2] range
        x[x == 8] = 0
        return x


def make_minigrid(size: int, view_size: int):
    assert size in {5, 6, 8, 16}
    env = gym.make(f'MiniGrid-Empty-{size}x{size}-v0')

    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(view_size, view_size, 3),
        dtype='uint8'
    )
    env.observation_space = gym.spaces.Dict({
        'image': observation_space
    })
    env.agent_view_size = view_size

    env = minigrid.wrappers.ImgObsWrapper(env)
    env = MiniGridObservationWrapper(env)
    return env