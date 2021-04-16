from typing import Tuple

import gym
import gym_minigrid as minigrid


class MiniGridObservationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, view_size: int):
        super().__init__(env)
        self.view_size = view_size

    def observation(self, observation):
        x = observation[:, :, 0].T.ravel().copy()
        # make data is categorical on [0, 2] range
        x[x == 8] = 0
        return x


class MiniGridNonSquareObservationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, view_size: Tuple[int, int]):
        super().__init__(env)
        self.view_width, self.view_height = view_size

    def observation(self, observation):
        new_width, orig_width = self.view_size, observation._shape_xy[0]
        left = (orig_width - new_width) // 2
        right = left + new_width
        return observation[left:right, -self.view_height:]


class MiniGridFullObservationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation.ravel()


def make_minigrid(size: int, view_size):
    if size in {5, 6, 8, 16}:
        env = gym.make(f'MiniGrid-Empty-{size}x{size}-v0')
    else:
        env = minigrid.envs.EmptyEnv(size=size)

    if view_size is None:
        env = minigrid.wrappers.FullyObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = MiniGridFullObservationWrapper(env)
    else:
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = minigrid.wrappers.ViewSizeWrapper(env, view_size)
        env = MiniGridObservationWrapper(env, view_size)

    return env

# def format_minigrid_observation(observation):
#     state_chars = ['X', '-', '#', '^']
#     n = int(sqrt(len(observation)))
#     return '\n'.join(
#         ''.join(state_chars[x] for x in observation[i:][:n])
#         for i in range(0, n*n, n)
#     )
#
#
# class MinigridStateFormatter:
#     state_chars = ['X', '-', '#', '^']
#
#     n_rows: int
#     n_cols: int
#
#     def __init__(self, n_rows: int, n_cols: int):
#         self.n_rows = n_rows
#         self.n_cols = n_cols
#
#     def format(self, sar: SarSuperposition) -> str:
#         return ' '.join([
#             self._str_from_state_superposition(sar.state),
#             self._str_from_superposition(sar.action, self.action_chars),
#             self._str_from_superposition(sar.reward, self.reward_chars),
#         ])
#
#     @staticmethod
#     def _str_from_superposition(x: Superposition, mapping: List[str], fixed_len: int = None) -> str:
#         n_x = len(isnone(x, []))
#         n = max(n_x, isnone(fixed_len, len(mapping)))
#         d = n - n_x
#         return ''.join(
#             mapping[x[i - d]] if d <= i else ' '
#             for i in range(n_x + d)
#         )
#
#     def _str_from_state_superposition(self, state: SuperpositionList) -> str:
#         def to_str(x: Superposition) -> str:
#             return self._str_from_superposition(x, self.state_chars)
#
#         n_rows, n_cols = self.n_rows, self.n_cols
#         return '\n'.join(
#             '|'.join(to_str(state[row * n_cols + col]) for col in range(n_cols))
#             for row in range(n_rows)
#         )


# class SarSuperpositionFormatter:
#     state_chars = ['X', '-', '#', '^']
#     action_chars = ['<', '>', '^']
#     reward_chars = ['.', '+', '-']
#
#     n_rows: int
#     n_cols: int
#
#     def __init__(self, n_rows: int, n_cols: int):
#         self.n_rows = n_rows
#         self.n_cols = n_cols
#
#     def format(self, sar: SarSuperposition) -> str:
#         return ' '.join([
#             self._str_from_state_superposition(sar.state),
#             self._str_from_superposition(sar.action, self.action_chars),
#             self._str_from_superposition(sar.reward, self.reward_chars),
#         ])
#
#     @staticmethod
#     def _str_from_superposition(x: Superposition, mapping: List[str], fixed_len: int = None) -> str:
#         n_x = len(isnone(x, []))
#         n = max(n_x, isnone(fixed_len, len(mapping)))
#         d = n - n_x
#         return ''.join(
#             mapping[x[i - d]] if d <= i else ' '
#             for i in range(n_x + d)
#         )
#
#     def _str_from_state_superposition(self, state: SuperpositionList) -> str:
#         def to_str(x: Superposition) -> str:
#             return self._str_from_superposition(x, self.state_chars)
#
#         n_rows, n_cols = self.n_rows, self.n_cols
#         return '\n'.join(
#             '|'.join(to_str(state[row * n_cols + col]) for col in range(n_cols))
#             for row in range(n_rows)
#         )
