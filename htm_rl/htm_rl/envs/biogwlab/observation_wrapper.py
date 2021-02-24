from typing import Tuple

import numpy as np

from htm_rl.common.sdr import dense_to_sparse
from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab.representers import BioGwLabEnvRepresentationWrapper


class BioGwLabEnvObservationWrapper(BioGwLabEnvRepresentationWrapper):
    i_transformation = [1, -1, -1, 1, ]
    j_transformation = [1, 1, -1, -1, ]

    view_rect: Tuple[Tuple[int, int], Tuple[int, int]]
    scent_rect: Tuple[Tuple[int, int], Tuple[int, int]]

    def __init__(
            self,
            *args,
            view_rect: Tuple[Tuple[int, int], Tuple[int, int]] = None,
            scent_rect: Tuple[Tuple[int, int], Tuple[int, int]] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.view_rect = self._view_to_machine(view_rect)
        self.scent_rect = self._view_to_machine(scent_rect)

    # noinspection PyRedundantParentheses
    @property
    def shape(self):
        (bi, lj), (ui, rj) = self.view_rect
        repr_len = super().shape[-1]
        return (ui-bi+1, rj-lj+1, repr_len)

    @property
    def output_shape(self):
        return self.shape

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def reset(self):
        vis_repr, scent_repr = super().reset()
        observation = self._to_observation(vis_repr, scent_repr)
        # print(observation.shape)
        return observation

    def step(self, action):
        (vis_repr, scent_repr), reward, is_done, _ = super().step(action)
        observation = self._to_observation(vis_repr, scent_repr)
        return observation, reward, is_done, {}

    def _to_observation(self, vis_repr, scent_repr):
        vis_observation = self._clip_observation(
            vis_repr, self.view_rect, self.visual_representer.init_outer_cells
        )
        # scent_observation = self._clip_observation(
        #     scent_repr, self.scent_rect, self.scent_representer.init_outer_cells
        # )
        #
        # observation = np.concatenate((vis_observation, scent_observation), axis=None)
        # return observation
        obs = vis_observation.ravel()
        obs = dense_to_sparse(obs)
        return obs

    def _clip_observation(self, full_repr, obs_rect, init_obs):
        state = self.state
        i, j = state.agent_position
        forward_dir = state.agent_direction

        i_high, j_high = self.state.size, self.state.size
        # print(i, j, forward_dir, i_high, j_high)
        # print(obs_rect)

        (bi, lj), (ui, rj) = obs_rect
        obs = np.zeros(self.shape, dtype=np.int8)
        init_obs(obs)

        # print('orig', bi, ui, lj, rj)
        # print('====')
        # for i in range(5):
        #     _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, i)
        #     print('rot', _bi, _ui, _lj, _rj)
        # print('====')

        _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, forward_dir)
        # print('rot', _bi, _ui, _lj, _rj)

        _bi = clip(i + _bi, i_high)
        _ui = clip(_ui + i, i_high) + 1
        _lj = clip(j + _lj, j_high)
        _rj = clip(_rj + j, j_high) + 1
        # print('repr', _bi, _ui, _lj, _rj)
        repr = full_repr[_bi:_ui, _lj:_rj].copy()
        # print(repr.shape)

        _bi, _ui = _bi - i, _ui - i - 1
        _lj, _rj = _lj - j, _rj - j - 1

        # print('cli', _bi, _ui, _lj, _rj)
        _bi, _ui, _lj, _rj = self._rotate(_bi, _ui, _lj, _rj, 2 - forward_dir)
        # print('clip', _bi, _ui, _lj, _rj)

        bi = _bi - bi
        ui = bi - _bi + _ui + 1
        lj = _lj - lj
        rj = lj - _lj + _rj + 1
        # print('res', bi, ui, lj, rj)

        if forward_dir == 0:
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 0)
        elif forward_dir == 1:
            repr = repr
        elif forward_dir == 2:
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 1)
        else:
            repr = np.flip(repr, (0, 1))

        # print('obs shape', obs.shape)
        # print(obs[__bi:__ui, __lj:__rj].shape)
        obs[bi:ui, lj:rj] = repr
        return obs

    def _rotate(self, bi, ui, lj, rj, direction):
        bi *= self.i_transformation[direction - 1]
        ui *= self.i_transformation[direction - 1]
        lj *= self.j_transformation[direction - 1]
        rj *= self.j_transformation[direction - 1]
        if direction % 2 == 0:
            bi, lj = lj, bi
            ui, rj = rj, ui
        if bi > ui:
            bi, ui = ui, bi
        if lj > rj:
            lj, rj = rj, lj
        return bi, ui, lj, rj

    def _view_to_machine(self, rect):
        (lj, bi), (rj, ui) = rect
        return (bi, -rj), (ui, -lj)