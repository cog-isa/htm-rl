from typing import Tuple

import numpy as np

from htm_rl.common.plot_utils import plot_grid_images
from htm_rl.common.sdr import dense_to_sparse
from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER


class ViewClipper:
    i_transformation = [1, -1, -1, 1, ]
    j_transformation = [1, 1, -1, -1, ]

    base_shape: Tuple[int, int]
    view_rectangle: Tuple[Tuple[int, int], Tuple[int, int]]

    def __init__(
            self,
            base_shape: Tuple[int, int],
            view_rectangle_xy: Tuple[Tuple[int, int], Tuple[int, int]],
    ):
        self.base_shape = base_shape
        self.view_rectangle = self.xy_to_ij(view_rectangle_xy)

    # noinspection PyRedundantParentheses
    @property
    def view_shape(self):
        (bi, lj), (ui, rj) = self.view_rectangle
        return (ui-bi+1, rj-lj+1)

    def clip(self, position, view_direction):
        # position = 1, 5
        # view_direction = 3
        i, j = position
        i_high, j_high = self.base_shape

        # print(i, j, forward_dir, i_high, j_high)
        # print(obs_rect)

        (bi, lj), (ui, rj) = self.view_rectangle
        absolute_indices = np.arange(i_high * j_high).reshape(self.base_shape)

        # print(absolute_indices)
        # print(position, DIRECTIONS_ORDER[view_direction])

        # print('orig', bi, ui, lj, rj)
        # print('====')
        # for i in range(5):
        #     _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, i)
        #     print('rot', _bi, _ui, _lj, _rj)
        # print('====')

        _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, view_direction)
        # print('rot', _bi, _ui, _lj, _rj)

        _bi = clip(i + _bi, i_high)
        _ui = clip(_ui + i, i_high) + 1
        _lj = clip(j + _lj, j_high)
        _rj = clip(_rj + j, j_high) + 1
        # print('repr', _bi, _ui, _lj, _rj)
        repr = absolute_indices[_bi:_ui, _lj:_rj].copy()
        img = np.zeros(self.base_shape, dtype=np.int8)
        img[_bi:_ui, _lj:_rj] = 1
        img[position] = 2
        # plot_grid_images(img)

        _bi, _ui = _bi - i, _ui - i - 1
        _lj, _rj = _lj - j, _rj - j - 1

        # print('cli', _bi, _ui, _lj, _rj)
        _bi, _ui, _lj, _rj = self._rotate(_bi, _ui, _lj, _rj, -view_direction)
        # print('clip', _bi, _ui, _lj, _rj)

        # lj, rj = -rj, -lj
        # print('orig', bi, ui, lj, rj)

        bi, ui = -bi + _bi, -bi + _ui + 1
        lj, rj = -lj + _lj, -lj + _rj + 1
        # print('res', bi, ui, lj, rj)

        if DIRECTIONS_ORDER[view_direction] == 'up':
            repr = repr
        elif DIRECTIONS_ORDER[view_direction] == 'right':
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 0)
        elif DIRECTIONS_ORDER[view_direction] == 'down':
            repr = np.flip(repr, (0, 1))
        elif DIRECTIONS_ORDER[view_direction] == 'left':
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 1)

        # print('obs shape', obs.shape)
        # print(obs[__bi:__ui, __lj:__rj].shape)
        view_shape = self.view_shape
        view_indices = np.arange(view_shape[0] * view_shape[1]).reshape(view_shape)

        # print(np.flip(view_indices, (0, 1)))
        # print(bi, ui, lj, rj)
        # view_indices = np.flip(view_indices, (0, 1))
        view_indices = view_indices[bi:ui, lj:rj]
        # print(view_indices)
        # print(repr)
        # assert False

        # view_indices = vis_observation.ravel()
        # view_indices = dense_to_sparse(view_indices)
        return repr, view_indices

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

    def xy_to_ij(self, rect):
        (lj, bi), (rj, ui) = rect
        return (bi, -rj), (ui, -lj)