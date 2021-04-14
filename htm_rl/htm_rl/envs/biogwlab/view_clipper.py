import dataclasses
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER, MoveDynamics


@dataclass
class ViewClip:
    shape: Tuple[int, int]
    abs_indices: np.ndarray
    view_indices: np.ndarray

    def __iter__(self):
        yield from dataclasses.astuple(self)


class ViewClipper:
    base_shape: Tuple[int, int]
    view_rectangle: Tuple[Tuple[int, int], Tuple[int, int]]

    natural_direction: int

    _map_abs_indices_cache: np.ndarray
    _view_indices_cache: np.ndarray

    def __init__(
            self,
            base_shape: Tuple[int, int],
            view_rectangle_xy: Tuple[Tuple[int, int], Tuple[int, int]],
    ):
        self.base_shape = base_shape
        # view direction with natural indices traversal
        self.natural_direction = DIRECTIONS_ORDER.index('down')
        self.view_rectangle = _xy_to_ij(view_rectangle_xy)

        h, w = self.base_shape
        self._map_abs_indices_cache = np.arange(h*w).reshape(self.base_shape)
        h, w = self.view_shape
        self._view_indices_cache = np.arange(h*w).reshape(self.view_shape)

    # noinspection PyRedundantParentheses
    @property
    def view_shape(self):
        (bi, lj), (ui, rj) = self.view_rectangle
        return (ui-bi+1, rj-lj+1)

    def clip(self, position, view_direction):
        # get relative direction
        view_direction = self.natural_direction - view_direction

        # abs rectangle from view rectangle
        abs_rect = self._get_abs_rectangle(self.view_rectangle, position, view_direction)
        abs_indices = self._get_abs_indices(abs_rect, view_direction)

        # [possibly clipped] view rectangle from abs rectangle
        view_rect = self._get_view_rect(abs_rect, position, view_direction)
        view_indices = self._get_view_indices(view_rect)

        return ViewClip(
            shape=self.view_shape,
            abs_indices=abs_indices.flatten(),
            view_indices=view_indices.flatten()
        )

    def _get_abs_rectangle(self, view_rect, position, view_direction):
        #  1. rotate view rectangle; 2. get abs position; 3. clip it
        abs_rect_vector = _rotate_rect(view_rect, view_direction)
        abs_rect = _ground_rectangle(abs_rect_vector, position)
        abs_rect = _clip_rectangle(abs_rect, self.base_shape)
        return abs_rect

    def _get_abs_indices(self, abs_rect, view_direction):
        # 1. get indices rect; 2. rotate to view
        abs_indices = _get_slice(self._map_abs_indices_cache, abs_rect)

        # traverse order is for the natural view direction ==>
        # correct traverse order for the current view direction
        abs_indices = _rotate_matrix(abs_indices, view_direction)
        return abs_indices

    def _get_view_rect(self, abs_rect, position, view_direction):
        # 1. get relative rect from abs; 2. get abs in view rect 2. rotate it back
        abs_rect_vector = _ground_rectangle(abs_rect, position, inverse=True)
        view_rect_vector = _rotate_rect(abs_rect_vector, -view_direction)

        # ground view rect relative to agent view position at point == -bot
        view_rect = _ground_rectangle(view_rect_vector, self.view_rectangle[0], inverse=True)
        return view_rect

    def _get_view_indices(self, clipped_view_rect):
        view_indices = _get_slice(self._view_indices_cache, clipped_view_rect)
        return view_indices


def _get_slice(mat, rect_slice):
    (bi, bj), (ti, tj) = rect_slice
    return mat[bi:ti+1, bj:tj+1]


def _rotate_matrix(mat: np.ndarray, direction: int):
    def rotate_90(m: np.ndarray):
        return np.flip(m.T, axis=1)

    if direction < 0:
        direction += 4

    for _ in range(direction):
        mat = rotate_90(mat)
    return mat


def _clip_rectangle(rect, shape):
    bot, top = rect
    bot = MoveDynamics.clip2d(bot, shape)
    top = MoveDynamics.clip2d(top, shape)
    return bot, top


def _ground_rectangle(rect, position, inverse=False):
    # if inverse, it's opposite - get relative rectangle
    k = 1 if not inverse else -1

    def ground_point(v, p):
        return v[0] + k * p[0], v[1] + k * p[1]

    bot = ground_point(rect[0], position)
    top = ground_point(rect[1], position)
    return bot, top


def _rotate_rect(rect, direction):
    bot, top = rect
    bot = _rotate_point(bot, direction)
    top = _rotate_point(top, direction)
    bot, top = _make_canonical(bot, top)
    return bot, top


def _make_canonical(p1, p2):
    i1, j1 = p1
    i2, j2 = p2
    if i1 > i2:
        i1, i2 = i2, i1
    if j1 > j2:
        j1, j2 = j2, j1
    return (i1, j1), (i2, j2)


def _rotate_point(p, direction):
    def rotate_90(p):
        i, j = p
        return -j, i

    if direction < 0:
        direction += 4
    for _ in range(direction):
        p = rotate_90(p)
    return p


def _xy_to_ij(rect_xy):
    (lx, by), (rx, ty) = rect_xy
    # flip x-axis, because left and right are swapped on upside-down image
    rect_ij = (by, -rx), (ty, -lx)
    return rect_ij
