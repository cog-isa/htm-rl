from typing import Optional

import numpy as np

from hima.envs.biogwlab.view_clipper import ViewClipper, _xy_to_ij


class EnvShapeParams:
    # effective env shape
    env_shape: tuple[int, int]
    # top left angle point of the effective env in full shape
    top_left_point: tuple[int, int]
    # full shape with outer walls
    full_shape: tuple[int, int]

    view_clipper: Optional[ViewClipper]

    def __init__(
            self, env_shape_xy: tuple[int, int],
            view_rectangle_xy: tuple[tuple[int, int], tuple[int, int]] = None
    ):
        # convert from x,y to i,j
        width, height = env_shape_xy
        self.env_shape = (height, width)

        if view_rectangle_xy is not None:
            # extend env with "outer walls" zone
            view_rectangle_ij = _xy_to_ij(rect_xy=view_rectangle_xy)
            self.full_shape, self.top_left_point = self._get_shape_with_walls(
                self.env_shape, view_rectangle_ij
            )
            self.view_clipper = ViewClipper(self.full_shape, view_rectangle_ij)
        else:
            # no outer walls needed
            self.full_shape = self.env_shape
            self.top_left_point = (0, 0)
            self.view_clipper = None

    def get_inner_area(self, full_area: np.ndarray) -> np.ndarray:
        i0, j0 = self.top_left_point
        h, w = self.env_shape
        return full_area[i0:i0 + h, j0:j0 + w]

    def set_inner_area(self, full_area: np.ndarray, new_val: np.ndarray):
        i0, j0 = self.top_left_point
        h, w = self.env_shape
        full_area[i0:i0 + h, j0:j0 + w] = new_val

    def shift_relative_to_corner(self, p: tuple[int, int]) -> tuple[int, int]:
        i0, j0 = self.top_left_point
        i, j = p
        return i0 + i, j0 + j

    @staticmethod
    def _get_shape_with_walls(
            base_shape: tuple[int, int],
            view_rectangle: tuple[tuple[int, int], tuple[int, int]]
    ):
        h, w = base_shape
        (low_h, low_w), (high_h, high_w) = view_rectangle
        low_h = max(-low_h, 0)
        low_w = max(-low_w, 0)
        high_h = max(high_h, 0)
        high_w = max(high_w, 0)

        full_shape = (h + low_h + high_h, w + low_w + high_w)
        top_left_point = low_w, high_h
        return full_shape, top_left_point