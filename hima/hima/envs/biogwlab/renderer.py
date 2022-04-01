from typing import Optional, Iterable

import numpy as np

from hima.common.sdr_encoders import SdrConcatenator
from hima.envs.biogwlab.env_shape_params import EnvShapeParams
from hima.envs.biogwlab.module import Entity, EntityType
from hima.envs.biogwlab.view_clipper import ViewClipper, ViewClip


class Renderer:
    shape: EnvShapeParams
    view_clipper: Optional[ViewClipper]

    channels_concatenator: Optional[SdrConcatenator]

    def __init__(self, shape_xy, view_rectangle=None):
        self.shape = EnvShapeParams(shape_xy, view_rectangle)
        self.view_clipper = self.shape.view_clipper

        # delayed initialization on the first render call
        self.channels_concatenator = None

    def render(self, position, view_direction, entities: Iterable[Entity]):
        view_clip = self.make_view_clip(position, view_direction)

        layers_with_sdr_size = []
        for entity in entities:
            if not entity.rendering:
                continue
            layer_with_sdr_size = entity.render(view_clip)
            if isinstance(layer_with_sdr_size, list):
                layers_with_sdr_size.extend(layer_with_sdr_size)
            elif layer_with_sdr_size[1]:
                layers_with_sdr_size.append(layer_with_sdr_size)

        assert layers_with_sdr_size, 'Rendering output is empty'
        layers, sdr_sizes = zip(*layers_with_sdr_size)

        if self.channels_concatenator is None:
            self.channels_concatenator = SdrConcatenator(list(sdr_sizes))

        observation = self.channels_concatenator.concatenate(*layers)
        return observation

    def render_rgb(
            self, position, view_direction,
            entities: dict[EntityType, list[Entity]],
            show_outer_walls: bool
    ):
        # fill with magenta to catch non-colored cells
        default_filler = np.array([255, 3, 209])

        img_map = np.empty(self.shape.full_shape + (3, ), dtype=np.int)
        img_map[:] = default_filler

        areas = entities[EntityType.Area]
        # areas: light blue
        area_color, area_dc = [117, 198, 230], [-12, -15, -6]
        self._draw_entities(img_map, areas, area_color, area_dc)

        obstacles = entities[EntityType.Obstacle]
        # obstacles: dark blue
        obstacle_color, obstacle_dc = [70, 40, 100], [-7, -4, -10]
        self._draw_entities(img_map, obstacles, obstacle_color, obstacle_dc)

        food = entities[EntityType.Consumable]
        # consumables: salad green
        food_color, food_dc = [112, 212, 17], [-4, -10, 4]
        self._draw_entities(img_map, food, food_color, food_dc)

        agent = entities[EntityType.Agent]
        # agent: yellow
        agent_color, agent_dc = [255, 255, 0], [0, 0, 0]
        self._draw_entities(img_map, agent, agent_color, agent_dc)

        view_clip = self.make_view_clip(position, view_direction)
        if view_clip is None:
            return img_map

        img_obs = np.empty(self.view_clipper.view_shape + (3, ), dtype=np.int)
        abs_indices = np.divmod(view_clip.abs_indices, img_map.shape[1])
        view_indices = np.divmod(view_clip.view_indices, img_obs.shape[1])

        # fill with `out-of-map` obstacles: black
        img_obs[:] = np.array([0, 0, 0])
        img_obs[view_indices] = img_map[abs_indices].copy()
        img_obs = np.flip(img_obs, axis=[0, 1])     # from ij to xy

        # `grey`-out view area
        img_map[abs_indices] += (.5 * (255 - img_map[abs_indices])).astype(np.int)

        if not show_outer_walls:
            # cut outer walls, keeping only "inner" env part
            img_map = self.shape.get_inner_area(img_map)
        return [img_map, img_obs]

    @property
    def output_sdr_size(self) -> int:
        return self.channels_concatenator.output_sdr_size

    def make_view_clip(self, position, view_direction):
        if self.view_clipper is None:
            return None
        return self.view_clipper.clip(position, view_direction)

    @staticmethod
    def _draw_entities(
            img: np.ndarray, entities: list[Entity], color: list[int],
            delta_color: list[int]
    ):
        mask = np.empty(img.shape[:2], dtype=np.bool)
        # color: RGB, i.e. 3-elem array
        color, delta_color = np.array(color), np.array(delta_color)
        for entity in entities:
            mask.fill(0)
            entity.append_mask(mask)
            img[mask] = color
            color += delta_color


def render_mask(mask: np.ndarray, view_clip: ViewClip) -> tuple[np.ndarray, int]:
    if view_clip is None:
        return np.flatnonzero(mask), mask.size

    clipped_mask = np.zeros(view_clip.shape, dtype=np.int).flatten()
    clipped_mask[view_clip.view_indices] = mask.flatten()[view_clip.abs_indices]
    return np.flatnonzero(clipped_mask), clipped_mask.size
