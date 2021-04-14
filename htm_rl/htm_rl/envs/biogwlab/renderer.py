from typing import Tuple, List, Optional, Iterable, Dict

import numpy as np

from htm_rl.common.sdr_encoders import SdrConcatenator
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.view_clipper import ViewClipper, ViewClip


class Renderer:
    output_sdr_size: int

    shape: Tuple[int, int]
    render: List[str]
    sdr_concatenator: Optional[SdrConcatenator]
    view_clipper: Optional[ViewClipper]

    def __init__(self, env, view_rectangle=None):
        self.shape = env.shape
        self.view_clipper = None
        if view_rectangle is not None:
            self.view_clipper = ViewClipper(env.shape, view_rectangle)
        self.sdr_concatenator = None

    def render(self, position, view_direction, entities: Iterable[Entity]):
        view_clip = self.make_view_clip(position, view_direction)

        layers_with_size = []
        for entity in entities:
            if not entity.rendering:
                continue
            layer = entity.render(view_clip)
            if isinstance(layer, list):
                layers_with_size.extend(layer)
            elif layer[1]:
                layers_with_size.append(layer)

        assert layers_with_size, 'Rendering output is empty'
        layers, sizes = zip(*layers_with_size)

        if self.sdr_concatenator is None:
            self.sdr_concatenator = SdrConcatenator(list(sizes))

        observation = self.sdr_concatenator.concatenate(*layers)
        return observation

    def render_rgb(self, position, view_direction, entities: Dict[EntityType, List[Entity]]):
        default_filler = np.array([255, 3, 209])

        img_map = np.empty(self.shape + (3, ), dtype=np.int)
        img_map[:] = default_filler

        areas = entities[EntityType.Area]
        area_color, area_dc = [117, 198, 230], [-12, -15, -6]
        self._draw_entities(img_map, areas, area_color, area_dc)

        obstacles = entities[EntityType.Obstacle]
        obstacle_color, obstacle_dc = [70, 40, 100], [-7, -4, -10]
        self._draw_entities(img_map, obstacles, obstacle_color, obstacle_dc)

        # TODO: reward based coloring
        food = entities[EntityType.Consumable]
        food_color, food_dc = [112, 212, 17], [-4, -10, 4]
        self._draw_entities(img_map, food, food_color, food_dc)

        agent = entities[EntityType.Agent]
        agent_color, agent_dc = [255, 255, 0], [0, 0, 0]
        self._draw_entities(img_map, agent, agent_color, agent_dc)

        view_clip = self.make_view_clip(position, view_direction)
        if view_clip is None:
            return img_map

        img_obs = np.empty(self.view_clipper.view_shape + (3, ), dtype=np.int)
        abs_indices = np.divmod(view_clip.abs_indices, img_map.shape[1])
        view_indices = np.divmod(view_clip.view_indices, img_obs.shape[1])

        img_obs[:] = np.array([0, 0, 0])
        img_obs[view_indices] = img_map[abs_indices].copy()
        img_map[abs_indices] += (.5 * (255 - img_map[abs_indices])).astype(np.int)

        return [img_map, img_obs]

    @property
    def output_sdr_size(self):
        return self.sdr_concatenator.output_sdr_size

    def make_view_clip(self, position, view_direction):
        if self.view_clipper is None:
            return None
        abs_indices, view_indices = self.view_clipper.clip(position, view_direction)
        abs_indices = abs_indices.flatten()
        view_indices = view_indices.flatten()
        return ViewClip(
            shape=self.view_clipper.view_shape,
            abs_indices=abs_indices,
            view_indices=view_indices
        )

    @staticmethod
    def _draw_entities(img: np.ndarray, entities: List[Entity], color: List[int], dc: List[int]):
        mask = np.empty(img.shape[:2], dtype=np.bool)
        color, dc = np.array(color), np.array(dc)
        for entity in entities:
            mask.fill(0)
            entity.append_mask(mask)
            img[mask] = color
            color += dc


def render_mask(mask: np.ndarray, view_clip: ViewClip = None):
    if view_clip is None:
        return np.flatnonzero(mask), mask.size

    clipped_mask = np.zeros(view_clip.shape, dtype=np.int).flatten()
    clipped_mask[view_clip.view_indices] = mask.flatten()[view_clip.abs_indices]
    return np.flatnonzero(clipped_mask), clipped_mask.size
