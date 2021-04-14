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
        self.view_shape = env.shape

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
        img = np.empty(self.shape + (3, ), dtype=np.int)
        img[:] = np.array([255, 3, 209])

        areas = entities[EntityType.Area]
        area_color, dc = [117, 198, 230], [-12, -15, -6]
        self._draw_entities(img, areas, area_color, dc)

        obstacles = entities[EntityType.Obstacle]
        obstacle_color, dc = [70, 70, 110], [-8, -8, 0]
        self._draw_entities(img, obstacles, obstacle_color, dc)

        # TODO: reward based coloring
        food = entities[EntityType.Consumable]
        food_color, dc = [112, 212, 17], [-4, -10, 4]
        self._draw_entities(img, food, food_color, dc)

        agent = entities[EntityType.Agent]
        agent_color, dc = [255, 255, 0], [0, 0, 0]
        self._draw_entities(img, agent, agent_color, dc)

        # view_clip = self.make_view_clip(position, view_direction)
        return [img]

    @property
    def output_sdr_size(self):
        return self.sdr_concatenator.output_sdr_size

    def make_view_clip(self, position, view_direction):
        if self.view_clipper is None:
            return None
        abs_indices, view_indices = self.view_clipper.clip(position, view_direction)
        abs_indices = abs_indices.flatten()
        view_indices = view_indices.flatten()
        return ViewClip(self.view_shape, view_indices, abs_indices)

    def _draw_entities(self, img: np.ndarray, entities: List[Entity], color: List[int], dc: List[int]):
        mask = np.empty(self.shape, dtype=np.bool)
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
