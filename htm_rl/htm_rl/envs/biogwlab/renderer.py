from typing import Tuple, List, Optional, Iterable

import numpy as np

from htm_rl.common.sdr_encoders import SdrConcatenator
from htm_rl.envs.biogwlab.module import Entity
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
        view_clip = None
        if self.view_clipper is not None:
            abs_indices, view_indices = self.view_clipper.clip(position, view_direction)
            abs_indices = abs_indices.flatten()
            view_indices = view_indices.flatten()
            view_clip = self.view_shape, view_indices, abs_indices

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

    @property
    def output_sdr_size(self):
        return self.sdr_concatenator.output_sdr_size


def render_mask(mask: np.ndarray, view_clip: ViewClip = None):
    if view_clip is None:
        return np.flatnonzero(mask), mask.size

    view_shape, abs_indices, view_indices = view_clip

    clipped_mask = np.zeros(view_shape, dtype=np.int).flatten()
    clipped_mask[view_indices] = mask.flatten()[abs_indices]
    return np.flatnonzero(clipped_mask), clipped_mask.size
