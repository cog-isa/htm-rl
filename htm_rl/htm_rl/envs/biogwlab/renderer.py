from typing import Tuple, List, Optional

import numpy as np

from htm_rl.common.sdr_encoders import SdrConcatenator, IntBucketEncoder, IntArrayEncoder
from htm_rl.envs.biogwlab.move_dynamics import MOVE_DIRECTIONS
from htm_rl.envs.biogwlab.view_clipper import ViewClipper


class Renderer:
    output_sdr_size: int

    shape: Tuple[int, int]
    view_shape: Tuple[int, int]
    render: List[str]
    sdr_concatenator: SdrConcatenator
    view_clipper: Optional[ViewClipper]

    def __init__(self, env, render: List[str], view_rectangle=None, **renderer):
        self.encoders = dict()
        self._cached_renderings = dict()

        self.shape = env.shape
        self.view_shape = env.shape
        self.view_clipper = None
        if view_rectangle is not None:
            self.view_clipper = ViewClipper(env.shape, view_rectangle)
            self.view_shape = self.view_clipper.view_shape

        for data_name in render:
            if data_name == 'position':
                n_cells = env.shape[0] * env.shape[1]
                encoder = self.get_agent_position_renderer(env.agent, n_cells, **renderer)
            elif data_name == 'direction':
                encoder = self.get_agent_direction_renderer(env.agent, **renderer)
            else:
                encoder = self.get_renderer(env.get_module(data_name))

            if encoder is not None:
                self.encoders[data_name] = encoder

        self.output_sdr_size = 0
        if len(self.encoders) == 1:
            _, encoder = list(self.encoders.values())[0]
            self.output_sdr_size = encoder.output_sdr_size
        else:
            encoders = [encoder for _, encoder in self.encoders.values()]
            self._encoding_sdr_concatenator = SdrConcatenator(encoders)
            self.output_sdr_size = self._encoding_sdr_concatenator.output_sdr_size

    def render(self, position, view_direction):
        observation = []

        view_clip = None
        if self.view_clipper is not None:
            abs_indices, view_indices = self.view_clipper.clip(position, view_direction)
            abs_indices = abs_indices.flatten()
            view_indices = view_indices.flatten()
            view_clip = view_indices, abs_indices

        for data_name, (entity, encoder) in self.encoders.items():
            if data_name == 'position':
                agent = entity
                position_fl = self._flatten_position(agent.position)
                encoded_data = encoder.encode(position_fl)
            elif data_name == 'direction':
                agent = entity
                encoded_data = encoder.encode(agent.view_direction)
            else:
                encoded_data = self.render_entity(entity, encoder, view_clip)

            observation.append(encoded_data)

        if len(observation) == 1:
            return observation[0]
        else:
            return self._encoding_sdr_concatenator.concatenate(*observation)

    def get_agent_position_renderer(self, agent, n_cells, **renderer):
        return agent, IntBucketEncoder(n_values=n_cells, **renderer)

    def get_agent_direction_renderer(self, agent, **renderer):
        return agent, IntBucketEncoder(n_values=len(MOVE_DIRECTIONS), **renderer)

    def get_renderer(self, entity):
        if entity.n_types == 1 and entity.mask is None:
            return None
        return entity, IntArrayEncoder(shape=self.view_shape, n_types=entity.n_types)

    def _flatten_position(self, position):
        i, j = position
        return i * self.shape[1] + j

    def render_entity(self, entity, encoder, view_clip):
        if view_clip is None:
            return encoder.encode(entity.map, entity.mask)

        view_indices, abs_indices = view_clip

        clipped_mask = None
        if entity.mask is not None:
            clipped_mask = np.ones(self.view_shape, dtype=np.bool).flatten()
            clipped_mask[view_indices] = entity.mask.flatten()[abs_indices]

        clipped_map = None
        if entity.map is not None:
            clipped_map = np.zeros(self.view_shape, dtype=np.int).flatten()
            clipped_map[view_indices] = entity.map.flatten()[abs_indices]
        return encoder.encode(clipped_map, clipped_mask)