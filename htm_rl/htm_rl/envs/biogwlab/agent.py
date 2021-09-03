from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.move_dynamics import MoveDynamics, DIRECTIONS_ORDER, MOVE_DIRECTIONS
from htm_rl.envs.biogwlab.view_clipper import ViewClip

if TYPE_CHECKING:
    from htm_rl.envs.biogwlab.environment import Environment


class AgentRenderer:
    position_encoder: Optional[IntBucketEncoder]
    view_direction_encoder: Optional[IntBucketEncoder]

    def __init__(self, env_shape: tuple[int, int], what: list[str], **encoder):
        n_cells = env_shape[0] * env_shape[1]
        self.position_encoder = None
        self.view_direction_encoder = None

        if 'position' in what:
            self.position_encoder = IntBucketEncoder(n_values=n_cells, **encoder)
        if 'view direction' in what:
            self.view_direction_encoder = IntBucketEncoder(
                n_values=len(MOVE_DIRECTIONS), **encoder
            )

    def render(self, position, view_direction):
        result = []
        if self.position_encoder:
            result.append(self._render_one(self.position_encoder, position))
        if self.view_direction_encoder:
            result.append(self._render_one(self.view_direction_encoder, view_direction))
        return result

    @staticmethod
    def _render_one(encoder: IntBucketEncoder, value: int):
        return encoder.encode(value), encoder.output_sdr_size


class Agent(Entity):
    family = 'agent'
    type = EntityType.Agent

    position: tuple[int, int]
    view_direction: Optional[int]

    renderer: AgentRenderer
    env: Environment

    def __init__(self, env: Environment, rendering=False, **entity):
        super(Agent, self).__init__(rendering=rendering, **entity)
        self.env = env

        if rendering:
            assert isinstance(rendering, dict)
            self.renderer = AgentRenderer(env_shape=env.shape, **rendering)

    def generate(self, seeds):
        # we should not take this entity into account
        # in aggregated masks during generation
        self.initialized = False

        rng = np.random.default_rng(seeds['agent'])
        empty_positions_mask = ~self.env.aggregated_mask[EntityType.NonEmpty]

        empty_positions_fl = np.flatnonzero(empty_positions_mask)
        position_fl = rng.choice(empty_positions_fl)

        self.position = self._unflatten_position(position_fl)
        self.view_direction = rng.choice(len(DIRECTIONS_ORDER))
        self.initialized = True

    def move(self, direction):
        obstacles = self.env.aggregated_mask[EntityType.Obstacle]
        self.position, success = MoveDynamics.try_move(
            self.position, direction, self.env.shape, obstacles
        )

    def turn(self, turn_direction):
        self.view_direction = MoveDynamics.turn(self.view_direction, turn_direction)

    def render(self, view_clip: ViewClip = None):
        position_fl = self.position[0] * self.env.shape[1] + self.position[1]
        return self.renderer.render(position_fl, self.view_direction)

    def append_mask(self, mask: np.ndarray):
        if self.initialized:
            mask[self.position] = 1

    def append_position(self, exist: bool, position):
        return exist or (self.initialized and self.position == position)

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.env.shape[1])


