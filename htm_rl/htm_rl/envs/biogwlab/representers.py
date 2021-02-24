from itertools import product

import numpy as np
from numpy.random._generator import Generator

from htm_rl.envs.biogwlab.dynamics import BioGwLabEnvDynamics
from htm_rl.envs.biogwlab.environment import BioGwLabEnvironment
from htm_rl.envs.biogwlab.environment_state import BioGwLabEnvState


class BioGwLabStateVisualRepresenter:
    repr_len: int

    def __init__(self, state: BioGwLabEnvState):
        self.repr_len = (
                1 + state.n_types_obstacle + state.n_types_area + state.n_types_food
        )

    def get_representation(self, state: BioGwLabEnvState):
        size = state.size
        vis_repr = np.zeros((size, size, self.repr_len), dtype=np.int8)
        for i, j in product(range(size), range(size)):
            vis_repr[i, j] = self.get_cell_representation(i, j, state)
        return vis_repr

    def get_cell_representation(self, i, j, state: BioGwLabEnvState):
        result = np.zeros(self.repr_len, dtype=np.int8)
        shift = 1
        if state.obstacle_mask[i, j]:
            result[shift + state.obstacle_map[i, j]] = 1
            return result

        shift += state.n_types_obstacle
        result[shift + state.areas_map[i, j]] = 1
        if state.food_mask[i, j]:
            shift += state.n_types_area
            result[shift + state.food_map[i, j]] = 1
        return result

    def init_outer_cells(self, repr):
        repr[:, :, 0] = 1


class BioGwLabStateScentRepresenter:
    rnd: Generator

    def __init__(self, seed: int):
        self.rnd = np.random.default_rng(seed=seed)

    def generate_scent_map(self, state: BioGwLabEnvState):
        channels = state.n_scent_channels
        size = state.size
        scent_map = np.zeros((size, size, channels), dtype=np.int8)
        # for channel in range(channels):
        #     scent = state.food_scent[:, :, channel].ravel()
        #     n_cells = scent.size
        #     n_active = int(.2 * n_cells)
        #     activations = self.rnd.choice(n_cells, p=scent, size=n_active)
        #     scent_map[np.divmod(activations, size), channel] = 1
        return scent_map

    def init_outer_cells(self, repr):
        pass


class BioGwLabEnvRepresentationWrapper(BioGwLabEnvironment):
    visual_representer: BioGwLabStateVisualRepresenter
    scent_representer: BioGwLabStateScentRepresenter

    visual_representation: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.visual_representer = BioGwLabStateVisualRepresenter(self.state)
        self.scent_representer = BioGwLabStateScentRepresenter(self.state.seed)

    # noinspection PyRedundantParentheses
    @property
    def shape(self):
        size = self.state.size
        repr_len = self.visual_representer.repr_len
        return (size, size, repr_len)

    def reset(self):
        _ = super().reset()

        self.visual_representation = self.visual_representer.get_representation(
            self.state
        )
        visual_representation = self.visual_representation
        scent_representation = self.scent_representer.generate_scent_map(self.state)
        return visual_representation, scent_representation

    def step(self, action):
        _, reward, is_done, _ = super().step(action)

        # update current cell representation, because only this one could be changed
        i, j = self.state.agent_position
        self.visual_representation[i, j] = self.visual_representer.get_cell_representation(
            i, j, self.state
        )

        visual_representation = self.visual_representation
        scent_representation = self.scent_representer.generate_scent_map(self.state)
        return (visual_representation, scent_representation), reward, is_done, {}