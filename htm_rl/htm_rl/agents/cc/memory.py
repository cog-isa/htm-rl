from htm_rl.agents.htm.connections import Connections
from htm.bindings.sdr import SDR
import numpy as np

EPS = 1e-12


class PatternMemory:
    def __init__(self,
                 input_size,
                 num_cells,
                 radius,
                 min_distance,
                 permanence_increment=0.1,
                 permanence_decrement=0.01,
                 segment_decrement=0.1,
                 permanence_connected_threshold=0.5,
                 activation_threshold=1,
                 activity_period=1000,
                 min_activity_threshold=0.001,
                 boost_factor=2,
                 seed=0):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.input_size = input_size
        self.num_cells = num_cells
        self.radius = radius
        self.min_distance = min_distance
        self.permanence_connected_threshold = permanence_connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.segment_decrement = segment_decrement
        self.activation_threshold = activation_threshold
        self.min_activity_threshold = min_activity_threshold
        self.boost_factor = boost_factor
        self.learning_threshold = 1 - self.min_distance

        self.connections = Connections(self.num_cells, connectedThreshold=self.permanence_connected_threshold,
                                       timeseries=False)

        for cell in range(self.num_cells):
            segment = self.connections.createSegment(cell, 1)
            syn_permanences = self._rng.uniform(low=self.permanence_connected_threshold-self.permanence_increment*0.1,
                                                high=self.permanence_connected_threshold+self.permanence_increment*0.1,
                                                size=self.input_size)
            for pre_syn_cell, permanence in zip(range(self.input_size), syn_permanences):
                self.connections.createSynapse(segment, pre_syn_cell, permanence)

        self.cells_activity = np.ones(self.num_cells, dtype='float32')
        self.sm_activity = 1 - 1/activity_period

        self.active_cell = None

    def get_pattern(self, cell):
        synapses = self.connections.synapsesForSegment(cell)
        connected = [self.connections.presynapticCellForSynapse(syn) for syn in synapses if self.connections.permanenceForSynapse(syn) >= self.permanence_connected_threshold]
        return connected

    def compute(self, input_pattern: SDR, learn: bool):
        overlap = self.connections.computeActivity(input_pattern, learn)
        num_connected = np.array([self.connections.numConnectedSynapses(seg) for seg in range(len(overlap))])
        score = overlap / (num_connected + EPS)

        top_k_segments = np.argpartition(score, -self.radius)[-self.radius:]
        top_k_segments = top_k_segments[overlap[top_k_segments] > self.activation_threshold]
        top_k_segments = top_k_segments[np.argsort(-score[top_k_segments])]

        if top_k_segments.size > 0:
            max_score = score[top_k_segments[0]]
        else:
            max_score = 0

        if (max_score < self.learning_threshold) and learn:
            # try to find a cell with minimum activity for a period
            cell_with_min_activity = np.argmin(self.cells_activity)
            if (self.cells_activity[cell_with_min_activity] <= self.min_activity_threshold) or (top_k_segments.size == 0):
                # force cell to become active
                self._boost_cell(input_pattern, cell_with_min_activity)
                self.active_cell = cell_with_min_activity
            else:
                self.active_cell = top_k_segments[0]
                cell_with_min_activity = None
        else:
            if top_k_segments.size > 0:
                self.active_cell = top_k_segments[0]
            else:
                self.active_cell = None

            cell_with_min_activity = None

        if learn:
            # learn top k segments which don't exceed learning threshold and punish other
            learning_segments = top_k_segments[score[top_k_segments] < self.learning_threshold]
            segments_to_punish = top_k_segments[score[top_k_segments] > self.learning_threshold]
            # exclude best segment
            if segments_to_punish.size > 1:
                segments_to_punish = segments_to_punish[1:]
            n_syn_to_punish = (overlap[segments_to_punish] * self.min_distance).astype('uint32')

            factors = np.exp(-(1-score[top_k_segments])*3)
            factors[0] = 1
            permanence_increments = factors * self.permanence_increment
            permanence_decrements = factors * self.permanence_decrement

            self._learn(input_pattern, learning_segments, segments_to_punish, n_syn_to_punish,
                        permanence_increments, permanence_decrements)

            # update cells' activity
            cells_activity = np.zeros_like(self.cells_activity)
            cells_activity[top_k_segments] = 1
            self.cells_activity = self.cells_activity * self.sm_activity + (1 - self.sm_activity) * cells_activity

            if cell_with_min_activity is not None:
                self.cells_activity[cell_with_min_activity] = 1

        return self.active_cell

    def _boost_cell(self, active_input, cell):
        self.connections.adaptSegment(cell, active_input, self.permanence_increment * self.boost_factor, self.permanence_decrement * self.boost_factor)

    def _learn(self, active_cells, learning_segments, segments_to_punish, n_syn_to_punish, permanence_increments, permanence_decrements):

        for segment, permanence_increment, permanence_decrement in zip(learning_segments, permanence_increments, permanence_decrements):
            self.connections.adaptSegment(segment, active_cells, permanence_increment, permanence_decrement, False, 0)

        active_cells_noise = SDR(active_cells.dense.size)
        for segment, n_syn in zip(segments_to_punish, n_syn_to_punish):
            active_cells_noise.sparse = self._rng.choice(active_cells.sparse, n_syn, replace=False)
            self.connections.adaptSegment(segment, active_cells_noise, -self.segment_decrement, 0, False, 0)
