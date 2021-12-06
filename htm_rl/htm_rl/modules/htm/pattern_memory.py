from htm_rl.modules.htm.connections import Connections
from htm.bindings.sdr import SDR
import numpy as np
from htm.bindings.math import Random

EPS = 1e-12


class SpatialMemory:
    def __init__(self,
                 overlap_threshold: float,
                 initial_permanence: float = 0.55,
                 permanence_increment: float = 0.1,
                 permanence_decrement: float = 0.05,
                 permanence_threshold: float = 0.5,
                 permanence_forgetting_decrement: float = 0.01,
                 activation_threshold: int = 0):
        self.permanence_forgetting_decrement = permanence_forgetting_decrement
        self.permanence_threshold = permanence_threshold
        self.permanence_decrement = permanence_decrement
        self.permanence_increment = permanence_increment
        self.initial_permanence = initial_permanence
        self.patterns = np.empty(0)
        self.permanence = np.empty(0)
        self.unique_id = np.empty(0)
        self.overlap_threshold = overlap_threshold
        self.activation_threshold = activation_threshold

        self.id_counter = 0

    def __len__(self):
        return self.patterns.shape[0]

    def add(self, dense_pattern: np.array):
        if dense_pattern.sum() >= self.activation_threshold:
            if self.patterns.size == 0:
                self.patterns = dense_pattern.reshape((1, -1))
                self.permanence = np.array([self.initial_permanence])
                self.unique_id = np.array([self.id_counter])
                self.id_counter += 1
            else:
                pattern_sizes = self.patterns.sum(axis=1)
                overlaps = 1 - np.sum(np.abs(self.patterns - dense_pattern), axis=1) / (pattern_sizes + EPS)

                if np.any(overlaps >= self.overlap_threshold):
                    best_index = np.argmax(overlaps)
                    self.patterns[best_index] = dense_pattern
                    self.permanence[best_index] += self.permanence_increment
                else:
                    self.patterns = np.vstack([self.patterns, dense_pattern.reshape(1, -1)])
                    self.permanence = np.append(self.permanence, self.initial_permanence)
                    self.unique_id = np.append(self.unique_id, self.id_counter)
                    self.id_counter += 1

    def reinforce(self, values: np.array):
        """
        Reinforce plausible patterns, forget patterns with permanence under threshold
        :param values: array of values from BG
        :return:
        """
        values -= values.mean()
        values /= (values.std() + EPS)
        positive = values > 0
        values[positive] *= self.permanence_increment
        values[~positive] *= self.permanence_decrement

        self.permanence += values
        patterns_alive = self.permanence > self.permanence_threshold
        self.patterns = self.patterns[patterns_alive]
        self.permanence = self.permanence[patterns_alive]
        self.unique_id = self.unique_id[patterns_alive]

    def forget(self):
        self.permanence -= self.permanence_forgetting_decrement
        patterns_alive = self.permanence > self.permanence_threshold
        self.patterns = self.patterns[patterns_alive]
        self.permanence = self.permanence[patterns_alive]
        self.unique_id = self.unique_id[patterns_alive]

    def get_options(self, dense_pattern, return_indices=False):
        if self.patterns.size == 0:
            if return_indices:
                return list(), list()
            else:
                return list()
        else:
            overlaps = np.dot(dense_pattern, self.patterns.T)
            pattern_sizes = self.patterns.sum(axis=1)
            overlaps = overlaps / pattern_sizes
            indices = np.flatnonzero(overlaps >= self.overlap_threshold)
            options = self.patterns[indices]
            # convert to list of sparse representations
            if return_indices:
                return [np.flatnonzero(option) for option in options], indices
            else:
                return [np.flatnonzero(option) for option in options]

    def get_sparse_patterns(self):
        return [np.flatnonzero(pattern) for pattern in self.patterns]

    def get_options_by_id(self, unique_ids):
        return self.patterns[np.in1d(self.unique_id, unique_ids)]


class PatternMemory:
    def __init__(self,
                 input_size,
                 max_segments,
                 min_distance,
                 permanence_increment=0.1,
                 permanence_decrement=0.01,
                 segment_decrement=0.1,
                 permanence_connected_threshold=0.5,
                 seed=0):
        self.seed = seed
        self._rng = Random(seed)

        self.input_size = input_size
        self.max_segments = max_segments
        self.min_distance = min_distance
        self.permanence_connected_threshold = permanence_connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.segment_decrement = segment_decrement
        self.learning_threshold = 1 - self.min_distance

        self.connections = Connections(1, connectedThreshold=self.permanence_connected_threshold,
                                       timeseries=False)

        self.active_segment = None

    def get_pattern(self, segment):
        synapses = self.connections.synapsesForSegment(segment)
        connected = [self.connections.presynapticCellForSynapse(syn) for syn in synapses
                     if self.connections.permanenceForSynapse(syn) >= self.permanence_connected_threshold]
        return connected

    def compute(self, input_pattern: SDR, learn: bool):
        overlap = self.connections.computeActivity(input_pattern, learn)
        num_connected = np.array([self.connections.numConnectedSynapses(seg) for seg in range(len(overlap))])
        score = overlap / (num_connected + EPS)

        active_segments = np.flatnonzero(score > self.learning_threshold)
        if active_segments.size > 0:
            self.active_segment = active_segments[np.argmax(score[active_segments])]
            if learn:
                self.connections.adaptSegment(self.active_segment, input_pattern,
                                              self.permanence_increment, self.permanence_decrement)
                max_new = input_pattern.sparse.size
                self.connections.growSynapses(self.active_segment, input_pattern.sparse,
                                              self.permanence_connected_threshold, self._rng, max_new)
        elif learn:
            new_segment = self.connections.createSegment(0, self.max_segments)
            self.connections.growSynapses(new_segment, input_pattern.sparse, self.permanence_connected_threshold, self._rng,
                                          input_pattern.sparse.size)
        else:
            self.active_segment = None

        return self.active_segment


class KohonenPatternMemory:
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
                 learning_rate=1,
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
        self.learning_rate = learning_rate

        self.connections = Connections(self.num_cells, connectedThreshold=self.permanence_connected_threshold,
                                       timeseries=False)

        for cell in range(self.num_cells):
            segment = self.connections.createSegment(cell, 1)
            syn_permanences = self._rng.uniform(low=0.0,
                                                high=1.0,
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
            if top_k_segments.size > 0:
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
                learning_rates = factors * self.learning_rate

                self._learn(input_pattern, learning_segments, segments_to_punish, n_syn_to_punish,
                            permanence_increments, permanence_decrements, learning_rates)

            # update cells' activity
            cells_activity = np.zeros_like(self.cells_activity)
            cells_activity[top_k_segments] = 1
            self.cells_activity = self.cells_activity * self.sm_activity + (1 - self.sm_activity) * cells_activity

            if cell_with_min_activity is not None:
                self.cells_activity[cell_with_min_activity] = 1

        return self.active_cell

    def _boost_cell(self, active_input, cell):
        self.connections.adaptSegment(cell, active_input, self.permanence_increment * self.boost_factor, self.permanence_decrement * self.boost_factor)

    def _learn(self, active_cells, learning_segments, segments_to_punish, n_syn_to_punish, permanence_increments, permanence_decrements,
               learning_rates):
        active_cells_noise = SDR(self.input_size)

        for segment, permanence_increment, permanence_decrement, learning_rate in zip(learning_segments, permanence_increments, permanence_decrements, learning_rates):
            active_cells_noise.sparse = self._rng.choice(active_cells.sparse, int(active_cells.sparse.size * self.learning_rate), replace=False)
            self.connections.adaptSegment(segment, active_cells_noise, permanence_increment, 0, False, 0)
            active_cells_noise.sparse = self._rng.choice(np.setdiff1d(np.arange(self.input_size), active_cells.sparse),
                                                         int((self.input_size - active_cells.sparse.size) * (1 - self.learning_rate)),
                                                         replace=False)
            active_cells_noise.sparse = np.union1d(active_cells_noise.sparse, active_cells.sparse)
            self.connections.adaptSegment(segment, active_cells_noise, 0, permanence_decrement,
                                          False, 0)

        for segment, n_syn in zip(segments_to_punish, n_syn_to_punish):
            active_cells_noise.sparse = self._rng.choice(active_cells.sparse, n_syn, replace=False)
            self.connections.adaptSegment(segment, active_cells_noise, -self.segment_decrement, 0, False, 0)
