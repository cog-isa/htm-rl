from htm.advanced.algorithms.connections import Connections
from htm.bindings.sdr import SDR
from htm.bindings.math import Random
import numpy as np


class Muscles:
    def __init__(self,
                 input_size,
                 muscles_size,
                 activation_threshold,
                 learning_threshold,
                 permanence_increment,
                 permanence_decrement,
                 depolarized_decrement,
                 connected_permanence,
                 initial_permanence,
                 max_synapses_per_segment,
                 sample_size,
                 max_segments_per_cell,
                 seed):
        self.max_segments_per_cell = max_segments_per_cell
        self.initial_permanence = initial_permanence
        self.depolarized_decrement = depolarized_decrement
        self.connected_permanence = connected_permanence
        self.sample_size = sample_size
        self.max_synapses_per_segment = max_synapses_per_segment
        self.permanence_decrement = permanence_decrement
        self.permanence_increment = permanence_increment
        self.learning_threshold = learning_threshold
        self.input_size = input_size
        self.muscles_size = muscles_size
        self.activation_threshold = activation_threshold

        self.total_size = self.input_size + self.muscles_size

        self.connections = Connections(self.total_size,
                                       self.connected_permanence)
        self.input_range = (0, self.input_size)
        self.muscles_range = (self.input_range[1], self.input_range[1] + self.muscles_size)

        self.active_muscles = SDR(self.total_size)
        self.active_input = SDR(self.total_size)
        self.depolarized_muscles = SDR(self.total_size)

        self.active_segments = np.empty(0)
        self.matching_segments = np.empty(0)
        self.num_potential = np.empty(0)

        if seed:
            self.rng = Random(seed)
        else:
            self.rng = Random()

    def set_active_muscles(self, sparse_pattern):
        self.active_muscles.sparse = np.array(sparse_pattern, dtype='uint32') + self.muscles_range[0]

    def set_active_input(self, sparse_pattern):
        self.active_input.sparse = np.array(sparse_pattern, dtype='uint32') + self.input_range[0]

    def get_active_muscles(self):
        return self.active_muscles - self.muscles_range[0]

    def get_active_input(self):
        return self.active_input - self.input_range[0]

    def get_depolarized_muscles(self):
        return self.depolarized_muscles.sparse - self.muscles_range[0]

    def depolarize_muscles(self):
        # Active
        num_connected, num_potential = self.connections.computeActivityFull(self.active_input,
                                                                            False)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= self.activation_threshold)
        depolarized_cells = self.connections.mapSegmentsToCells(active_segments)  # with duplicates

        # Matching
        matching_segments = np.flatnonzero(num_potential >= self.learning_threshold)

        self.active_segments = active_segments
        self.matching_segments = matching_segments
        self.num_potential = num_potential
        self.depolarized_muscles.sparse = depolarized_cells

    def learn(self):
        learning_active_segments = self.connections.filterSegmentsByCell(self.active_segments,
                                                                         self.active_muscles.sparse)
        active_segments_to_punish = np.setdiff1d(self.active_segments, learning_active_segments)

        learning_matching_segments = self.connections.filterSegmentsByCell(self.matching_segments,
                                                                           self.active_muscles.sparse)
        matching_segments_to_punish = np.setdiff1d(self.active_segments, learning_matching_segments)

        cells_with_segments = np.union1d(self.connections.mapSegmentsToCells(learning_active_segments),
                                         self.connections.mapSegmentsToCells(learning_matching_segments))
        cells_to_grow_new_segments = np.setdiff1d(self.active_muscles.sparse, cells_with_segments)

        for segments in (learning_active_segments, learning_matching_segments):
            self._learn(self.connections, segments,
                        self.active_input.sparse,
                        self.active_input.sparse,
                        self.num_potential,
                        self.sample_size,
                        self.max_synapses_per_segment)

        if self.depolarized_decrement != 0.0:
            for segments in (active_segments_to_punish, matching_segments_to_punish):
                for segment in segments:
                    self.connections.adaptSegment(segment, self.active_input.sparse,
                                                  -self.depolarized_decrement, 0.0, False)

        if len(self.active_input.sparse) > 0:
            self._learn_on_new_segments(self.connections,
                                        cells_to_grow_new_segments,
                                        self.active_input.sparse,
                                        self.sample_size, self.max_synapses_per_segment)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment):
        """
        Learn on specified segments
        :param connections: exemplar of Connections class
        :param learning_segments: list of segments' id
        :param active_cells: list of active cells' id
        :param winner_cells: list of winner cells' id (cells to which connections will be grown)
        :param num_potential: list of counts of potential synapses for every segment
        :return:
        """
        for segment in learning_segments:
            connections.adaptSegment(segment, active_cells, self.permanence_increment, self.permanence_decrement, False)

            if sample_size == -1:
                max_new = len(winner_cells)
            else:
                max_new = sample_size - num_potential[segment]

            if max_synapses_per_segment != -1:
                synapse_counts = connections.numSynapses(segment)
                num_synapses_to_reach_max = max_synapses_per_segment - synapse_counts
                max_new = min(max_new, num_synapses_to_reach_max)
            if max_new > 0:
                connections.growSynapses(segment, winner_cells, self.initial_permanence, self.rng, max_new)

    def _learn_on_new_segments(self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
                               max_synapses_per_segment):
        """
        Grows new segments and learn on them
        :param connections:
        :param new_segment_cells: cells' id to grow new segments on
        :param growth_candidates: cells' id to grow synapses to
        :return:
        """
        num_new_synapses = len(growth_candidates)

        if sample_size != -1:
            num_new_synapses = min(num_new_synapses, sample_size)

        if max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, max_synapses_per_segment)

        for cell in new_segment_cells:
            new_segment = connections.createSegment(cell, self.max_segments_per_cell)
            connections.growSynapses(new_segment, growth_candidates, self.initial_permanence, self.rng,
                                     maxNew=num_new_synapses)
