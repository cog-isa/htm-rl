from htm_rl.agents.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.advanced.support.numpy_helpers import setCompare, argmaxMulti, getAllCellsInColumns
import numpy as np
from htm.bindings.math import Random
from math import exp
import copy
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.cc.utils import ExponentialDecayFunction, NoDecayFunction, LogisticExciteFunction, \
    FixedExciteFunction
from functools import reduce
from htm_rl.agents.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.advanced.support.numpy_helpers import setCompare, argmaxMulti, getAllCellsInColumns
import numpy as np
from htm.bindings.math import Random
from math import exp


EPS = 1e-12
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 0.000001


class GeneralFeedbackTM:
    def __init__(self,
                 columns,
                 cells_per_column,
                 context_cells,
                 feedback_cells,
                 activation_threshold_basal,
                 learning_threshold_basal,
                 activation_threshold_apical,
                 learning_threshold_apical,
                 connected_threshold_basal=0.5,
                 permanence_increment_basal=0.1,
                 permanence_decrement_basal=0.01,
                 initial_permanence_basal=0.4,
                 predicted_segment_decrement_basal=0.001,
                 sample_size_basal=-1,
                 max_synapses_per_segment_basal=-1,
                 max_segments_per_cell_basal=255,
                 connected_threshold_apical=0.5,
                 permanence_increment_apical=0.1,
                 permanence_decrement_apical=0.01,
                 initial_permanence_apical=0.4,
                 predicted_segment_decrement_apical=0.001,
                 sample_size_apical=-1,
                 max_synapses_per_segment_apical=-1,
                 max_segments_per_cell_apical=255,
                 prune_zero_synapses=True,
                 timeseries=False,
                 anomaly_window=1000,
                 confidence_window=1000,
                 noise_tolerance=0.0,
                 sm_ac=0,
                 seed=None,
                 ):

        self.columns = columns
        self.cells_per_column = cells_per_column
        self.local_cells = columns * cells_per_column
        self.context_cells = context_cells
        self.feedback_cells = feedback_cells
        self.activation_threshold_basal = activation_threshold_basal
        self.learning_threshold_basal = learning_threshold_basal
        self.activation_threshold_apical = activation_threshold_apical
        self.learning_threshold_apical = learning_threshold_apical
        self.connected_threshold_basal = connected_threshold_basal
        self.permanence_increment_basal = permanence_increment_basal
        self.permanence_decrement_basal = permanence_decrement_basal
        self.initial_permanence_basal = initial_permanence_basal
        self.predicted_segment_decrement_basal = predicted_segment_decrement_basal
        self.sample_size_basal = sample_size_basal
        self.max_synapses_per_segment_basal = max_synapses_per_segment_basal
        self.max_segments_per_cell_basal = max_segments_per_cell_basal
        self.connected_threshold_apical = connected_threshold_apical
        self.permanence_increment_apical = permanence_increment_apical
        self.permanence_decrement_apical = permanence_decrement_apical
        self.initial_permanence_apical = initial_permanence_apical
        self.predicted_segment_decrement_apical = predicted_segment_decrement_apical
        self.sample_size_apical = sample_size_apical
        self.max_synapses_per_segment_apical = max_synapses_per_segment_apical
        self.max_segments_per_cell_apical = max_segments_per_cell_apical
        self.timeseries = timeseries
        self.prune_zero_synapses = prune_zero_synapses
        self.sm_ac = sm_ac
        self.noise_tolerance = noise_tolerance

        self.total_cells = self.local_cells + self.context_cells + self.feedback_cells

        self.local_range = (0, self.local_cells)
        self.context_range = (self.local_range[1], self.local_range[1] + self.context_cells)
        self.feedback_range = (self.context_range[1], self.context_range[1] + self.feedback_cells)

        self.basal_connections = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold_basal,
                                             timeseries=self.timeseries)

        self.apical_connections = Connections(numCells=self.total_cells,
                                              connectedThreshold=self.connected_threshold_apical,
                                              timeseries=self.timeseries)

        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)
        self.correct_predicted_cells = SDR(self.total_cells)

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

        self.anomaly_window = anomaly_window
        self.confidence_window = confidence_window
        self.anomaly = [0.0 for _ in range(self.anomaly_window)]
        self.confidence = [0.0 for _ in range(self.confidence_window)]
        self.anomaly_threshold = 0
        self.confidence_threshold = 0
        self.mean_active_columns = 0

        if seed:
            self.rng = Random(seed)
        else:
            self.rng = Random()

    def reset(self):
        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

    # input
    def set_active_columns(self, columns_id):
        self.active_columns.sparse = np.array(columns_id)

    def set_active_context_cells(self, cells_id):
        self.active_cells_context.sparse = np.array(cells_id) + self.context_range[0]

    def set_active_feedback_cells(self, cells_id):
        self.active_cells_feedback.sparse = np.array(cells_id) + self.feedback_range[0]

    # output
    def get_active_columns(self):
        return np.copy(self.active_columns.sparse)

    def get_predicted_columns(self):
        return self.predicted_columns.sparse

    def get_active_cells(self):
        return self.active_cells.sparse - self.local_range[0]

    def get_winner_cells(self):
        return self.winner_cells.sparse - self.local_range[0]

    def get_correctly_predicted_cells(self):
        return self.correct_predicted_cells.sparse - self.local_range[0]

    # processing
    def activate_basal_dendrites(self, learn):
        self.active_segments_basal, self.matching_segments_basal, self.predictive_cells_basal, self.num_potential_basal = self._activate_dendrites(
            self.basal_connections, self.active_cells_context, self.activation_threshold_basal,
            self.learning_threshold_basal, learn
        )

    def activate_apical_dendrites(self, learn):
        self.active_segments_apical, self.matching_segments_apical, self.predictive_cells_apical, self.num_potential_apical = self._activate_dendrites(
            self.apical_connections, self.active_cells_feedback, self.activation_threshold_apical,
            self.learning_threshold_apical, learn
        )

    def predict_cells(self):
        """
        Calculates predicted cells. Should be called after dendrite activations.
        :return:
        """
        # basal and apical coincidence predict first
        predicted_cells = np.intersect1d(self.predictive_cells_basal, self.predictive_cells_apical)
        # if there is no coincidence, predict all possible cases
        if predicted_cells.size == 0:
            predicted_cells = self.predictive_cells_basal

        self.predicted_cells.sparse = predicted_cells.astype('uint32')
        self.predicted_columns.sparse = np.unique(self._columns_for_cells(self.predicted_cells.sparse))

        confidence = min(len(self.predicted_cells.sparse) / (self.mean_active_columns + EPS), 1.0)
        self.confidence_threshold = self.confidence_threshold + (
                confidence - self.confidence[0]) / self.confidence_window
        self.confidence.append(confidence)
        self.confidence.pop(0)

    def activate_cells(self, learn: bool):
        """
        Calculates new active cells and performs connections' learning.
        :param learn: if true, connections will learn patterns from previous step
        :return:
        """
        # Calculate active cells
        correct_predicted_cells, bursting_columns = setCompare(self.predicted_cells.sparse, self.active_columns.sparse,
                                                               aKey=self._columns_for_cells(
                                                                   self.predicted_cells.sparse),
                                                               rightMinusLeft=True)
        self.correct_predicted_cells.sparse = correct_predicted_cells
        new_active_cells = np.concatenate((correct_predicted_cells,
                                           getAllCellsInColumns(bursting_columns,
                                                                self.cells_per_column) + self.local_range[0]))

        (learning_active_basal_segments,
         learning_matching_basal_segments,
         learning_matching_apical_segments,
         cells_to_grow_apical_segments,
         basal_segments_to_punish,
         apical_segments_to_punish,
         cells_to_grow_apical_and_basal_segments,
         new_winner_cells) = self._calculate_learning(bursting_columns, correct_predicted_cells)

        # Learn
        if learn:
            # Learn on existing segments
            if self.active_cells_context.sparse.size > 0:
                for learning_segments in (learning_active_basal_segments, learning_matching_basal_segments):
                    self._learn(self.basal_connections, learning_segments, self.active_cells_context,
                                self.active_cells_context.sparse,
                                self.num_potential_basal, self.sample_size_basal, self.max_synapses_per_segment_basal,
                                self.initial_permanence_basal, self.permanence_increment_basal,
                                self.permanence_decrement_basal,
                                self.learning_threshold_basal)
            if self.active_cells_feedback.sparse.size > 0:
                self._learn(self.apical_connections, learning_matching_apical_segments, self.active_cells_feedback,
                            self.active_cells_feedback.sparse,
                            self.num_potential_apical, self.sample_size_apical, self.max_synapses_per_segment_apical,
                            self.initial_permanence_apical, self.permanence_increment_apical,
                            self.permanence_decrement_apical,
                            self.learning_threshold_apical)

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                if self.active_cells_context.sparse.size > 0:
                    for segment in basal_segments_to_punish:
                        self.basal_connections.adaptSegment(segment, self.active_cells_context,
                                                            -self.predicted_segment_decrement_basal, 0.0,
                                                            self.prune_zero_synapses, self.learning_threshold_basal)
                if self.active_cells_feedback.sparse.size > 0:
                    for segment in apical_segments_to_punish:
                        self.apical_connections.adaptSegment(segment, self.active_cells_feedback,
                                                             -self.predicted_segment_decrement_apical, 0.0,
                                                             self.prune_zero_synapses, self.learning_threshold_apical)

            # Grow new segments
            if self.active_cells_context.sparse.size > 0:
                self._learn_on_new_segments(self.basal_connections,
                                            cells_to_grow_apical_and_basal_segments,
                                            self.active_cells_context.sparse,
                                            self.sample_size_basal, self.max_synapses_per_segment_basal,
                                            self.initial_permanence_basal,
                                            self.max_segments_per_cell_basal)
            if self.active_cells_feedback.sparse.size > 0:
                self._learn_on_new_segments(self.apical_connections,
                                            np.concatenate((cells_to_grow_apical_segments,
                                                            cells_to_grow_apical_and_basal_segments)),
                                            self.active_cells_feedback.sparse,
                                            self.sample_size_apical, self.max_synapses_per_segment_apical,
                                            self.initial_permanence_apical,
                                            self.max_segments_per_cell_apical)

        self.active_cells.sparse = np.unique(new_active_cells.astype('uint32'))
        self.winner_cells.sparse = np.unique(new_winner_cells)

        n_active_columns = self.active_columns.sparse.size
        self.mean_active_columns = self.sm_ac * self.mean_active_columns + (
                1 - self.sm_ac) * n_active_columns
        if n_active_columns != 0:
            anomaly = len(bursting_columns) / n_active_columns
        else:
            anomaly = 1.0

        self.anomaly_threshold = self.anomaly_threshold + (anomaly - self.anomaly[0]) / self.anomaly_window
        self.anomaly.append(anomaly)
        self.anomaly.pop(0)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
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
            connections.adaptSegment(segment, active_cells, permanence_increment, permanence_decrement,
                                     self.prune_zero_synapses, segmentThreshold)

            if sample_size == -1:
                max_new = len(winner_cells)
            else:
                max_new = sample_size - num_potential[segment]

            if max_synapses_per_segment != -1:
                synapse_counts = connections.numSynapses(segment)
                num_synapses_to_reach_max = max_synapses_per_segment - synapse_counts
                max_new = min(max_new, num_synapses_to_reach_max)
            if max_new > 0:
                connections.growSynapses(segment, winner_cells, initial_permanence, self.rng, max_new)

    def _learn_on_new_segments(self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
                               max_synapses_per_segment,
                               initial_permanence, max_segments_per_cell):
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
            new_segment = connections.createSegment(cell, max_segments_per_cell)
            connections.growSynapses(new_segment, growth_candidates, initial_permanence, self.rng,
                                     maxNew=num_new_synapses)

    def _calculate_learning(self, bursting_columns, correct_predicted_cells):
        """
        Calculates which segments to train and where to grow new segments
        :param bursting_columns: numpy array of columns' id
        :param correct_predicted_cells: numpy array of cells' id
        :return:
        """
        # Correctly predicted columns
        # choose active segments for correctly predicted cells
        learning_active_basal_segments = self.basal_connections.filterSegmentsByCell(self.active_segments_basal,
                                                                                     correct_predicted_cells)
        # choose all matching apical segments for correctly predicted segments
        # if there is no matching segment, we should grow an apical segment on this cell
        learning_matching_apical_segments, cells_to_grow_apical_segments = setCompare(self.matching_segments_apical,
                                                                                      correct_predicted_cells,
                                                                                      aKey=self.apical_connections.mapSegmentsToCells(
                                                                                          self.matching_segments_apical),
                                                                                      rightMinusLeft=True)
        # narrow apical segments to the best one per correctly predicted cell
        learning_matching_apical_segments = self._choose_best_segment_per_cell(self.apical_connections,
                                                                               correct_predicted_cells,
                                                                               learning_matching_apical_segments,
                                                                               self.num_potential_apical)
        # all cells with matching segments
        cells_for_matching_basal = self.basal_connections.mapSegmentsToCells(self.matching_segments_basal)
        cells_for_matching_apical = self.apical_connections.mapSegmentsToCells(self.matching_segments_apical)
        matching_cells = np.unique(cells_for_matching_basal)

        matching_cells_in_bursting_columns, bursting_columns_with_no_match = setCompare(matching_cells,
                                                                                        bursting_columns,
                                                                                        aKey=self._columns_for_cells(
                                                                                            matching_cells),
                                                                                        rightMinusLeft=True)
        # choose the best segment per cell
        if matching_cells_in_bursting_columns.size > 0:
            (learning_matching_basal_segments,
             learning_matching_apical_segments2,
             cells_to_grow_apical_segments2
             ) = self._choose_best_segment_per_column(
                matching_cells_in_bursting_columns)
        else:
            learning_matching_basal_segments = np.empty(0, dtype=np.int32)
            learning_matching_apical_segments2 = np.empty(0, dtype=np.int32)
            cells_to_grow_apical_segments2 = np.empty(0, dtype=np.int32)
        # cells on which new apical and basal segments will be grown
        if bursting_columns_with_no_match.size > 0:
            cells_to_grow_apical_and_basal_segments = self._get_cells_with_fewest_segments(self.basal_connections,
                                                                                           self.apical_connections,
                                                                                           bursting_columns_with_no_match)
        else:
            cells_to_grow_apical_and_basal_segments = np.empty(0, dtype=UINT_DTYPE)

        # compile all segments and cells together
        cells_to_grow_apical_segments = np.concatenate([cells_to_grow_apical_segments, cells_to_grow_apical_segments2])

        learning_matching_apical_segments = np.concatenate(
            [learning_matching_apical_segments, learning_matching_apical_segments2])

        winner_cells = np.concatenate(
            (correct_predicted_cells,
             self.basal_connections.mapSegmentsToCells(learning_matching_basal_segments),
             cells_to_grow_apical_and_basal_segments)
        )

        # Incorrectly predicted columns
        incorrect_matching_basal_mask = np.isin(self._columns_for_cells(cells_for_matching_basal),
                                                self.active_columns.sparse, invert=True)
        incorrect_matching_apical_mask = np.isin(self._columns_for_cells(cells_for_matching_apical),
                                                 self.active_columns.sparse, invert=True)

        basal_segments_to_punish = self.matching_segments_basal[incorrect_matching_basal_mask]
        apical_segments_to_punish = self.matching_segments_apical[incorrect_matching_apical_mask]

        return (learning_active_basal_segments.astype('uint32'),
                learning_matching_basal_segments.astype('uint32'),
                learning_matching_apical_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'),
                basal_segments_to_punish.astype('uint32'),
                apical_segments_to_punish.astype('uint32'),
                cells_to_grow_apical_and_basal_segments.astype('uint32'),
                winner_cells.astype('uint32'))

    def _choose_best_segment_per_column(self, cells):
        """
        Chooses best matching segment per column among the cells, using apical tie breaking.
        :param cells: numpy array of cells' id
        :return:
        """
        candidate_basal_segments = self.basal_connections.filterSegmentsByCell(self.matching_segments_basal, cells)
        candidate_apical_segments = self._choose_best_segment_per_cell(self.apical_connections, cells,
                                                                       self.matching_segments_apical,
                                                                       self.num_potential_apical)
        intersection_mask = np.in1d(self.basal_connections.mapSegmentsToCells(candidate_basal_segments),
                                    self.apical_connections.mapSegmentsToCells(candidate_apical_segments))
        candidate_basal_with_apical_neighbour = candidate_basal_segments[intersection_mask]

        # for segment, that have no adjacent apical segment the score is zero, else score is sigmoid(best_apical_segment) - 0.5
        cells_for_apical_segments = self.apical_connections.mapSegmentsToCells(candidate_apical_segments)
        cells_for_basal_segments = self.basal_connections.mapSegmentsToCells(candidate_basal_with_apical_neighbour)
        tiebreaker = np.zeros_like(candidate_basal_segments)
        # WARNING, lazy realization of tiebreaking! May be slow!
        # TODO make optimized tiebreaking
        tiebreaker[intersection_mask] = np.array(
            [exp(self.num_potential_apical[candidate_apical_segments[cells_for_apical_segments == x]].sum()) for x
             in cells_for_basal_segments]
        )
        #
        one_per_column_filter = argmaxMulti(
            self.num_potential_basal[candidate_basal_segments] + tiebreaker / (tiebreaker + 1) - 0.5,
            groupKeys=self._columns_for_cells(
                self.basal_connections.mapSegmentsToCells(candidate_basal_segments)))
        learning_basal_segments = candidate_basal_segments[one_per_column_filter]
        cells_for_learning_basal_segments = self.basal_connections.mapSegmentsToCells(learning_basal_segments)
        learning_apical_segments = candidate_apical_segments[np.in1d(cells_for_apical_segments,
                                                                     cells_for_learning_basal_segments)]
        # if there is no matching apical segment on learning_basal_segment: grow one
        cells_to_grow_apical_segments = cells_for_learning_basal_segments[np.in1d(cells_for_learning_basal_segments,
                                                                                  cells_for_apical_segments,
                                                                                  invert=True)]

        return (learning_basal_segments.astype('uint32'),
                learning_apical_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'))

    @staticmethod
    def _choose_best_segment_per_cell(connections, cells, segments, num_potential):
        """
        Calculates best matching segment per cell.
        :param connections:
        :param cells: numpy array of cells' id
        :param segments: numpy array of segments' id
        :param num_potential:
        :return:
        """
        candidate_segments = connections.filterSegmentsByCell(segments, cells)

        # Narrow it down to one pair per cell.
        if candidate_segments.size > 0:
            one_per_cell_filter = argmaxMulti(num_potential[candidate_segments],
                                              groupKeys=connections.mapSegmentsToCells(candidate_segments))
            learning_segments = candidate_segments[one_per_cell_filter]
        else:
            learning_segments = np.empty(0)
        return learning_segments.astype('uint32')

    def _get_cells_with_fewest_segments(self, basal_connections, apical_connections, columns):
        """
        Calculates cells with fewest segments per column.
        :param basal_connections:
        :param apical_connections:
        :param columns:
        :return:
        """
        candidate_cells = getAllCellsInColumns(columns, self.cells_per_column) + self.local_range[0]

        # Arrange the segment counts into one row per minicolumn.
        # count apical and basal segments per cell
        segment_counts = np.reshape(
            basal_connections.getSegmentCounts(candidate_cells) + apical_connections.getSegmentCounts(candidate_cells),
            newshape=(len(columns), self.cells_per_column))

        # Filter to just the cells that are tied for fewest in their minicolumn.
        tiebreaker = np.empty_like(segment_counts, dtype='float64')
        self.rng.initializeReal64Array(tiebreaker)
        segment_counts = segment_counts + tiebreaker * 0.1

        min_segment_counts = np.amin(segment_counts, axis=1, keepdims=True)
        candidate_cells = candidate_cells[np.flatnonzero(segment_counts == min_segment_counts)]

        return candidate_cells.astype('uint32')

    @staticmethod
    def _activate_dendrites(connections, presynaptic_cells, activation_threshold, learning_threshold, learn):
        """
        Calculates active and matching segments and predictive cells.
        :param connections:
        :param presynaptic_cells:
        :param activation_threshold:
        :param learning_threshold:
        :return:
        """
        # Active
        num_connected, num_potential = connections.computeActivityFull(presynaptic_cells,
                                                                       learn)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= activation_threshold)
        predictive_cells = np.unique(connections.mapSegmentsToCells(active_segments))

        # Matching
        matching_segments = np.flatnonzero(num_potential >= learning_threshold)

        return active_segments, matching_segments, predictive_cells, num_potential

    def _columns_for_cells(self, cells):
        """
        Calculates column numbers for basal cells
        :param cells: numpy array of cells id
        :return: numpy array of columns id for every cell
        """
        if np.any(cells < self.local_range[0]) or np.any(cells >= self.local_range[1]):
            raise ValueError('cells are not in bounds')

        local_cells = cells - self.local_range[0]
        columns = local_cells // self.cells_per_column
        return columns.astype('int32')

    def _filter_by_columns(self, cells, columns, invert=False):
        """
        Filters cells by specified columns
        :param cells: numpy array of cells id
        :param columns: numpy array of columns id
        :param invert: if true then return cells that not in columns
        :return: numpy array of cells id
        """
        columns_for_cells = self._columns_for_cells(cells)
        return cells[np.in1d(columns_for_cells, columns, invert=invert)]


class DelayedFeedbackTM:
    """
    Updates apical connections only when propagate method is called.
    Stores all cells to grow apical connections in union sparse array.
    """
    def __init__(self,
                 columns,
                 cells_per_column,
                 context_cells,
                 feedback_cells,
                 activation_threshold_basal,
                 learning_threshold_basal,
                 activation_threshold_apical,
                 learning_threshold_apical,
                 connected_threshold_basal=0.5,
                 permanence_increment_basal=0.1,
                 permanence_decrement_basal=0.01,
                 initial_permanence_basal=0.4,
                 predicted_segment_decrement_basal=0.001,
                 sample_size_basal=-1,
                 max_synapses_per_segment_basal=-1,
                 max_segments_per_cell_basal=255,
                 connected_threshold_apical=0.5,
                 permanence_increment_apical=0.1,
                 permanence_decrement_apical=0.01,
                 initial_permanence_apical=0.4,
                 predicted_segment_decrement_apical=0.001,
                 sample_size_apical=-1,
                 max_synapses_per_segment_apical=-1,
                 max_segments_per_cell_apical=255,
                 prune_zero_synapses=True,
                 timeseries=False,
                 anomaly_window=1000,
                 confidence_window=1000,
                 noise_tolerance=0.0,
                 sm_ac=0,
                 seed=None,
                 ):

        self.columns = columns
        self.cells_per_column = cells_per_column
        self.local_cells = columns * cells_per_column
        self.context_cells = context_cells
        self.feedback_cells = feedback_cells
        self.activation_threshold_basal = activation_threshold_basal
        self.learning_threshold_basal = learning_threshold_basal
        self.activation_threshold_apical = activation_threshold_apical
        self.learning_threshold_apical = learning_threshold_apical
        self.connected_threshold_basal = connected_threshold_basal
        self.permanence_increment_basal = permanence_increment_basal
        self.permanence_decrement_basal = permanence_decrement_basal
        self.initial_permanence_basal = initial_permanence_basal
        self.predicted_segment_decrement_basal = predicted_segment_decrement_basal
        self.sample_size_basal = sample_size_basal
        self.max_synapses_per_segment_basal = max_synapses_per_segment_basal
        self.max_segments_per_cell_basal = max_segments_per_cell_basal
        self.connected_threshold_apical = connected_threshold_apical
        self.permanence_increment_apical = permanence_increment_apical
        self.permanence_decrement_apical = permanence_decrement_apical
        self.initial_permanence_apical = initial_permanence_apical
        self.predicted_segment_decrement_apical = predicted_segment_decrement_apical
        self.sample_size_apical = sample_size_apical
        self.max_synapses_per_segment_apical = max_synapses_per_segment_apical
        self.max_segments_per_cell_apical = max_segments_per_cell_apical
        self.timeseries = timeseries
        self.prune_zero_synapses = prune_zero_synapses
        self.sm_ac = sm_ac
        self.noise_tolerance = noise_tolerance

        self.total_cells = self.local_cells + self.context_cells + self.feedback_cells

        self.local_range = (0, self.local_cells)
        self.context_range = (self.local_range[1], self.local_range[1] + self.context_cells)
        self.feedback_range = (self.context_range[1], self.context_range[1] + self.feedback_cells)

        self.basal_connections = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold_basal,
                                             timeseries=self.timeseries)

        self.apical_connections = Connections(numCells=self.total_cells,
                                              connectedThreshold=self.connected_threshold_apical,
                                              timeseries=self.timeseries)

        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)
        self.correct_predicted_cells = SDR(self.total_cells)
        self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

        self.anomaly_window = anomaly_window
        self.confidence_window = confidence_window
        self.anomaly = [0.0 for _ in range(self.anomaly_window)]
        self.confidence = [0.0 for _ in range(self.confidence_window)]
        self.anomaly_threshold = 0
        self.confidence_threshold = 0
        self.mean_active_columns = 0

        if seed:
            self.rng = Random(seed)
        else:
            self.rng = Random()

    def reset(self):
        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)
        self.correct_predicted_cells = SDR(self.total_cells)
        self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

    # input
    def set_active_columns(self, columns_id):
        self.active_columns.sparse = np.array(columns_id)

    def set_active_context_cells(self, cells_id):
        self.active_cells_context.sparse = np.array(cells_id) + self.context_range[0]

    def set_active_feedback_cells(self, cells_id):
        self.active_cells_feedback.sparse = np.array(cells_id) + self.feedback_range[0]

    # output
    def get_active_columns(self):
        return np.copy(self.active_columns.sparse)

    def get_predicted_columns(self):
        return self.predicted_columns.sparse

    def get_active_cells(self):
        return self.active_cells.sparse - self.local_range[0]

    def get_winner_cells(self):
        return self.winner_cells.sparse - self.local_range[0]

    def get_correctly_predicted_cells(self):
        return self.correct_predicted_cells.sparse - self.local_range[0]

    # processing
    def activate_basal_dendrites(self, learn):
        self.active_segments_basal, self.matching_segments_basal, self.predictive_cells_basal, self.num_potential_basal = self._activate_dendrites(
            self.basal_connections, self.active_cells_context, self.activation_threshold_basal,
            self.learning_threshold_basal, learn
        )

    def activate_apical_dendrites(self, learn):
        self.active_segments_apical, self.matching_segments_apical, self.predictive_cells_apical, self.num_potential_apical = self._activate_dendrites(
            self.apical_connections, self.active_cells_feedback, self.activation_threshold_apical,
            self.learning_threshold_apical, learn
        )

    def predict_cells(self):
        """
        Calculates predicted cells. Should be called after dendrite activations.
        :return:
        """
        # basal and apical coincidence predict first
        predicted_cells = np.intersect1d(self.predictive_cells_basal, self.predictive_cells_apical)
        # if there is no coincidence, predict all possible cases
        if predicted_cells.size == 0:
            predicted_cells = self.predictive_cells_basal

        self.predicted_cells.sparse = predicted_cells.astype('uint32')
        self.predicted_columns.sparse = np.unique(self._columns_for_cells(self.predicted_cells.sparse))

        confidence = min(len(self.predicted_cells.sparse) / (self.mean_active_columns + EPS), 1.0)
        self.confidence_threshold = self.confidence_threshold + (
                confidence - self.confidence[0]) / self.confidence_window
        self.confidence.append(confidence)
        self.confidence.pop(0)

    def activate_cells(self, learn: bool):
        """
        Calculates new active cells and performs connections' learning.
        cells for growing apical segments will be accumulated
        :param learn: if true, connections will learn patterns from previous step
        :return:
        """
        # Calculate active cells
        correct_predicted_cells, bursting_columns = setCompare(self.predicted_cells.sparse, self.active_columns.sparse,
                                                               aKey=self._columns_for_cells(
                                                                   self.predicted_cells.sparse),
                                                               rightMinusLeft=True)
        self.correct_predicted_cells.sparse = correct_predicted_cells
        new_active_cells = np.concatenate((correct_predicted_cells,
                                           getAllCellsInColumns(bursting_columns,
                                                                self.cells_per_column) + self.local_range[0]))

        (learning_active_basal_segments,
         learning_matching_basal_segments,
         basal_segments_to_punish,
         cells_to_grow_apical_and_basal_segments,
         cells_to_grow_apical_segments,
         new_winner_cells) = self._calculate_basal_learning(bursting_columns, correct_predicted_cells)
        # accumulate all cells to grow apical segments
        self.cells_to_grow_apical_segments = reduce(np.union1d, [self.cells_to_grow_apical_segments,
                                                                 cells_to_grow_apical_segments,
                                                                 cells_to_grow_apical_and_basal_segments,
                                                                 correct_predicted_cells])
        # Learn
        if learn:
            # Learn on existing segments
            if self.active_cells_context.sparse.size > 0:
                for learning_segments in (learning_active_basal_segments, learning_matching_basal_segments):
                    self._learn(self.basal_connections, learning_segments, self.active_cells_context,
                                self.active_cells_context.sparse,
                                self.num_potential_basal, self.sample_size_basal, self.max_synapses_per_segment_basal,
                                self.initial_permanence_basal, self.permanence_increment_basal,
                                self.permanence_decrement_basal,
                                self.learning_threshold_basal)

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                if self.active_cells_context.sparse.size > 0:
                    for segment in basal_segments_to_punish:
                        self.basal_connections.adaptSegment(segment, self.active_cells_context,
                                                            -self.predicted_segment_decrement_basal, 0.0,
                                                            self.prune_zero_synapses, self.learning_threshold_basal)

            # Grow new segments
            if self.active_cells_context.sparse.size > 0:
                self._learn_on_new_segments(self.basal_connections,
                                            cells_to_grow_apical_and_basal_segments,
                                            self.active_cells_context.sparse,
                                            self.sample_size_basal, self.max_synapses_per_segment_basal,
                                            self.initial_permanence_basal,
                                            self.max_segments_per_cell_basal)

        self.active_cells.sparse = np.unique(new_active_cells.astype('uint32'))
        self.winner_cells.sparse = np.unique(new_winner_cells)

        n_active_columns = self.active_columns.sparse.size
        self.mean_active_columns = self.sm_ac * self.mean_active_columns + (
                1 - self.sm_ac) * n_active_columns
        if n_active_columns != 0:
            anomaly = len(bursting_columns) / n_active_columns
        else:
            anomaly = 1.0

        self.anomaly_threshold = self.anomaly_threshold + (anomaly - self.anomaly[0]) / self.anomaly_window
        self.anomaly.append(anomaly)
        self.anomaly.pop(0)

    def propagate_feedback(self):
        # calculate apical learning
        if self.active_cells_feedback.size > 0:
            (learning_matching_apical_segments,
             cells_to_grow_apical_segments,
             apical_segments_to_punish) = self._calculate_apical_learning(self.cells_to_grow_apical_segments)
            # Learn
            # Learn on existing segments
            self._learn(self.apical_connections, learning_matching_apical_segments, self.active_cells_feedback,
                        self.active_cells_feedback.sparse,
                        self.num_potential_apical, self.sample_size_apical, self.max_synapses_per_segment_apical,
                        self.initial_permanence_apical, self.permanence_increment_apical,
                        self.permanence_decrement_apical,
                        self.learning_threshold_apical)

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                for segment in apical_segments_to_punish:
                    self.apical_connections.adaptSegment(segment, self.active_cells_feedback,
                                                         -self.predicted_segment_decrement_apical, 0.0,
                                                         self.prune_zero_synapses, self.learning_threshold_apical)

            # Grow new segments
            self._learn_on_new_segments(self.apical_connections,
                                        cells_to_grow_apical_segments,
                                        self.active_cells_feedback.sparse,
                                        self.sample_size_apical, self.max_synapses_per_segment_apical,
                                        self.initial_permanence_apical,
                                        self.max_segments_per_cell_apical)

            self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
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
            connections.adaptSegment(segment, active_cells, permanence_increment, permanence_decrement,
                                     self.prune_zero_synapses, segmentThreshold)

            if sample_size == -1:
                max_new = len(winner_cells)
            else:
                max_new = sample_size - num_potential[segment]

            if max_synapses_per_segment != -1:
                synapse_counts = connections.numSynapses(segment)
                num_synapses_to_reach_max = max_synapses_per_segment - synapse_counts
                max_new = min(max_new, num_synapses_to_reach_max)
            if max_new > 0:
                connections.growSynapses(segment, winner_cells, initial_permanence, self.rng, max_new)

    def _learn_on_new_segments(self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
                               max_synapses_per_segment,
                               initial_permanence, max_segments_per_cell):
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
            new_segment = connections.createSegment(cell, max_segments_per_cell)
            connections.growSynapses(new_segment, growth_candidates, initial_permanence, self.rng,
                                     maxNew=num_new_synapses)

    def _calculate_basal_learning(self, bursting_columns, correct_predicted_cells):
        """
        Calculates which segments to train and where to grow new segments
        :param bursting_columns: numpy array of columns' id
        :param correct_predicted_cells: numpy array of cells' id
        :return:
        """
        # Correctly predicted columns
        # choose active segments for correctly predicted cells
        learning_active_basal_segments = self.basal_connections.filterSegmentsByCell(self.active_segments_basal,
                                                                                     correct_predicted_cells)

        # all cells with matching segments
        cells_for_matching_basal = self.basal_connections.mapSegmentsToCells(self.matching_segments_basal)
        matching_cells = np.unique(cells_for_matching_basal)

        matching_cells_in_bursting_columns, bursting_columns_with_no_match = setCompare(matching_cells,
                                                                                        bursting_columns,
                                                                                        aKey=self._columns_for_cells(
                                                                                            matching_cells),
                                                                                        rightMinusLeft=True)
        # choose the best segment per cell
        if matching_cells_in_bursting_columns.size > 0:
            (learning_matching_basal_segments,
             cells_to_grow_apical_segments
             ) = self._choose_best_segment_per_column(
                matching_cells_in_bursting_columns)
        else:
            learning_matching_basal_segments = np.empty(0, dtype=np.int32)
            cells_to_grow_apical_segments = np.empty(0, dtype=np.int32)
        # cells on which new apical and basal segments will be grown
        if bursting_columns_with_no_match.size > 0:
            cells_to_grow_apical_and_basal_segments = self._get_cells_with_fewest_segments(self.basal_connections,
                                                                                           self.apical_connections,
                                                                                           bursting_columns_with_no_match)
        else:
            cells_to_grow_apical_and_basal_segments = np.empty(0, dtype=UINT_DTYPE)

        winner_cells = np.concatenate(
            (correct_predicted_cells,
             self.basal_connections.mapSegmentsToCells(learning_matching_basal_segments),
             cells_to_grow_apical_and_basal_segments)
        )

        # Incorrectly predicted columns
        incorrect_matching_basal_mask = np.isin(self._columns_for_cells(cells_for_matching_basal),
                                                self.active_columns.sparse, invert=True)
        basal_segments_to_punish = self.matching_segments_basal[incorrect_matching_basal_mask]

        return (learning_active_basal_segments.astype('uint32'),
                learning_matching_basal_segments.astype('uint32'),
                basal_segments_to_punish.astype('uint32'),
                cells_to_grow_apical_and_basal_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'),
                winner_cells.astype('uint32'))

    def _calculate_apical_learning(self, candidate_cells):
        learning_matching_apical_segments, apical_segments_to_punish, cells_to_grow_apical_segments = setCompare(
            self.matching_segments_apical,
            candidate_cells,
            aKey=self.apical_connections.mapSegmentsToCells(
                self.matching_segments_apical),
            leftMinusRight=True,
            rightMinusLeft=True
        )
        learning_matching_apical_segments = self._choose_best_segment_per_cell(self.apical_connections,
                                                                               candidate_cells,
                                                                               learning_matching_apical_segments,
                                                                               self.num_potential_apical)
        return learning_matching_apical_segments, cells_to_grow_apical_segments, apical_segments_to_punish

    def _choose_best_segment_per_column(self, cells):
        """
        Chooses best matching segment per column among the cells, using apical tie breaking.
        :param cells: numpy array of cells' id
        :return:
        """
        candidate_basal_segments = self.basal_connections.filterSegmentsByCell(self.matching_segments_basal, cells)
        tiebreaker = np.empty_like(candidate_basal_segments)
        self.rng.initializeReal64Array(tiebreaker)
        one_per_column_filter = argmaxMulti(
            self.num_potential_basal[candidate_basal_segments] + tiebreaker * 0.1,
            groupKeys=self._columns_for_cells(
                self.basal_connections.mapSegmentsToCells(candidate_basal_segments)))
        learning_basal_segments = candidate_basal_segments[one_per_column_filter]
        cells_for_learning_basal_segments = self.basal_connections.mapSegmentsToCells(learning_basal_segments)

        return learning_basal_segments.astype(UINT_DTYPE), cells_for_learning_basal_segments.astype(UINT_DTYPE)

    @staticmethod
    def _choose_best_segment_per_cell(connections, cells, segments, num_potential):
        """
        Calculates best matching segment per cell.
        :param connections:
        :param cells: numpy array of cells' id
        :param segments: numpy array of segments' id
        :param num_potential:
        :return:
        """
        candidate_segments = connections.filterSegmentsByCell(segments, cells)

        # Narrow it down to one pair per cell.
        if candidate_segments.size > 0:
            one_per_cell_filter = argmaxMulti(num_potential[candidate_segments],
                                              groupKeys=connections.mapSegmentsToCells(candidate_segments))
            learning_segments = candidate_segments[one_per_cell_filter]
        else:
            learning_segments = np.empty(0)
        return learning_segments.astype('uint32')

    def _get_cells_with_fewest_segments(self, basal_connections, apical_connections, columns):
        """
        Calculates cells with fewest segments per column.
        :param basal_connections:
        :param apical_connections:
        :param columns:
        :return:
        """
        candidate_cells = getAllCellsInColumns(columns, self.cells_per_column) + self.local_range[0]

        # Arrange the segment counts into one row per minicolumn.
        # count apical and basal segments per cell
        segment_counts = np.reshape(
            basal_connections.getSegmentCounts(candidate_cells) + apical_connections.getSegmentCounts(candidate_cells),
            newshape=(len(columns), self.cells_per_column))

        # Filter to just the cells that are tied for fewest in their minicolumn.
        tiebreaker = np.empty_like(segment_counts, dtype='float64')
        self.rng.initializeReal64Array(tiebreaker)
        segment_counts = segment_counts + tiebreaker * 0.1

        min_segment_counts = np.amin(segment_counts, axis=1, keepdims=True)
        candidate_cells = candidate_cells[np.flatnonzero(segment_counts == min_segment_counts)]

        return candidate_cells.astype('uint32')

    @staticmethod
    def _activate_dendrites(connections, presynaptic_cells, activation_threshold, learning_threshold, learn):
        """
        Calculates active and matching segments and predictive cells.
        :param connections:
        :param presynaptic_cells:
        :param activation_threshold:
        :param learning_threshold:
        :return:
        """
        # Active
        num_connected, num_potential = connections.computeActivityFull(presynaptic_cells,
                                                                       learn)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= activation_threshold)
        predictive_cells = connections.mapSegmentsToCells(active_segments)  # with duplicates

        # Matching
        matching_segments = np.flatnonzero(num_potential >= learning_threshold)

        return active_segments, matching_segments, predictive_cells, num_potential

    def _columns_for_cells(self, cells):
        """
        Calculates column numbers for basal cells
        :param cells: numpy array of cells id
        :return: numpy array of columns id for every cell
        """
        if np.any(cells < self.local_range[0]) or np.any(cells >= self.local_range[1]):
            raise ValueError('cells are not in bounds')

        local_cells = cells - self.local_range[0]
        columns = local_cells // self.cells_per_column
        return columns.astype('int32')

    def _filter_by_columns(self, cells, columns, invert=False):
        """
        Filters cells by specified columns
        :param cells: numpy array of cells id
        :param columns: numpy array of columns id
        :param invert: if true then return cells that not in columns
        :return: numpy array of cells id
        """
        columns_for_cells = self._columns_for_cells(cells)
        return cells[np.in1d(columns_for_cells, columns, invert=invert)]


class MovingDelayedFeedbackTM:
    """
    Updates apical connections only when propagate method is called.
    Stores all cells to grow apical segments in a list for every step.
    When it comes to grow apical connections, it unites all patterns in history.
    """
    def __init__(self,
                 columns,
                 cells_per_column,
                 context_cells,
                 feedback_cells,
                 activation_threshold_basal,
                 learning_threshold_basal,
                 activation_threshold_apical,
                 learning_threshold_apical,
                 max_steps_history=1,
                 connected_threshold_basal=0.5,
                 permanence_increment_basal=0.1,
                 permanence_decrement_basal=0.01,
                 initial_permanence_basal=0.4,
                 predicted_segment_decrement_basal=0.001,
                 sample_size_basal=-1,
                 max_synapses_per_segment_basal=-1,
                 max_segments_per_cell_basal=255,
                 connected_threshold_apical=0.5,
                 permanence_increment_apical=0.1,
                 permanence_decrement_apical=0.01,
                 initial_permanence_apical=0.4,
                 predicted_segment_decrement_apical=0.001,
                 sample_size_apical=-1,
                 max_synapses_per_segment_apical=-1,
                 max_segments_per_cell_apical=255,
                 prune_zero_synapses=True,
                 timeseries=False,
                 anomaly_window=1000,
                 confidence_window=1000,
                 noise_tolerance=0.0,
                 sm_ac=0,
                 seed=None,
                 ):

        self.columns = columns
        self.cells_per_column = cells_per_column
        self.local_cells = columns * cells_per_column
        self.context_cells = context_cells
        self.feedback_cells = feedback_cells
        self.max_steps_history = max_steps_history
        self.activation_threshold_basal = activation_threshold_basal
        self.learning_threshold_basal = learning_threshold_basal
        self.activation_threshold_apical = activation_threshold_apical
        self.learning_threshold_apical = learning_threshold_apical
        self.connected_threshold_basal = connected_threshold_basal
        self.permanence_increment_basal = permanence_increment_basal
        self.permanence_decrement_basal = permanence_decrement_basal
        self.initial_permanence_basal = initial_permanence_basal
        self.predicted_segment_decrement_basal = predicted_segment_decrement_basal
        self.sample_size_basal = sample_size_basal
        self.max_synapses_per_segment_basal = max_synapses_per_segment_basal
        self.max_segments_per_cell_basal = max_segments_per_cell_basal
        self.connected_threshold_apical = connected_threshold_apical
        self.permanence_increment_apical = permanence_increment_apical
        self.permanence_decrement_apical = permanence_decrement_apical
        self.initial_permanence_apical = initial_permanence_apical
        self.predicted_segment_decrement_apical = predicted_segment_decrement_apical
        self.sample_size_apical = sample_size_apical
        self.max_synapses_per_segment_apical = max_synapses_per_segment_apical
        self.max_segments_per_cell_apical = max_segments_per_cell_apical
        self.timeseries = timeseries
        self.prune_zero_synapses = prune_zero_synapses
        self.sm_ac = sm_ac
        self.noise_tolerance = noise_tolerance

        self.total_cells = self.local_cells + self.context_cells + self.feedback_cells

        self.local_range = (0, self.local_cells)
        self.context_range = (self.local_range[1], self.local_range[1] + self.context_cells)
        self.feedback_range = (self.context_range[1], self.context_range[1] + self.feedback_cells)

        self.basal_connections = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold_basal,
                                             timeseries=self.timeseries)

        self.apical_connections = Connections(numCells=self.total_cells,
                                              connectedThreshold=self.connected_threshold_apical,
                                              timeseries=self.timeseries)

        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)
        self.correct_predicted_cells = SDR(self.total_cells)
        self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)
        self.cells_to_grow_apical_segments_history = list()

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

        self.anomaly_window = anomaly_window
        self.confidence_window = confidence_window
        self.anomaly = [0.0 for _ in range(self.anomaly_window)]
        self.confidence = [0.0 for _ in range(self.confidence_window)]
        self.anomaly_threshold = 0
        self.confidence_threshold = 0
        self.mean_active_columns = 0

        if seed:
            self.rng = Random(seed)
        else:
            self.rng = Random()

    def reset(self):
        self.active_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.predicted_cells = SDR(self.total_cells)
        self.active_columns = SDR(self.columns)
        self.predicted_columns = SDR(self.columns)
        self.correct_predicted_cells = SDR(self.total_cells)
        self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)
        self.cells_to_grow_apical_segments_history = list()

        self.active_cells_context = SDR(self.total_cells)
        self.active_cells_feedback = SDR(self.total_cells)

        self.predictive_cells_basal = np.empty(0)
        self.active_segments_basal = np.empty(0)
        self.matching_segments_basal = np.empty(0)
        self.num_potential_basal = np.empty(0)

        self.predictive_cells_apical = np.empty(0)
        self.active_segments_apical = np.empty(0)
        self.matching_segments_apical = np.empty(0)
        self.num_potential_apical = np.empty(0)

    # input
    def set_active_columns(self, columns_id):
        self.active_columns.sparse = np.array(columns_id)

    def set_active_context_cells(self, cells_id):
        self.active_cells_context.sparse = np.array(cells_id) + self.context_range[0]

    def set_active_feedback_cells(self, cells_id):
        self.active_cells_feedback.sparse = np.array(cells_id) + self.feedback_range[0]

    # output
    def get_active_columns(self):
        return np.copy(self.active_columns.sparse)

    def get_predicted_columns(self):
        return self.predicted_columns.sparse

    def get_active_cells(self):
        return self.active_cells.sparse - self.local_range[0]

    def get_winner_cells(self):
        return self.winner_cells.sparse - self.local_range[0]

    def get_correctly_predicted_cells(self):
        return self.correct_predicted_cells.sparse - self.local_range[0]

    # processing
    def activate_basal_dendrites(self, learn):
        self.active_segments_basal, self.matching_segments_basal, self.predictive_cells_basal, self.num_potential_basal = self._activate_dendrites(
            self.basal_connections, self.active_cells_context, self.activation_threshold_basal,
            self.learning_threshold_basal, learn
        )

    def activate_apical_dendrites(self, learn):
        self.active_segments_apical, self.matching_segments_apical, self.predictive_cells_apical, self.num_potential_apical = self._activate_dendrites(
            self.apical_connections, self.active_cells_feedback, self.activation_threshold_apical,
            self.learning_threshold_apical, learn
        )

    def predict_cells(self):
        """
        Calculates predicted cells. Should be called after dendrite activations.
        :return:
        """
        # basal and apical coincidence predict first
        if self.predictive_cells_apical.size > 0:
            predicted_cells = np.intersect1d(self.predictive_cells_basal, self.predictive_cells_apical)
        else:
            predicted_cells = self.predictive_cells_basal

        self.predicted_cells.sparse = predicted_cells.astype('uint32')
        self.predicted_columns.sparse = np.unique(self._columns_for_cells(self.predicted_cells.sparse))

        confidence = min(len(self.predicted_cells.sparse) / (self.mean_active_columns + EPS), 1.0)
        self.confidence_threshold = self.confidence_threshold + (
                confidence - self.confidence[0]) / self.confidence_window
        self.confidence.append(confidence)
        self.confidence.pop(0)

    def activate_cells(self, learn: bool):
        """
        Calculates new active cells and performs connections' learning.
        cells for growing apical segments will be accumulated
        :param learn: if true, connections will learn patterns from previous step
        :return:
        """
        # Calculate active cells
        correct_predicted_cells, bursting_columns = setCompare(self.predicted_cells.sparse, self.active_columns.sparse,
                                                               aKey=self._columns_for_cells(
                                                                   self.predicted_cells.sparse),
                                                               rightMinusLeft=True)
        self.correct_predicted_cells.sparse = correct_predicted_cells
        new_active_cells = np.concatenate((correct_predicted_cells,
                                           getAllCellsInColumns(bursting_columns,
                                                                self.cells_per_column) + self.local_range[0]))

        (learning_active_basal_segments,
         learning_matching_basal_segments,
         basal_segments_to_punish,
         cells_to_grow_apical_and_basal_segments,
         cells_to_grow_apical_segments,
         new_winner_cells) = self._calculate_basal_learning(bursting_columns, correct_predicted_cells)
        # accumulate all cells to grow apical segments
        all_cells_to_grow_apical_segments = reduce(np.union1d, [cells_to_grow_apical_segments,
                                                                cells_to_grow_apical_and_basal_segments,
                                                                correct_predicted_cells])

        self.cells_to_grow_apical_segments_history.append(all_cells_to_grow_apical_segments.copy())
        if len(self.cells_to_grow_apical_segments_history) > self.max_steps_history:
            self.cells_to_grow_apical_segments_history.pop(0)
        # Learn
        if learn:
            # Learn on existing segments
            if self.active_cells_context.sparse.size > 0:
                for learning_segments in (learning_active_basal_segments, learning_matching_basal_segments):
                    self._learn(self.basal_connections, learning_segments, self.active_cells_context,
                                self.active_cells_context.sparse,
                                self.num_potential_basal, self.sample_size_basal, self.max_synapses_per_segment_basal,
                                self.initial_permanence_basal, self.permanence_increment_basal,
                                self.permanence_decrement_basal,
                                self.learning_threshold_basal)

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                if self.active_cells_context.sparse.size > 0:
                    for segment in basal_segments_to_punish:
                        self.basal_connections.adaptSegment(segment, self.active_cells_context,
                                                            -self.predicted_segment_decrement_basal, 0.0,
                                                            self.prune_zero_synapses, self.learning_threshold_basal)

            # Grow new segments
            if self.active_cells_context.sparse.size > 0:
                self._learn_on_new_segments(self.basal_connections,
                                            cells_to_grow_apical_and_basal_segments,
                                            self.active_cells_context.sparse,
                                            self.sample_size_basal, self.max_synapses_per_segment_basal,
                                            self.initial_permanence_basal,
                                            self.max_segments_per_cell_basal)

        self.active_cells.sparse = np.unique(new_active_cells.astype('uint32'))
        self.winner_cells.sparse = np.unique(new_winner_cells)

        n_active_columns = self.active_columns.sparse.size
        self.mean_active_columns = self.sm_ac * self.mean_active_columns + (
                1 - self.sm_ac) * n_active_columns
        if n_active_columns != 0:
            anomaly = len(bursting_columns) / n_active_columns
        else:
            anomaly = 1.0

        self.anomaly_threshold = self.anomaly_threshold + (anomaly - self.anomaly[0]) / self.anomaly_window
        self.anomaly.append(anomaly)
        self.anomaly.pop(0)

    def propagate_feedback(self):
        # calculate apical learning
        if (self.active_cells_feedback.size > 0) and (len(self.cells_to_grow_apical_segments_history) > 0):
            self.cells_to_grow_apical_segments = reduce(np.union1d, self.cells_to_grow_apical_segments_history)
            (learning_matching_apical_segments,
             cells_to_grow_apical_segments,
             apical_segments_to_punish) = self._calculate_apical_learning(self.cells_to_grow_apical_segments)
            # Learn
            # Learn on existing segments
            self._learn(self.apical_connections, learning_matching_apical_segments, self.active_cells_feedback,
                        self.active_cells_feedback.sparse,
                        self.num_potential_apical, self.sample_size_apical, self.max_synapses_per_segment_apical,
                        self.initial_permanence_apical, self.permanence_increment_apical,
                        self.permanence_decrement_apical,
                        self.learning_threshold_apical)

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                for segment in apical_segments_to_punish:
                    self.apical_connections.adaptSegment(segment, self.active_cells_feedback,
                                                         -self.predicted_segment_decrement_apical, 0.0,
                                                         self.prune_zero_synapses, self.learning_threshold_apical)

            # Grow new segments
            self._learn_on_new_segments(self.apical_connections,
                                        cells_to_grow_apical_segments,
                                        self.active_cells_feedback.sparse,
                                        self.sample_size_apical, self.max_synapses_per_segment_apical,
                                        self.initial_permanence_apical,
                                        self.max_segments_per_cell_apical)

            self.cells_to_grow_apical_segments = np.empty(0, dtype=UINT_DTYPE)
            self.predictive_cells_apical = np.empty(0)
            self.active_segments_apical = np.empty(0)
            self.matching_segments_apical = np.empty(0)
            self.num_potential_apical = np.empty(0)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
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
            connections.adaptSegment(segment, active_cells, permanence_increment, permanence_decrement,
                                     self.prune_zero_synapses, segmentThreshold)

            if sample_size == -1:
                max_new = len(winner_cells)
            else:
                max_new = sample_size - num_potential[segment]

            if max_synapses_per_segment != -1:
                synapse_counts = connections.numSynapses(segment)
                num_synapses_to_reach_max = max_synapses_per_segment - synapse_counts
                max_new = min(max_new, num_synapses_to_reach_max)
            if max_new > 0:
                connections.growSynapses(segment, winner_cells, initial_permanence, self.rng, max_new)

    def _learn_on_new_segments(self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
                               max_synapses_per_segment,
                               initial_permanence, max_segments_per_cell):
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
            new_segment = connections.createSegment(cell, max_segments_per_cell)
            connections.growSynapses(new_segment, growth_candidates, initial_permanence, self.rng,
                                     maxNew=num_new_synapses)

    def _calculate_basal_learning(self, bursting_columns, correct_predicted_cells):
        """
        Calculates which segments to train and where to grow new segments
        :param bursting_columns: numpy array of columns' id
        :param correct_predicted_cells: numpy array of cells' id
        :return:
        """
        # Correctly predicted columns
        # choose active segments for correctly predicted cells
        learning_active_basal_segments = self.basal_connections.filterSegmentsByCell(self.active_segments_basal,
                                                                                     correct_predicted_cells)

        # all cells with matching segments
        cells_for_matching_basal = self.basal_connections.mapSegmentsToCells(self.matching_segments_basal)
        matching_cells = np.unique(cells_for_matching_basal)

        matching_cells_in_bursting_columns, bursting_columns_with_no_match = setCompare(matching_cells,
                                                                                        bursting_columns,
                                                                                        aKey=self._columns_for_cells(
                                                                                            matching_cells),
                                                                                        rightMinusLeft=True)
        # choose the best segment per cell
        if matching_cells_in_bursting_columns.size > 0:
            (learning_matching_basal_segments,
             cells_to_grow_apical_segments
             ) = self._choose_best_segment_per_column(
                matching_cells_in_bursting_columns)
        else:
            learning_matching_basal_segments = np.empty(0, dtype=np.int32)
            cells_to_grow_apical_segments = np.empty(0, dtype=np.int32)
        # cells on which new apical and basal segments will be grown
        if bursting_columns_with_no_match.size > 0:
            cells_to_grow_apical_and_basal_segments = self._get_cells_with_fewest_segments(self.basal_connections,
                                                                                           self.apical_connections,
                                                                                           bursting_columns_with_no_match)
        else:
            cells_to_grow_apical_and_basal_segments = np.empty(0, dtype=UINT_DTYPE)

        winner_cells = np.concatenate(
            (correct_predicted_cells,
             self.basal_connections.mapSegmentsToCells(learning_matching_basal_segments),
             cells_to_grow_apical_and_basal_segments)
        )

        # Incorrectly predicted columns
        incorrect_matching_basal_mask = np.isin(self._columns_for_cells(cells_for_matching_basal),
                                                self.active_columns.sparse, invert=True)
        basal_segments_to_punish = self.matching_segments_basal[incorrect_matching_basal_mask]

        return (learning_active_basal_segments.astype('uint32'),
                learning_matching_basal_segments.astype('uint32'),
                basal_segments_to_punish.astype('uint32'),
                cells_to_grow_apical_and_basal_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'),
                winner_cells.astype('uint32'))

    def _calculate_apical_learning(self, candidate_cells):
        learning_matching_apical_segments, apical_segments_to_punish, cells_to_grow_apical_segments = setCompare(
            self.matching_segments_apical,
            candidate_cells,
            aKey=self.apical_connections.mapSegmentsToCells(
                self.matching_segments_apical),
            leftMinusRight=True,
            rightMinusLeft=True
        )
        learning_matching_apical_segments = self._choose_best_segment_per_cell(self.apical_connections,
                                                                               candidate_cells,
                                                                               learning_matching_apical_segments,
                                                                               self.num_potential_apical)
        return learning_matching_apical_segments, cells_to_grow_apical_segments, apical_segments_to_punish

    def _choose_best_segment_per_column(self, cells):
        """
        Chooses best matching segment per column among the cells, using apical tie breaking.
        :param cells: numpy array of cells' id
        :return:
        """
        candidate_basal_segments = self.basal_connections.filterSegmentsByCell(self.matching_segments_basal, cells)
        tiebreaker = np.empty_like(candidate_basal_segments)
        self.rng.initializeReal64Array(tiebreaker)
        one_per_column_filter = argmaxMulti(
            self.num_potential_basal[candidate_basal_segments] + tiebreaker * 0.1,
            groupKeys=self._columns_for_cells(
                self.basal_connections.mapSegmentsToCells(candidate_basal_segments)))
        learning_basal_segments = candidate_basal_segments[one_per_column_filter]
        cells_for_learning_basal_segments = self.basal_connections.mapSegmentsToCells(learning_basal_segments)

        return learning_basal_segments.astype(UINT_DTYPE), cells_for_learning_basal_segments.astype(UINT_DTYPE)

    @staticmethod
    def _choose_best_segment_per_cell(connections, cells, segments, num_potential):
        """
        Calculates best matching segment per cell.
        :param connections:
        :param cells: numpy array of cells' id
        :param segments: numpy array of segments' id
        :param num_potential:
        :return:
        """
        candidate_segments = connections.filterSegmentsByCell(segments, cells)

        # Narrow it down to one pair per cell.
        if candidate_segments.size > 0:
            one_per_cell_filter = argmaxMulti(num_potential[candidate_segments],
                                              groupKeys=connections.mapSegmentsToCells(candidate_segments))
            learning_segments = candidate_segments[one_per_cell_filter]
        else:
            learning_segments = np.empty(0)
        return learning_segments.astype('uint32')

    def _get_cells_with_fewest_segments(self, basal_connections, apical_connections, columns):
        """
        Calculates cells with fewest segments per column.
        :param basal_connections:
        :param apical_connections:
        :param columns:
        :return:
        """
        candidate_cells = getAllCellsInColumns(columns, self.cells_per_column) + self.local_range[0]

        # Arrange the segment counts into one row per minicolumn.
        # count apical and basal segments per cell
        segment_counts = np.reshape(
            basal_connections.getSegmentCounts(candidate_cells) + apical_connections.getSegmentCounts(candidate_cells),
            newshape=(len(columns), self.cells_per_column))

        # Filter to just the cells that are tied for fewest in their minicolumn.
        tiebreaker = np.empty_like(segment_counts, dtype='float64')
        self.rng.initializeReal64Array(tiebreaker)
        segment_counts = segment_counts + tiebreaker * 0.1

        min_segment_counts = np.amin(segment_counts, axis=1, keepdims=True)
        candidate_cells = candidate_cells[np.flatnonzero(segment_counts == min_segment_counts)]

        return candidate_cells.astype('uint32')

    @staticmethod
    def _activate_dendrites(connections, presynaptic_cells, activation_threshold, learning_threshold, learn):
        """
        Calculates active and matching segments and predictive cells.
        :param connections:
        :param presynaptic_cells:
        :param activation_threshold:
        :param learning_threshold:
        :return:
        """
        # Active
        num_connected, num_potential = connections.computeActivityFull(presynaptic_cells,
                                                                       learn)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= activation_threshold)
        predictive_cells = connections.mapSegmentsToCells(active_segments)  # with duplicates

        # Matching
        matching_segments = np.flatnonzero(num_potential >= learning_threshold)

        return active_segments, matching_segments, predictive_cells, num_potential

    def _columns_for_cells(self, cells):
        """
        Calculates column numbers for basal cells
        :param cells: numpy array of cells id
        :return: numpy array of columns id for every cell
        """
        if np.any(cells < self.local_range[0]) or np.any(cells >= self.local_range[1]):
            raise ValueError('cells are not in bounds')

        local_cells = cells - self.local_range[0]
        columns = local_cells // self.cells_per_column
        return columns.astype('int32')

    def _filter_by_columns(self, cells, columns, invert=False):
        """
        Filters cells by specified columns
        :param cells: numpy array of cells id
        :param columns: numpy array of columns id
        :param invert: if true then return cells that not in columns
        :return: numpy array of cells id
        """
        columns_for_cells = self._columns_for_cells(cells)
        return cells[np.in1d(columns_for_cells, columns, invert=invert)]


class ApicalBasalFeedbackTM:
    def __init__(self,
                 apical_columns, apical_cells_per_column,
                 basal_columns, basal_cells_per_column,
                 feedback_columns,
                 activation_threshold,
                 learning_threshold,
                 activation_apical_threshold,
                 learning_apical_threshold,
                 activation_inhib_basal_threshold,
                 learning_inhib_basal_threshold,
                 activation_inhib_feedback_threshold,
                 learning_inhib_feedback_threshold,
                 learning_exec_threshold,
                 activation_exec_threshold,
                 connected_threshold=0.5,
                 seed=None,
                 permanence_increment=0.1,
                 permanence_decrement=0.01,
                 initial_permanence=0.4,
                 predicted_segment_decrement=0.001,
                 sample_size=-1,
                 sample_apical_size=-1,
                 sample_inhib_basal_size=-1,
                 sample_inhib_feedback_size=-1,
                 sample_exec_size=-1,
                 max_synapses_per_segment=-1,
                 max_apical_synapses_per_segment=-1,
                 max_inhib_synapses_per_segment=-1,
                 max_exec_synapses_per_segment=-1,
                 max_segments_per_cell=255,
                 anomaly_window=1000,
                 confidence_window=1000,
                 noise_tolerance=0.0,
                 connected_threshold_apical=0.5,
                 permanence_increment_apical=0.1,
                 permanence_decrement_apical=0.01,
                 predicted_segment_decrement_apical=0.001,
                 initial_permanence_apical=0.4,
                 predicted_segment_decrement_exec=0.001,
                 initial_permanence_exec=0.4,
                 permanence_decrement_exec=0.01,
                 permanence_increment_exec=0.1,
                 connected_threshold_exec=0.5,
                 predicted_segment_decrement_inhib=0.001,
                 initial_permanence_inhib=0.4,
                 permanence_decrement_inhib=0.01,
                 permanence_increment_inhib=0.1,
                 connected_threshold_inhib=0.5,
                 enable_pruning_basal=True,
                 enable_pruning_apical=True,
                 enable_pruning_exec=True,
                 enable_pruning_inhib=True,
                 pruning_period_basal=1000,
                 pruning_period_apical=1000,
                 pruning_period_exec=1000,
                 pruning_period_inhib=1000,
                 prune_zero_synapses=True,
                 max_segments_per_cell_apical=255,
                 max_segments_per_cell_exec=255,
                 max_segments_per_cell_inhib=255,
                 timeseries=False):
        self.step = 0

        self.apical_columns = apical_columns
        self.apical_cells_per_column = apical_cells_per_column
        self.apical_total_cells = apical_columns * apical_cells_per_column

        self.basal_columns = basal_columns
        self.basal_cells_per_column = basal_cells_per_column
        self.basal_total_cells = basal_columns * basal_cells_per_column

        self.feedback_columns = feedback_columns

        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold

        self.activation_inhib_basal_threshold = activation_inhib_basal_threshold
        self.learning_inhib_basal_threshold = learning_inhib_basal_threshold
        self.activation_inhib_feedback_threshold = activation_inhib_feedback_threshold
        self.learning_inhib_feedback_threshold = learning_inhib_feedback_threshold

        self.activation_apical_threshold = activation_apical_threshold
        self.learning_apical_threshold = learning_apical_threshold

        self.activation_exec_threshold = activation_exec_threshold
        self.learning_exec_threshold = learning_exec_threshold

        self.connected_threshold = connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_increment_init = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.permanence_decrement_init = permanence_decrement
        self.initial_permanence = initial_permanence
        self.predicted_segment_decrement = predicted_segment_decrement
        self.predicted_segment_decrement_init = predicted_segment_decrement

        self.connected_threshold_apical = connected_threshold_apical
        self.permanence_increment_apical = permanence_increment_apical
        self.permanence_increment_apical_init = permanence_increment_apical
        self.permanence_decrement_apical = permanence_decrement_apical
        self.permanence_decrement_apical_init = permanence_decrement_apical
        self.initial_permanence_apical = initial_permanence_apical
        self.predicted_segment_decrement_apical = predicted_segment_decrement_apical
        self.predicted_segment_decrement_apical_init = predicted_segment_decrement_apical

        self.connected_threshold_inhib = connected_threshold_inhib
        self.permanence_increment_inhib = permanence_increment_inhib
        self.permanence_increment_inhib_init = permanence_increment_inhib
        self.permanence_decrement_inhib = permanence_decrement_inhib
        self.permanence_decrement_inhib_init = permanence_decrement_inhib
        self.initial_permanence_inhib = initial_permanence_inhib
        self.predicted_segment_decrement_inhib = predicted_segment_decrement_inhib
        self.predicted_segment_decrement_inhib_init = predicted_segment_decrement_inhib

        self.connected_threshold_exec = connected_threshold_exec
        self.permanence_increment_exec = permanence_increment_exec
        self.permanence_increment_exec_init = permanence_increment_exec
        self.permanence_decrement_exec = permanence_decrement_exec
        self.permanence_decrement_exec_init = permanence_decrement_exec
        self.initial_permanence_exec = initial_permanence_exec
        self.predicted_segment_decrement_exec = predicted_segment_decrement_exec
        self.predicted_segment_decrement_exec_init = predicted_segment_decrement_exec

        self.sample_size = sample_size
        self.sample_inhib_basal_size = sample_inhib_basal_size
        self.sample_inhib_feedback_size = sample_inhib_feedback_size
        self.sample_exec_size = sample_exec_size
        self.sample_apical_size = sample_apical_size

        self.max_synapses_per_segment = max_synapses_per_segment
        self.max_inhib_synapses_per_segment = max_inhib_synapses_per_segment
        self.max_exec_synapses_per_segment = max_exec_synapses_per_segment
        self.max_apical_synapses_per_segment = max_apical_synapses_per_segment
        self.max_segments_per_cell = max_segments_per_cell
        self.max_segments_per_cell_apical = max_segments_per_cell_apical
        self.max_segments_per_cell_exec = max_segments_per_cell_exec
        self.max_segments_per_cell_inhib = max_segments_per_cell_inhib

        self.total_cells = (self.apical_total_cells +
                            self.basal_total_cells +
                            self.feedback_columns)

        self.apical_range = (0, self.apical_total_cells)
        self.basal_range = (self.apical_range[1], self.apical_range[1] + self.basal_total_cells)
        self.feedback_range = (self.basal_range[1], self.basal_range[1] + self.feedback_columns)

        self.timeseries = timeseries

        self.apical_connections = Connections(numCells=self.total_cells,
                                              connectedThreshold=self.connected_threshold_apical,
                                              timeseries=self.timeseries)
        self.basal_connections = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold,
                                             timeseries=self.timeseries)
        self.exec_feedback_connections = Connections(numCells=self.total_cells,
                                                     connectedThreshold=self.connected_threshold_exec,
                                                     timeseries=self.timeseries)
        self.inhib_connections = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold_inhib,
                                             timeseries=self.timeseries)

        self.active_basal_cells = SDR(self.total_cells)
        self.winner_basal_cells = SDR(self.total_cells)
        self.active_basal_segments = np.empty(0)
        self.matching_basal_segments = np.empty(0)
        self.basal_predictive_cells = np.empty(0)
        self.num_basal_potential = np.empty(0)

        self.active_apical_cells = SDR(self.total_cells)
        self.winner_apical_cells = SDR(self.total_cells)
        self.active_apical_segments = np.empty(0)
        self.matching_apical_segments = np.empty(0)
        self.apical_predictive_cells = np.empty(0)
        self.num_apical_potential = np.empty(0)

        self.active_inhib_basal_segments = np.empty(0)
        self.matching_inhib_basal_segments = np.empty(0)
        self.inhibited_basal_cells = np.empty(0)
        self.num_inhib_basal_potential = np.empty(0)

        self.active_inhib_feedback_segments = np.empty(0)
        self.matching_inhib_feedback_segments = np.empty(0)
        self.inhibited_feedback_cells = np.empty(0)
        self.num_inhib_feedback_potential = np.empty(0)

        self.inhib_presynaptic_cells = SDR(self.total_cells)
        self.inhib_receptive_field = SDR(self.total_cells)
        self.active_inhib_segments = np.empty(0)
        self.matching_inhib_segments = np.empty(0)
        self.inhibited_cells = np.empty(0)

        self.active_exec_segments = np.empty(0)
        self.matching_exec_segments = np.empty(0)
        self.exec_predictive_cells = np.empty(0)
        self.num_exec_potential = np.empty(0)

        self.active_columns = SDR(self.basal_columns)
        self.predicted_cells = SDR(self.total_cells)
        self.predicted_columns = SDR(self.basal_columns)
        self.active_feedback_columns = SDR(self.total_cells)

        self.anomaly_window = anomaly_window
        self.confidence_window = confidence_window
        self.anomaly = [0 for _ in range(self.anomaly_window)]
        self.confidence = [0 for _ in range(self.confidence_window)]
        self.anomaly_threshold = 0
        self.confidence_threshold = 0

        self.noise_tolerance = noise_tolerance

        self.enable_pruning_basal = enable_pruning_basal
        self.enable_pruning_apical = enable_pruning_apical
        self.enable_pruning_exec = enable_pruning_exec
        self.enable_pruning_inhib = enable_pruning_inhib

        self.pruning_period_basal = pruning_period_basal
        self.pruning_period_apical = pruning_period_apical
        self.pruning_period_exec = pruning_period_exec
        self.pruning_period_inhib = pruning_period_inhib

        self.segments_activity_basal = np.empty(0)
        self.segments_activity_apical = np.empty(0)
        self.segments_activity_exec = np.empty(0)
        self.segments_activity_inhib = np.empty(0)

        self.prune_zero_synapses = prune_zero_synapses

        if seed:
            self.rng = Random(seed)
        else:
            self.rng = Random()

    def reset(self):
        self.active_basal_cells = SDR(self.total_cells)
        self.winner_basal_cells = SDR(self.total_cells)
        self.active_basal_segments = np.empty(0)
        self.matching_basal_segments = np.empty(0)
        self.basal_predictive_cells = np.empty(0)
        self.num_basal_potential = np.empty(0)

        self.active_apical_cells = SDR(self.total_cells)
        self.winner_apical_cells = SDR(self.total_cells)
        self.active_apical_segments = np.empty(0)
        self.matching_apical_segments = np.empty(0)
        self.apical_predictive_cells = np.empty(0)
        self.num_apical_potential = np.empty(0)

        self.active_inhib_basal_segments = np.empty(0)
        self.matching_inhib_basal_segments = np.empty(0)
        self.inhibited_basal_cells = np.empty(0)
        self.num_inhib_basal_potential = np.empty(0)

        self.active_inhib_feedback_segments = np.empty(0)
        self.matching_inhib_feedback_segments = np.empty(0)
        self.inhibited_feedback_cells = np.empty(0)
        self.num_inhib_feedback_potential = np.empty(0)

        self.inhib_presynaptic_cells = SDR(self.total_cells)
        self.inhib_receptive_field = SDR(self.total_cells)
        self.active_inhib_segments = np.empty(0)
        self.matching_inhib_segments = np.empty(0)
        self.inhibited_cells = np.empty(0)

        self.active_exec_segments = np.empty(0)
        self.matching_exec_segments = np.empty(0)
        self.exec_predictive_cells = np.empty(0)
        self.num_exec_potential = np.empty(0)

        self.active_columns = SDR(self.basal_columns)
        self.predicted_cells = SDR(self.total_cells)
        self.predicted_columns = SDR(self.basal_columns)
        self.active_feedback_columns = SDR(self.total_cells)

    # input
    def set_active_columns(self, columns_id):
        self.active_columns.sparse = np.array(columns_id)

    def set_active_feedback_columns(self, columns_id):
        self.active_feedback_columns.sparse = np.array(columns_id) + self.feedback_range[0]

    def set_active_apical_cells(self, cells_id):
        self.active_apical_cells.sparse = np.array(cells_id) + self.apical_range[0]

    def set_winner_apical_cells(self, cells_id):
        self.winner_apical_cells.sparse = np.array(cells_id) + self.apical_range[0]

    def set_learning_rate(self, learning_rate_factor):
        self.permanence_increment = self.permanence_increment_init * learning_rate_factor
        self.permanence_decrement = self.permanence_decrement_init * learning_rate_factor
        self.permanence_increment_apical = self.permanence_increment_apical_init * learning_rate_factor
        self.permanence_increment_exec = self.permanence_increment_exec_init * learning_rate_factor
        self.permanence_increment_inhib = self.permanence_increment_inhib_init * learning_rate_factor
        self.predicted_segment_decrement = self.predicted_segment_decrement_init * learning_rate_factor
        self.predicted_segment_decrement_apical = self.predicted_segment_decrement_apical_init * learning_rate_factor
        self.predicted_segment_decrement_exec = self.predicted_segment_decrement_exec_init * learning_rate_factor
        self.predicted_segment_decrement_inhib = self.predicted_segment_decrement_inhib_init * learning_rate_factor

    # output
    def get_active_columns(self):
        return np.copy(self.active_columns.sparse)

    def get_predicted_columns(self, add_exec=False, add_apical=False):
        predictions = self.predicted_columns.sparse
        if add_exec:
            predictions = np.union1d(predictions, self._basal_columns_for_cells(self.exec_predictive_cells))
            self.inactivate_exec_dendrites()
        if add_apical:
            predictions = np.union1d(predictions, self._basal_columns_for_cells(self.apical_predictive_cells))
        return predictions

    def get_active_cells(self):
        return self.active_basal_cells.sparse - self.basal_range[0]

    def get_winner_cells(self):
        return self.winner_basal_cells.sparse - self.basal_range[0]

    def inactivate_exec_dendrites(self):
        self.active_exec_segments = np.empty(0)
        self.matching_exec_segments = np.empty(0)
        self.exec_predictive_cells = np.empty(0)
        self.num_exec_potential = np.empty(0)

    # processing
    def activate_basal_dendrites(self):
        self.active_basal_segments, self.matching_basal_segments, self.basal_predictive_cells, self.num_basal_potential = self._activate_dendrites(
            self.basal_connections, self.active_basal_cells, self.activation_threshold, self.learning_threshold
        )

    def activate_apical_dendrites(self):
        self.active_apical_segments, self.matching_apical_segments, self.apical_predictive_cells, self.num_apical_potential = self._activate_dendrites(
            self.apical_connections, self.active_apical_cells, self.activation_apical_threshold,
            self.learning_apical_threshold
        )

    def activate_inhib_dendrites(self):
        self.active_inhib_basal_segments, self.matching_inhib_basal_segments, self.inhibited_basal_cells, self.num_inhib_basal_potential = self._activate_dendrites(
            self.inhib_connections, self.active_basal_cells, self.activation_inhib_basal_threshold,
            self.learning_inhib_basal_threshold
        )

        self.active_inhib_feedback_segments, self.matching_inhib_feedback_segments, self.inhibited_feedback_cells, self.num_inhib_feedback_potential = self._activate_dendrites(
            self.inhib_connections, self.active_feedback_columns, self.activation_inhib_feedback_threshold,
            self.learning_inhib_feedback_threshold
        )

        self.active_inhib_segments = np.intersect1d(self.active_inhib_basal_segments,
                                                    self.active_inhib_feedback_segments)
        self.matching_inhib_segments = np.intersect1d(self.matching_inhib_basal_segments,
                                                      self.matching_inhib_feedback_segments)
        self.inhibited_cells = np.intersect1d(self.inhibited_basal_cells, self.inhibited_feedback_cells)

    def activate_exec_dendrites(self):
        self.active_exec_segments, self.matching_exec_segments, self.exec_predictive_cells, self.num_exec_potential = self._activate_dendrites(
            self.exec_feedback_connections, self.active_feedback_columns, self.activation_threshold,
            self.learning_threshold
        )

    def predict_cells(self):
        """
        Calculates predicted cells. Should be called after dendrite activations.
        :return: nothing
        """
        # exclude inhibited cells
        if self.inhibited_cells.size > 0:
            candidate_basal_predictive_cells = self.inhibited_cells
        else:
            candidate_basal_predictive_cells = self.basal_predictive_cells
        predicted_cells = list()
        # basal and apical coincidence predict first
        predicted_cells.append(np.intersect1d(candidate_basal_predictive_cells, self.apical_predictive_cells))
        # filter basal cells by already predicted columns and predict them
        predicted_columns = list()
        predicted_columns.append(np.unique(self._basal_columns_for_cells(predicted_cells[0])))
        predicted_cells.append(
            self._filter_by_basal_columns(candidate_basal_predictive_cells, predicted_columns[0], invert=True))

        self.predicted_cells.sparse = np.unique(np.concatenate(predicted_cells)).astype('uint32')
        self.predicted_columns.sparse = np.unique(self._basal_columns_for_cells(self.predicted_cells.sparse))

        confidence = min(len(self.predicted_cells.sparse) / self.activation_threshold, 1.0)
        self.confidence_threshold = self.confidence_threshold + (
                confidence - self.confidence[0]) / self.confidence_window
        self.confidence.append(confidence)
        self.confidence.pop(0)

    def learn_exec_feedback_segments(self):
        """
        Process one step of feedback excitatory connections' learning.
        :return:
        """
        mask1 = np.in1d(self.exec_feedback_connections.mapSegmentsToCells(self.matching_exec_segments),
                        self.winner_basal_cells.sparse)
        self._learn(self.exec_feedback_connections,
                    self.matching_exec_segments[mask1],
                    self.active_feedback_columns,
                    self.active_feedback_columns.sparse,
                    self.num_exec_potential,
                    self.sample_exec_size,
                    self.max_exec_synapses_per_segment,
                    self.initial_permanence_exec,
                    self.permanence_increment_exec,
                    self.permanence_decrement_exec,
                    self.learning_exec_threshold)
        # punish segments
        for segment in self.matching_exec_segments[~mask1]:
            self.exec_feedback_connections.adaptSegment(segment, self.active_feedback_columns,
                                                        -self.predicted_segment_decrement_exec, 0.0,
                                                        self.prune_zero_synapses)
        # grow new segments
        mask2 = np.in1d(self.winner_basal_cells.sparse,
                        self.exec_feedback_connections.mapSegmentsToCells(self.matching_exec_segments),
                        invert=True)
        cells_to_grow_exec_segments = self.winner_basal_cells.sparse[mask2]
        self._learn_on_new_segments(self.exec_feedback_connections,
                                    cells_to_grow_exec_segments,
                                    self.active_feedback_columns.sparse,
                                    self.sample_exec_size,
                                    self.max_exec_synapses_per_segment,
                                    self.initial_permanence_exec,
                                    self.max_segments_per_cell_exec
                                    )

    def activate_cells(self, learn: bool):
        """
        Calculates new active cells and performs connections' learning.
        :param learn: if true, connections will learn patterns from previous step
        :return:
        """
        # Calculate active cells
        correct_predicted_cells, bursting_columns = setCompare(self.predicted_cells.sparse, self.active_columns.sparse,
                                                               aKey=self._basal_columns_for_cells(
                                                                   self.predicted_cells.sparse),
                                                               rightMinusLeft=True)
        new_active_cells = np.concatenate((correct_predicted_cells,
                                           getAllCellsInColumns(bursting_columns,
                                                                self.basal_cells_per_column) + self.basal_range[0]))

        (learning_active_basal_segments,
         learning_matching_basal_segments,
         cells_to_grow_basal_segments,
         learning_matching_apical_segments,
         cells_to_grow_apical_segments,
         learning_active_inhibit_segments,
         learning_matching_inhibit_segments,
         cells_to_grow_inhibit_segments,
         basal_segments_to_punish,
         apical_segments_to_punish,
         inhibit_segments_to_punish,
         cells_to_grow_apical_and_basal_segments,
         new_winner_cells) = self._calculate_learning(bursting_columns, correct_predicted_cells)

        # Learn
        if learn:
            # Learn on existing segments
            for learning_segments in (learning_active_basal_segments, learning_matching_basal_segments):
                self._learn(self.basal_connections, learning_segments, self.active_basal_cells,
                            self.winner_basal_cells.sparse,
                            self.num_basal_potential, self.sample_size, self.max_synapses_per_segment,
                            self.initial_permanence, self.permanence_increment, self.permanence_decrement,
                            self.learning_threshold)
            if self.active_feedback_columns.sparse.size > 0 and self.active_basal_cells.sparse.size > 0:
                self.inhib_presynaptic_cells.sparse = np.concatenate((self.active_basal_cells.sparse,
                                                                      self.active_feedback_columns.sparse))
                # separate sampling
                if (self.winner_basal_cells.sparse.size <= self.sample_inhib_basal_size or
                        self.sample_inhib_basal_size == -1):
                    inhib_basal_candidates = self.winner_basal_cells.sparse
                else:
                    inhib_basal_candidates = np.random.choice(self.winner_basal_cells.sparse,
                                                              self.sample_inhib_basal_size,
                                                              replace=False)
                if (self.active_feedback_columns.sparse.size <= self.sample_inhib_feedback_size or
                        self.sample_inhib_feedback_size == -1):
                    inhib_feedback_candidates = self.active_feedback_columns.sparse
                else:
                    inhib_feedback_candidates = np.random.choice(self.active_feedback_columns.sparse,
                                                                 self.sample_inhib_feedback_size, replace=False)
                self.inhib_receptive_field.sparse = np.concatenate((inhib_basal_candidates,
                                                                    inhib_feedback_candidates))

                for learning_segments in (learning_active_inhibit_segments, learning_matching_inhibit_segments):
                    self._learn(self.inhib_connections, learning_segments,
                                self.inhib_presynaptic_cells,
                                self.inhib_receptive_field.sparse,
                                self.num_inhib_basal_potential + self.num_inhib_feedback_potential,
                                self.sample_inhib_basal_size + self.sample_inhib_feedback_size,
                                self.max_inhib_synapses_per_segment,
                                self.initial_permanence_inhib,
                                self.permanence_increment_inhib,
                                self.permanence_decrement_inhib,
                                self.learning_inhib_feedback_threshold + self.learning_inhib_basal_threshold
                                )

            self._learn(self.apical_connections, learning_matching_apical_segments, self.active_apical_cells,
                        self.winner_apical_cells.sparse,
                        self.num_apical_potential,
                        self.sample_apical_size, self.max_apical_synapses_per_segment,
                        self.initial_permanence_apical,
                        self.permanence_increment_apical,
                        self.permanence_decrement_apical,
                        self.learning_apical_threshold)

            # Punish incorrect predictions
            if self.predicted_segment_decrement != 0.0:
                for segment in basal_segments_to_punish:
                    self.basal_connections.adaptSegment(segment, self.active_basal_cells,
                                                        -self.predicted_segment_decrement, 0.0,
                                                        self.prune_zero_synapses, self.learning_threshold)
                for segment in apical_segments_to_punish:
                    self.apical_connections.adaptSegment(segment, self.active_apical_cells,
                                                         -self.predicted_segment_decrement_apical, 0.0,
                                                         self.prune_zero_synapses, self.learning_apical_threshold)
                if self.active_feedback_columns.sparse.size > 0:
                    for segment in inhibit_segments_to_punish:
                        self.inhib_connections.adaptSegment(segment, self.inhib_presynaptic_cells,
                                                            -self.predicted_segment_decrement_inhib, 0.0,
                                                            self.prune_zero_synapses,
                                                            self.learning_inhib_feedback_threshold + self.learning_inhib_basal_threshold)

            # Grow new segments
            if len(self.winner_basal_cells.sparse) > 0:
                self._learn_on_new_segments(self.basal_connections,
                                            np.concatenate((cells_to_grow_basal_segments,
                                                            cells_to_grow_apical_and_basal_segments)),
                                            self.winner_basal_cells.sparse,
                                            self.sample_size, self.max_synapses_per_segment,
                                            self.initial_permanence,
                                            self.max_segments_per_cell)
            if len(self.winner_apical_cells.sparse) > 0:
                self._learn_on_new_segments(self.apical_connections,
                                            np.concatenate((cells_to_grow_apical_segments,
                                                            cells_to_grow_apical_and_basal_segments)),
                                            self.winner_apical_cells.sparse,
                                            self.sample_apical_size, self.max_apical_synapses_per_segment,
                                            self.initial_permanence_apical,
                                            self.max_segments_per_cell_apical)
            # the problem may be that we don't specify sample size for feedback and basal separately
            if len(self.active_feedback_columns.sparse) > 0:
                self._learn_on_new_segments(self.inhib_connections,
                                            cells_to_grow_inhibit_segments,
                                            self.inhib_receptive_field.sparse,
                                            self.sample_inhib_basal_size + self.sample_inhib_feedback_size,
                                            self.max_inhib_synapses_per_segment,
                                            self.initial_permanence_inhib,
                                            self.max_segments_per_cell_inhib)

        self.active_basal_cells.sparse = np.unique(new_active_cells.astype('uint32'))
        self.winner_basal_cells.sparse = np.unique(new_winner_cells)

        n_active_columns = self.active_columns.sparse.size
        if n_active_columns != 0:
            anomaly = len(bursting_columns) / n_active_columns
        else:
            anomaly = 1.0

        self.anomaly_threshold = self.anomaly_threshold + (anomaly - self.anomaly[0]) / self.anomaly_window
        self.anomaly.append(anomaly)
        self.anomaly.pop(0)
        # pruning
        if learn:
            self.step += 1
            if self.enable_pruning_basal:
                self.segments_activity_basal = self.prune_segments(self.basal_connections, self.segments_activity_basal,
                                                                   self.active_basal_segments.astype('uint32'),
                                                                   self.pruning_period_basal)

            if self.enable_pruning_apical:
                self.segments_activity_apical = self.prune_segments(self.apical_connections,
                                                                    self.segments_activity_apical,
                                                                    self.active_apical_segments.astype('uint32'),
                                                                    self.pruning_period_apical)

            if self.enable_pruning_exec:
                self.segments_activity_exec = self.prune_segments(self.exec_feedback_connections,
                                                                  self.segments_activity_exec,
                                                                  self.active_exec_segments.astype('uint32'),
                                                                  self.pruning_period_exec)

            if self.enable_pruning_inhib:
                self.segments_activity_inhib = self.prune_segments(self.inhib_connections, self.segments_activity_inhib,
                                                                   self.active_inhib_segments.astype('uint32'),
                                                                   self.pruning_period_inhib)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
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
            connections.adaptSegment(segment, active_cells, permanence_increment, permanence_decrement,
                                     self.prune_zero_synapses, segmentThreshold)

            if sample_size == -1:
                max_new = len(winner_cells)
            else:
                max_new = sample_size - num_potential[segment]

            if max_synapses_per_segment != -1:
                synapse_counts = connections.numSynapses(segment)
                num_synapses_to_reach_max = max_synapses_per_segment - synapse_counts
                max_new = min(max_new, num_synapses_to_reach_max)
            if max_new > 0:
                connections.growSynapses(segment, winner_cells, initial_permanence, self.rng, max_new)

    def _learn_on_new_segments(self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
                               max_synapses_per_segment,
                               initial_permanence, max_segments_per_cell):
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
            new_segment = connections.createSegment(cell, max_segments_per_cell)
            connections.growSynapses(new_segment, growth_candidates, initial_permanence, self.rng,
                                     maxNew=num_new_synapses)

    def _calculate_learning(self, bursting_columns, correct_predicted_cells):
        """
        Calculates which segments to train and where to grow new segments
        :param bursting_columns: numpy array of columns' id
        :param correct_predicted_cells: numpy array of cells' id
        :return:
        """
        # Correctly predicted columns
        # choose active segments for correctly predicted cells
        learning_active_basal_segments = self.basal_connections.filterSegmentsByCell(self.active_basal_segments,
                                                                                     correct_predicted_cells)
        # choose all matching apical segments for correctly predicted segments
        # if there is no matching segment, we should grow an apical segment on this cell
        learning_matching_apical_segments, cells_to_grow_apical_segments = setCompare(self.matching_apical_segments,
                                                                                      correct_predicted_cells,
                                                                                      aKey=self.apical_connections.mapSegmentsToCells(
                                                                                          self.matching_apical_segments),
                                                                                      rightMinusLeft=True)
        # narrow apical segments to the best one per correctly predicted cell
        learning_matching_apical_segments = self._choose_best_segment_per_cell(self.apical_connections,
                                                                               correct_predicted_cells,
                                                                               learning_matching_apical_segments,
                                                                               self.num_apical_potential)
        # all cells with matching segments
        cells_for_matching_basal = self.basal_connections.mapSegmentsToCells(self.matching_basal_segments)
        cells_for_matching_apical = self.apical_connections.mapSegmentsToCells(self.matching_apical_segments)
        matching_cells = np.unique(cells_for_matching_basal)

        matching_cells_in_bursting_columns, bursting_columns_with_no_match = setCompare(matching_cells,
                                                                                        bursting_columns,
                                                                                        aKey=self._basal_columns_for_cells(
                                                                                            matching_cells),
                                                                                        rightMinusLeft=True)
        # choose cells with matching exec segments first for bursting columns
        if self.matching_exec_segments.size > 0:
            cells_with_matching_exec_segments = setCompare(matching_cells_in_bursting_columns,
                                                           self.matching_exec_segments,
                                                           bKey=self.exec_feedback_connections.mapSegmentsToCells(
                                                               self.matching_exec_segments))
            matching_cells_in_bursting_columns = self._filter_by_basal_columns(matching_cells_in_bursting_columns,
                                                                               np.unique(self._basal_columns_for_cells(
                                                                                   cells_with_matching_exec_segments)),
                                                                               invert=True)
            learning_matching_basal_segments3 = self._choose_best_segment_per_cell(self.basal_connections,
                                                                                   cells_with_matching_exec_segments,
                                                                                   self.matching_basal_segments,
                                                                                   self.num_basal_potential)
            learning_matching_apical_segments3 = self._choose_best_segment_per_cell(self.apical_connections,
                                                                                    cells_with_matching_exec_segments,
                                                                                    self.matching_apical_segments,
                                                                                    self.num_apical_potential)
            cells_without_apical_segments_mask = np.in1d(cells_with_matching_exec_segments,
                                                         self.apical_connections.mapSegmentsToCells(
                                                             self.matching_apical_segments),
                                                         invert=True)
            cells_to_grow_apical_segments3 = cells_with_matching_exec_segments[cells_without_apical_segments_mask]
            cells_without_basal_segments_mask = np.in1d(cells_with_matching_exec_segments,
                                                        self.basal_connections.mapSegmentsToCells(
                                                            self.matching_basal_segments),
                                                        invert=True)
            cells_to_grow_basal_segments3 = cells_with_matching_exec_segments[cells_without_basal_segments_mask]
        else:
            learning_matching_apical_segments3 = None
            learning_matching_basal_segments3 = None
            cells_to_grow_apical_segments3 = None
            cells_to_grow_basal_segments3 = None

        # then choose the best segments per cell
        if matching_cells_in_bursting_columns.size > 0:
            (learning_matching_basal_segments,
             learning_matching_apical_segments2,
             cells_to_grow_apical_segments2
             ) = self._choose_best_segment_per_column(
                matching_cells_in_bursting_columns)
        else:
            learning_matching_basal_segments = np.empty(0, dtype=np.int32)
            learning_matching_apical_segments2 = np.empty(0, dtype=np.int32)
            cells_to_grow_apical_segments2 = np.empty(0, dtype=np.int32)
        # cells on which new apical and basal segments will be grown
        cells_to_grow_apical_and_basal_segments = self._get_cells_with_fewest_segments(self.basal_connections,
                                                                                       self.apical_connections,
                                                                                       bursting_columns_with_no_match)

        # compile all segments and cells together
        cells_to_grow_apical_segments = [cells_to_grow_apical_segments, cells_to_grow_apical_segments2]
        if cells_to_grow_apical_segments3 is not None:
            cells_to_grow_apical_segments.append(cells_to_grow_apical_segments3)

        cells_to_grow_apical_segments = np.concatenate(cells_to_grow_apical_segments)

        learning_matching_apical_segments = [learning_matching_apical_segments, learning_matching_apical_segments2]
        if learning_matching_apical_segments3 is not None:
            learning_matching_apical_segments.append(learning_matching_apical_segments3)
        learning_matching_apical_segments = np.concatenate(learning_matching_apical_segments)

        if cells_to_grow_basal_segments3 is not None:
            cells_to_grow_basal_segments = cells_to_grow_basal_segments3
        else:
            cells_to_grow_basal_segments = np.empty(0)

        if learning_matching_basal_segments3 is not None:
            learning_matching_basal_segments = np.concatenate(
                [learning_matching_basal_segments, learning_matching_basal_segments3])

        winner_cells = np.concatenate(
            (correct_predicted_cells,
             self.basal_connections.mapSegmentsToCells(learning_matching_basal_segments),
             cells_to_grow_basal_segments,
             cells_to_grow_apical_and_basal_segments)
        )

        # Incorrectly predicted columns
        incorrect_matching_basal_mask = np.isin(self._basal_columns_for_cells(cells_for_matching_basal),
                                                self.active_columns.sparse, invert=True)
        incorrect_matching_apical_mask = np.isin(self._basal_columns_for_cells(cells_for_matching_apical),
                                                 self.active_columns.sparse, invert=True)

        basal_segments_to_punish = self.matching_basal_segments[incorrect_matching_basal_mask]
        apical_segments_to_punish = self.matching_apical_segments[incorrect_matching_apical_mask]

        # cells on which new inhibitory segments will be grown and matching segments that will be learned
        cells_to_inhibit = winner_cells
        learning_matching_inhibit_segments, cells_to_grow_inhibit_segments = setCompare(self.matching_inhib_segments,
                                                                                        cells_to_inhibit,
                                                                                        aKey=self.inhib_connections.mapSegmentsToCells(
                                                                                            self.matching_inhib_segments),
                                                                                        rightMinusLeft=True)
        # reinforce correct inhibition and punish incorrect
        correct_inhibit_cells, incorrect_inhibit_cells = setCompare(self.inhibited_cells, self.active_columns.sparse,
                                                                    aKey=self._basal_columns_for_cells(
                                                                        self.inhibited_cells),
                                                                    leftMinusRight=True)
        learning_active_inhibit_segments = self.inhib_connections.filterSegmentsByCell(self.active_inhib_segments,
                                                                                       correct_inhibit_cells)
        inhibit_segments_to_punish = self.inhib_connections.filterSegmentsByCell(self.active_inhib_segments,
                                                                                 incorrect_inhibit_cells)
        # maybe need to punish also incorrectly matching inhibit segments

        return (learning_active_basal_segments.astype('uint32'),
                learning_matching_basal_segments.astype('uint32'),
                cells_to_grow_basal_segments.astype('uint32'),
                learning_matching_apical_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'),
                learning_active_inhibit_segments.astype('uint32'),
                learning_matching_inhibit_segments.astype('uint32'),
                cells_to_grow_inhibit_segments.astype('uint32'),
                basal_segments_to_punish.astype('uint32'),
                apical_segments_to_punish.astype('uint32'),
                inhibit_segments_to_punish.astype('uint32'),
                cells_to_grow_apical_and_basal_segments.astype('uint32'),
                winner_cells.astype('uint32'))

    def _choose_best_segment_per_column(self, cells):
        """
        Chooses best matching segment per column among the cells, using apical tie breaking.
        :param cells: numpy array of cells' id
        :return:
        """
        candidate_basal_segments = self.basal_connections.filterSegmentsByCell(self.matching_basal_segments, cells)
        candidate_apical_segments = self._choose_best_segment_per_cell(self.apical_connections, cells,
                                                                       self.matching_apical_segments,
                                                                       self.num_apical_potential)
        intersection_mask = np.in1d(self.basal_connections.mapSegmentsToCells(candidate_basal_segments),
                                    self.apical_connections.mapSegmentsToCells(candidate_apical_segments))
        candidate_basal_with_apical_neighbour = candidate_basal_segments[intersection_mask]

        # for segment, that have no adjacent apical segment the score is zero, else score is sigmoid(best_apical_segment) - 0.5
        cells_for_apical_segments = self.apical_connections.mapSegmentsToCells(candidate_apical_segments)
        cells_for_basal_segments = self.basal_connections.mapSegmentsToCells(candidate_basal_with_apical_neighbour)
        tiebreaker = np.zeros_like(candidate_basal_segments)
        # WARNING, lazy realization of tiebreaking! May be slow!
        # TODO make optimized tiebreaking
        tiebreaker[intersection_mask] = np.array(
            [exp(self.num_apical_potential[candidate_apical_segments[cells_for_apical_segments == x]].sum()) for x
             in cells_for_basal_segments]
        )
        #
        one_per_column_filter = argmaxMulti(
            self.num_basal_potential[candidate_basal_segments] + tiebreaker / (tiebreaker + 1) - 0.5,
            groupKeys=self._basal_columns_for_cells(
                self.basal_connections.mapSegmentsToCells(candidate_basal_segments)))
        learning_basal_segments = candidate_basal_segments[one_per_column_filter]
        cells_for_learning_basal_segments = self.basal_connections.mapSegmentsToCells(learning_basal_segments)
        learning_apical_segments = candidate_apical_segments[np.in1d(cells_for_apical_segments,
                                                                     cells_for_learning_basal_segments)]
        # if there is no matching apical segment on learning_basal_segment: grow one
        cells_to_grow_apical_segments = cells_for_learning_basal_segments[np.in1d(cells_for_learning_basal_segments,
                                                                                  cells_for_apical_segments,
                                                                                  invert=True)]

        return (learning_basal_segments.astype('uint32'),
                learning_apical_segments.astype('uint32'),
                cells_to_grow_apical_segments.astype('uint32'))

    @staticmethod
    def _choose_best_segment_per_cell(connections, cells, segments, num_potential):
        """
        Calculates best matching segment per cell.
        :param connections:
        :param cells: numpy array of cells' id
        :param segments: numpy array of segments' id
        :param num_potential:
        :return:
        """
        candidate_segments = connections.filterSegmentsByCell(segments, cells)

        # Narrow it down to one pair per cell.
        if candidate_segments.size > 0:
            one_per_cell_filter = argmaxMulti(num_potential[candidate_segments],
                                              groupKeys=connections.mapSegmentsToCells(candidate_segments))
            learning_segments = candidate_segments[one_per_cell_filter]
        else:
            learning_segments = np.empty(0)
        return learning_segments.astype('uint32')

    def _get_cells_with_fewest_segments(self, basal_connections, apical_connections, columns):
        """
        Calculates cells with fewest segments per column.
        :param basal_connections:
        :param apical_connections:
        :param columns:
        :return:
        """
        candidate_cells = getAllCellsInColumns(columns, self.basal_cells_per_column) + self.basal_range[0]

        # Arrange the segment counts into one row per minicolumn.
        # count apical and basal segments per cell
        segment_counts = np.reshape(
            basal_connections.getSegmentCounts(candidate_cells) + apical_connections.getSegmentCounts(candidate_cells),
            newshape=(len(columns), self.basal_cells_per_column))

        # Filter to just the cells that are tied for fewest in their minicolumn.
        min_segment_counts = np.amin(segment_counts, axis=1, keepdims=True)
        candidate_cells = candidate_cells[np.flatnonzero(segment_counts == min_segment_counts)]

        # Filter to one cell per column, choosing randomly from the minimums.
        # To do the random choice, add a random offset to each index in-place, using
        # casting to floor the result.
        _, one_per_column_filter, num_candidates_in_columns = np.unique(self._basal_columns_for_cells(candidate_cells),
                                                                        return_index=True, return_counts=True)

        offset_percents = np.empty(len(columns), dtype="float64")
        self.rng.initializeReal64Array(offset_percents)

        np.add(one_per_column_filter, offset_percents * num_candidates_in_columns, out=one_per_column_filter,
               casting="unsafe")

        return candidate_cells[one_per_column_filter].astype('uint32')

    @staticmethod
    def _activate_dendrites(connections, presynaptic_cells, activation_threshold, learning_threshold):
        """
        Calculates active and matching segments and predictive cells.
        :param connections:
        :param presynaptic_cells:
        :param activation_threshold:
        :param learning_threshold:
        :return:
        """
        # Active
        num_connected, num_potential = connections.computeActivityFull(presynaptic_cells,
                                                                       False)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= activation_threshold)
        predictive_cells = connections.mapSegmentsToCells(active_segments)  # with duplicates

        # Matching
        matching_segments = np.flatnonzero(num_potential >= learning_threshold)

        return active_segments, matching_segments, predictive_cells, num_potential

    def _basal_columns_for_cells(self, cells):
        """
        Calculates columns numbers for basal cells
        :param cells: numpy array of cells id
        :return: numpy array of columns id for every cell
        """
        if np.any(cells < self.basal_range[0]) or np.any(cells >= self.basal_range[1]):
            raise ValueError('cells are not in bounds')

        basal_cells = cells - self.basal_range[0]
        basal_columns = basal_cells // self.basal_cells_per_column
        return basal_columns.astype('int32')

    def _filter_by_basal_columns(self, cells, columns, invert=False):
        """
        Filters cells by specified columns
        :param cells: numpy array of cells id
        :param columns: numpy array of columns id
        :param invert: if true then return cells that not in columns
        :return: numpy array of cells id
        """
        columns_for_cells = self._basal_columns_for_cells(cells)
        return cells[np.in1d(columns_for_cells, columns, invert=invert)]

    def prune_segments(self, connections, segments_activity, active_segments, pruning_period):
        """
        Destroy segments that haven't been active for a period of time.

        :param pruning_period:
        :param active_segments:
        :param segments_activity:
        :param connections:
        :return:
        """
        if connections.segmentFlatListLength() > segments_activity.size:
            new_segments_activity = np.zeros(connections.segmentFlatListLength())
            new_segments_activity[:segments_activity.size] = segments_activity
            segments_activity = new_segments_activity
        segments_activity[active_segments] += 1

        if (self.step % pruning_period) == 0:
            to_prune = segments_activity == 0
            for segment in np.flatnonzero(to_prune):
                connections.destroySegment(segment)
            segments_activity[:] = 0
            segments_activity[to_prune] = -1

        return segments_activity


if __name__ == '__main__':
    pass
