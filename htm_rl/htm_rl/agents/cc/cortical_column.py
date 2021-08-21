from htm_rl.agents.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.advanced.support.numpy_helpers import setCompare, argmaxMulti, getAllCellsInColumns
import numpy as np
from htm.bindings.math import Random
from math import exp
import copy
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.cc.utils import ExponentialDecayFunction, NoDecayFunction, LogisticExciteFunction, FixedExciteFunction

EPS = 1e-12
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
_TIE_BREAKER_FACTOR = 0.000001


class UnionTemporalPooler(SpatialPooler):
    """
  Experimental Union Temporal Pooler Python implementation. The Union Temporal
  Pooler builds a "union SDR" of the most recent sets of active columns. It is
  driven by active-cell input and, more strongly, by predictive-active cell
  input. The latter is more likely to produce active columns. Such winning
  columns will also tend to persist longer in the union SDR.
  """

    def __init__(self,
                 output_sparsity,
                 n_cortical_columns,
                 cells_per_cortical_column,
                 current_cc_id,
                 activeOverlapWeight=1.0,
                 predictedActiveOverlapWeight=0.0,
                 maxUnionActivity=0.20,
                 exciteFunctionType='Fixed',
                 decayFunctionType='NoDecay',
                 decayTimeConst=20.0,
                 synPermPredActiveInc=0.0,
                 synPermPreviousPredActiveInc=0.0,
                 historyLength=0,
                 minHistory=0,
                 connected_threshold_basal=0.5,
                 prune_zero_synapses_basal=True,
                 activation_threshold_basal=0,
                 learning_threshold_basal=0,
                 initial_permanence_basal=0.4,
                 permanence_increment_basal=0.1,
                 permanence_decrement_basal=0.01,
                 sample_size_basal=-1,
                 max_synapses_per_segment_basal=-1,
                 max_segments_per_cell_basal=255,
                 timeseries=True,
                 **kwargs):
        """
    Please see spatial_pooler.py in NuPIC for super class parameter
    descriptions.
    Class-specific parameters:
    -------------------------------------
    @param activeOverlapWeight: A multiplicative weight applied to
        the overlap between connected synapses and active-cell input
    @param predictedActiveOverlapWeight: A multiplicative weight applied to
        the overlap between connected synapses and predicted-active-cell input
    @param fixedPoolingActivationBurst: A Boolean, which, if True, has the
        Union Temporal Pooler grant a fixed amount of pooling activation to
        columns whenever they win the inhibition step. If False, columns'
        pooling activation is calculated based on their current overlap.
    @param exciteFunction: If fixedPoolingActivationBurst is False,
        this specifies the ExciteFunctionBase used to excite pooling
        activation.
    @param decayFunction: Specifies the DecayFunctionBase used to decay pooling
        activation.
    @param maxUnionActivity: Maximum sparsity of the union SDR
    @param decayTimeConst Time constant for the decay function
    @param minHistory don't perform union (output all zeros) until buffer
    length >= minHistory
    """

        super(UnionTemporalPooler, self).__init__(**kwargs)

        self._activeOverlapWeight = activeOverlapWeight
        self._predictedActiveOverlapWeight = predictedActiveOverlapWeight
        self._maxUnionActivity = maxUnionActivity

        self._exciteFunctionType = exciteFunctionType
        self._decayFunctionType = decayFunctionType
        self._synPermPredActiveInc = synPermPredActiveInc
        self._synPermPreviousPredActiveInc = synPermPreviousPredActiveInc

        self._historyLength = historyLength
        self._minHistory = minHistory

        self.n_cortical_columns = n_cortical_columns
        self.current_cc_id = current_cc_id
        self.output_sparsity = output_sparsity
        self.k = max(int(self.output_sparsity * self.getNumColumns()), 1)
        self.seed = kwargs['seed']

        # basal connections
        self.connected_threshold_basal = connected_threshold_basal
        self.activation_threshold_basal = activation_threshold_basal
        self.learning_threshold_basal = learning_threshold_basal
        self.initial_permanence_basal = initial_permanence_basal
        self.permanence_increment_basal = permanence_increment_basal
        self.permanence_decrement_basal = permanence_decrement_basal
        self.sample_size_basal = sample_size_basal
        self.max_synapses_per_segment_basal = max_synapses_per_segment_basal
        self.max_segments_per_cell_basal = max_segments_per_cell_basal
        self.prune_zero_synapses_basal = prune_zero_synapses_basal
        self.timeseries = timeseries

        if not isinstance(cells_per_cortical_column, list):
            self.cells_per_cortical_column = [cells_per_cortical_column]*self.n_cortical_columns
        else:
            assert len(cells_per_cortical_column) == self.n_cortical_columns
            self.cells_per_cortical_column = cells_per_cortical_column

        self.cc_id_range = list()
        shift = 0
        for cells in self.cells_per_cortical_column:
            self.cc_id_range.append((shift, shift+cells))
            shift += cells

        self.total_cells = sum(self.cells_per_cortical_column)
        assert 0 <= current_cc_id < n_cortical_columns

        self.connections_basal = Connections(numCells=self.total_cells,
                                             connectedThreshold=self.connected_threshold_basal,
                                             timeseries=self.timeseries)

        self.pre_all_cortical_activity = SDR(self.total_cells)
        self.matching_segments_basal = np.empty(0, dtype=UINT_DTYPE)
        self.cells_to_grow_segments_basal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_basal = np.empty(0, dtype=UINT_DTYPE)

        # initialize excite/decay functions

        if exciteFunctionType == 'Fixed':
            self._exciteFunction = FixedExciteFunction()
        elif exciteFunctionType == 'Logistic':
            self._exciteFunction = LogisticExciteFunction()
        else:
            raise NotImplementedError('unknown excite function type' + exciteFunctionType)

        if decayFunctionType == 'NoDecay':
            self._decayFunction = NoDecayFunction()
        elif decayFunctionType == 'Exponential':
            self._decayFunction = ExponentialDecayFunction(decayTimeConst)
        else:
            raise NotImplementedError('unknown decay function type' + decayFunctionType)

        # The maximum number of cells allowed in a single union SDR
        self._maxUnionCells = int(self.getNumColumns() * self._maxUnionActivity)

        # Scalar activation of potential union SDR cells; most active cells become
        # the union SDR
        self._poolingActivation = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)

        # include a small amount of tie-breaker when sorting pooling activation
        np.random.seed(self.seed)
        self._poolingActivation_tieBreaker = np.random.randn(self.getNumColumns()) * _TIE_BREAKER_FACTOR
        self._basalActivation_tieBreaker = np.random.randn(self.getNumColumns()) * _TIE_BREAKER_FACTOR

        # time since last pooling activation increment
        # initialized to be a large number
        self._poolingTimer = np.ones(self.getNumColumns(), dtype=REAL_DTYPE) * 1000

        # pooling activation level after the latest update, used for sigmoid decay function
        self._poolingActivationInitLevel = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)

        # Current union SDR; the output of the union pooler algorithm
        self._unionSDR = SDR(self.getNumColumns())

        # Indices of active cells from spatial pooler
        self._activeCells = SDR(self.getNumColumns())

        # lowest possible pooling activation level
        self._poolingActivationLowerBound = 0.1

        self._preActiveInput = np.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
        # predicted inputs from the last n steps
        self._prePredictedActiveInput = list()

        if self.seed:
            self.rng = Random(kwargs['seed'])
        else:
            self.rng = Random()

    def reset(self, boosting=True):
        """
    Reset the state of the Union Temporal Pooler.
    """

        # Reset Union Temporal Pooler fields
        self._poolingActivation = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
        self._unionSDR = SDR(self.getNumColumns())
        self._poolingTimer = np.ones(self.getNumColumns(), dtype=REAL_DTYPE) * 1000
        self._poolingActivationInitLevel = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
        self._preActiveInput = np.zeros(self.getNumInputs(), dtype=REAL_DTYPE)
        self._prePredictedActiveInput = list()

        # Reset Spatial Pooler fields
        if boosting:
            self.setOverlapDutyCycles(np.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
            self.setActiveDutyCycles(np.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
            self.setMinOverlapDutyCycles(np.zeros(self.getNumColumns(), dtype=REAL_DTYPE))
            self.setBoostFactors(np.ones(self.getNumColumns(), dtype=REAL_DTYPE))
        #
        self.pre_all_cortical_activity = SDR(self.total_cells)
        self.matching_segments_basal = np.empty(0, dtype=UINT_DTYPE)
        self.cells_to_grow_segments_basal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_basal = np.empty(0, dtype=UINT_DTYPE)

    def compute(self, input_active: SDR, correctly_predicted_input: SDR,
                learn: bool, pre_all_active_output: SDR = None):
        """
    Computes one cycle of the Union Temporal Pooler algorithm.
    @param input_active (SDR) Input bottom up feedforward activity
    @param correctly_predicted_input (SDR) Represents correctly predicted input
    @param learn (bool) A boolean value indicating whether learning should be performed
    @param pre_all_active_output (SDR) Represents all cortical output activity from previous step
    """
        assert input_active.dense.size == self.getNumInputs()
        assert correctly_predicted_input.dense.size == self.getNumInputs()
        self._updateBookeepingVars(learn)

        # Compute proximal dendrite overlaps with active and active-predicted inputs
        overlapsActive = self.connections.computeActivity(input_active, False)
        overlapsPredictedActive = self.connections.computeActivity(correctly_predicted_input, learn)
        totalOverlap = (overlapsActive * self._activeOverlapWeight +
                        overlapsPredictedActive *
                        self._predictedActiveOverlapWeight).astype(REAL_DTYPE)

        if learn:
            boostFactors = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
            self.getBoostFactors(boostFactors)
            boostedOverlaps = boostFactors * totalOverlap
        else:
            boostedOverlaps = totalOverlap

        activeCells = self._inhibitColumns(boostedOverlaps)
        self._activeCells.sparse = activeCells

        # Decrement pooling activation of all cells
        self._decayPoolingActivation()

        # Update the poolingActivation of current active Union Temporal Pooler cells
        self._addToPoolingActivation(activeCells, overlapsPredictedActive)

        # update union SDR
        self._getMostActiveCellsProximal()

        if pre_all_active_output is not None:
            self.pre_all_cortical_activity.sparse = pre_all_active_output.sparse
            self._getMostActiveCellsDistal(learn)

            if learn:
                self._adaptBasalSegments()
                self._growNewBasalSegments()

        if learn:
            # adapt permanence of connections from predicted active inputs to newly active cell
            # This step is the spatial pooler learning rule, applied only to the predictedActiveInput
            # Todo: should we also include unpredicted active input in this step?
            self._adaptSynapses(correctly_predicted_input, self._activeCells, self.getSynPermActiveInc(),
                                self.getSynPermInactiveDec())

            # Increase permanence of connections from predicted active inputs to cells in the union SDR
            # This is Hebbian learning applied to the current time step
            self._adaptSynapses(correctly_predicted_input, self._unionSDR, self._synPermPredActiveInc, 0.0)

            # adapt permanence of connections from previously predicted inputs to newly active cells
            # This is a reinforcement learning rule that considers previous input to the current cell
            for pre_input in self._prePredictedActiveInput:
                self._adaptSynapses(pre_input, self._activeCells,
                                    self._synPermPreviousPredActiveInc, 0.0)

            # Homeostasis learning inherited from the spatial pooler
            self._updateDutyCycles(totalOverlap.astype(UINT_DTYPE), self._activeCells)
            self._bumpUpWeakColumns()
            self._updateBoostFactors()
            if self._isUpdateRound():
                self._updateInhibitionRadius()
                self._updateMinDutyCycles()

        # save inputs from the previous time step
        self._preActiveInput = copy.copy(input_active.dense)
        if self._historyLength > 0:
            if len(self._prePredictedActiveInput) == self._historyLength:
                self._prePredictedActiveInput.pop(0)
            self._prePredictedActiveInput.append(correctly_predicted_input)

        return self._unionSDR

    def _decayPoolingActivation(self):
        """
    Decrements pooling activation of all cells
    """
        if self._decayFunctionType == 'NoDecay':
            self._poolingActivation = self._decayFunction.decay(self._poolingActivation)
        elif self._decayFunctionType == 'Exponential':
            self._poolingActivation = self._decayFunction.decay(
                self._poolingActivationInitLevel, self._poolingTimer)

        return self._poolingActivation

    def _addToPoolingActivation(self, activeCells, overlaps):
        """
    Adds overlaps from specified active cells to cells' pooling
    activation.
    @param activeCells: Indices of those cells winning the inhibition step
    @param overlaps: A current set of overlap values for each cell
    @return current pooling activation
    """
        self._poolingActivation[activeCells] = self._exciteFunction.excite(
            self._poolingActivation[activeCells], overlaps[activeCells])

        # increase pooling timers for all cells
        self._poolingTimer[self._poolingTimer >= 0] += 1

        # reset pooling timer for active cells
        self._poolingTimer[activeCells] = 0
        self._poolingActivationInitLevel[activeCells] = self._poolingActivation[activeCells]

        return self._poolingActivation

    def _getMostActiveCellsProximal(self):
        """
    Gets the most active cells in the Union SDR having at least non-zero
    activation in sorted order.
    @return: a list of cell indices
    """
        poolingActivation = self._poolingActivation
        nonZeroCells = np.argwhere(poolingActivation > 0)[:, 0]

        # include a tie-breaker before sorting
        poolingActivationSubset = poolingActivation[nonZeroCells] + \
                                  self._poolingActivation_tieBreaker[nonZeroCells]
        potentialUnionSDR = nonZeroCells[np.argsort(poolingActivationSubset)[::-1]]

        topCells = potentialUnionSDR[0: self._maxUnionCells]

        if max(self._poolingTimer) > self._minHistory:
            self._unionSDR.sparse = np.sort(topCells).astype(UINT_DTYPE)
        else:
            self._unionSDR.sparse = []

        return self._unionSDR

    def _getMostActiveCellsDistal(self, learn: bool):
        """
        Gets the most active from cells Union SDR by distal active segments.
        Called after feedforward proximal activation.
        :param learn:
        :return:
        """
        # Active
        num_connected, num_potential = self.connections_basal.computeActivityFull(self.pre_all_cortical_activity, learn)  # The role of "learn" parameter isn't clear
        active_segments = np.flatnonzero(num_connected >= self.activation_threshold_basal)
        self.matching_segments_basal = np.flatnonzero(num_potential >= self.learning_threshold_basal)
        self.num_potential_basal = num_potential
        # filter segments by active cells
        union_sdr = self._unionSDR.sparse + self.cc_id_range[self.current_cc_id][0]
        active_segments = self.connections_basal.filterSegmentsByCell(active_segments, union_sdr)
        self.matching_segments_basal = self.connections_basal.filterSegmentsByCell(self.matching_segments_basal, union_sdr)

        if active_segments.size > 0:
            active_cells = self.connections_basal.mapSegmentsToCells(active_segments)
            active_cells, counts = np.unique(active_cells, return_counts=True)
            # needs a tiebreaker because argpartition brings bias
            if active_cells.size >= self.k:
                indices = np.argpartition(counts + self._basalActivation_tieBreaker[active_cells], -self.k)[:self.k]
                top_k_cells = active_cells[indices]
                self._unionSDR.sparse = top_k_cells - self.cc_id_range[self.current_cc_id][0]
                self.cells_to_grow_segments_basal = np.empty(0, dtype=UINT_DTYPE)  #
            else:
                cells_with_matching_segments = np.unique(self.connections_basal.mapSegmentsToCells(self.matching_segments_basal))
                self.cells_to_grow_segments_basal = union_sdr[np.in1d(union_sdr, cells_with_matching_segments, invert=True)]
        else:
            cells_with_matching_segments = np.unique(
                self.connections_basal.mapSegmentsToCells(self.matching_segments_basal))
            self.cells_to_grow_segments_basal = union_sdr[
                np.in1d(union_sdr, cells_with_matching_segments, invert=True)]

    def _adaptBasalSegments(self):
            """
            Learn on basal segments.
            :return:
            """

            for segment in self.matching_segments_basal:
                self.connections_basal.adaptSegment(segment, self.pre_all_cortical_activity,
                                                    self.permanence_increment_basal, self.permanence_decrement_basal,
                                                    self.prune_zero_synapses_basal, self.learning_threshold_basal)
                cc_id = self._getSegmentInputCC(segment)
                receptor_field_cells = self._filterCellsByCC(self.pre_all_cortical_activity.sparse, cc_id)
                if self.sample_size_basal == -1:
                    max_new = len(receptor_field_cells)
                else:
                    max_new = self.sample_size_basal - self.num_potential_basal[segment]

                if self.max_synapses_per_segment_basal != -1:
                    synapse_counts = self.connections_basal.numSynapses(segment)
                    num_synapses_to_reach_max = self.max_synapses_per_segment_basal - synapse_counts
                    max_new = min(max_new, num_synapses_to_reach_max)
                if max_new > 0:
                    self.connections_basal.growSynapses(segment, receptor_field_cells, self.initial_permanence_basal, self.rng, max_new)

    def _growNewBasalSegments(self):
        # separate segment for every cc
        for cc_id in range(self.n_cortical_columns):
            growth_candidates = self._filterCellsByCC(self.pre_all_cortical_activity.sparse, cc_id)
            num_new_synapses = len(growth_candidates)
            if num_new_synapses > 0:
                if self.sample_size_basal != -1:
                    num_new_synapses = min(num_new_synapses, self.sample_size_basal)

                if self.max_synapses_per_segment_basal != -1:
                    num_new_synapses = min(num_new_synapses, self.max_synapses_per_segment_basal)

                for cell in self.cells_to_grow_segments_basal:
                    new_segment = self.connections_basal.createSegment(cell, self.max_segments_per_cell_basal)
                    self.connections_basal.growSynapses(new_segment, growth_candidates, self.initial_permanence_basal, self.rng,
                                             maxNew=num_new_synapses)

    def _filterCellsByCC(self, cells_id, cortical_column_id):
        r = self.cc_id_range[cortical_column_id]
        return cells_id[(cells_id >= r[0]) & (cells_id < r[1])]

    def _getCCForCell(self, cell):
        for cc_id, r in enumerate(self.cc_id_range):
            if r[0] <= cell < r[1]:
                return cc_id
        else:
            return None

    def _getSegmentInputCC(self, segment):
        presynaptic_cells = self.connections_basal.presynapticCellsForSegment(segment)
        return self._getCCForCell(presynaptic_cells[0])

    def getUnionSDR(self):
        return self._unionSDR


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
        segment_counts = segment_counts + tiebreaker*0.1

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


if __name__ == '__main__':
    pass
