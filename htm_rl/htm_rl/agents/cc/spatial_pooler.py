from htm_rl.agents.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.advanced.support.numpy_helpers import setCompare
import numpy as np
from htm.bindings.math import Random
import copy
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.cc.utils import ExponentialDecayFunction, NoDecayFunction, LogisticExciteFunction, \
    FixedExciteFunction

EPS = 1e-12
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 0.000001


class ColumnPooler(object):
    """
  This class constitutes a temporary implementation for a cross-column pooler.
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

    def __init__(self,
                 inputWidth,
                 cellCount=4096,
                 sdrSize=40,
                 onlineLearning=False,
                 maxSdrSize=None,
                 minSdrSize=None,
                 prune_zero_synapses=True,

                 # Proximal
                 synPermProximalInc=0.1,
                 synPermProximalDec=0.001,
                 initialProximalPermanence=0.6,
                 sampleSizeProximal=20,
                 minThresholdProximal=10,
                 connectedPermanenceProximal=0.50,
                 predictedInhibitionThreshold=20,

                 # Distal
                 synPermDistalInc=0.1,
                 synPermDistalDec=0.001,
                 initialDistalPermanence=0.6,
                 sampleSizeDistal=20,
                 activationThresholdDistal=13,
                 connectedPermanenceDistal=0.50,
                 inertiaFactor=1.,

                 seed=42):
        """
    Parameters:
    ----------------------------
    @param  inputWidth (int)
            The number of bits in the feedforward input
    @param  sdrSize (int)
            The number of active cells in an object SDR
    @param  onlineLearning (Bool)
            Whether or not the column pooler should learn in online mode.
    @param  maxSdrSize (int)
            The maximum SDR size for learning.  If the column pooler has more
            than this many cells active, it will refuse to learn.  This serves
            to stop the pooler from learning when it is uncertain of what object
            it is sensing.
    @param  minSdrSize (int)
            The minimum SDR size for learning.  If the column pooler has fewer
            than this many active cells, it will create a new representation
            and learn that instead.  This serves to create separate
            representations for different objects and sequences.
            If online learning is enabled, this parameter should be at least
            inertiaFactor*sdrSize.  Otherwise, two different objects may be
            incorrectly inferred to be the same, as SDRs may still be active
            enough to learn even after inertial decay.
    @param  synPermProximalInc (float)
            Permanence increment for proximal synapses
    @param  synPermProximalDec (float)
            Permanence decrement for proximal synapses
    @param  initialProximalPermanence (float)
            Initial permanence value for proximal synapses
    @param  sampleSizeProximal (int)
            Number of proximal synapses a cell should grow to each feedforward
            pattern, or -1 to connect to every active bit
    @param  minThresholdProximal (int)
            Number of active synapses required for a cell to have feedforward
            support
    @param  connectedPermanenceProximal (float)
            Permanence required for a proximal synapse to be connected
    @param  predictedInhibitionThreshold (int)
            How much predicted input must be present for inhibitory behavior
            to be triggered.  Only has effects if onlineLearning is true.
    @param  synPermDistalInc (float)
            Permanence increment for distal synapses
    @param  synPermDistalDec (float)
            Permanence decrement for distal synapses
    @param  sampleSizeDistal (int)
            Number of distal synapses a cell should grow to each lateral
            pattern, or -1 to connect to every active bit
    @param  initialDistalPermanence (float)
            Initial permanence value for distal synapses
    @param  activationThresholdDistal (int)
            Number of active synapses required to activate a distal segment
    @param  connectedPermanenceDistal (float)
            Permanence required for a distal synapse to be connected
    @param  inertiaFactor (float)
            The proportion of previously active cells that remain
            active in the next timestep due to inertia (in the absence of
            inhibition).  If onlineLearning is enabled, should be at most
            1 - learningTolerance, or representations may incorrectly become
            mixed.
    @param  seed (int)
            Random number generator seed
    """

        assert maxSdrSize is None or maxSdrSize >= sdrSize
        assert minSdrSize is None or minSdrSize <= sdrSize

        self.inputWidth = inputWidth
        self.cellCount = cellCount
        self.sdrSize = sdrSize
        self.onlineLearning = onlineLearning
        if maxSdrSize is None:
            self.maxSdrSize = sdrSize
        else:
            self.maxSdrSize = maxSdrSize
        if minSdrSize is None:
            self.minSdrSize = sdrSize
        else:
            self.minSdrSize = minSdrSize
        self.synPermProximalInc = synPermProximalInc
        self.synPermProximalDec = synPermProximalDec
        self.initialProximalPermanence = initialProximalPermanence
        self.connectedPermanenceProximal = connectedPermanenceProximal
        self.sampleSizeProximal = sampleSizeProximal
        self.minThresholdProximal = minThresholdProximal
        self.predictedInhibitionThreshold = predictedInhibitionThreshold
        self.synPermDistalInc = synPermDistalInc
        self.synPermDistalDec = synPermDistalDec
        self.initialDistalPermanence = initialDistalPermanence
        self.connectedPermanenceDistal = connectedPermanenceDistal
        self.sampleSizeDistal = sampleSizeDistal
        self.activationThresholdDistal = activationThresholdDistal
        self.inertiaFactor = inertiaFactor
        self.prune_zero_synapses = prune_zero_synapses

        self.activeCells = SDR(cellCount)
        self.prevActiveCells = SDR(cellCount)
        self.active_segments_proximal = np.empty(0, dtype=UINT_DTYPE)
        self.active_segments_distal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_proximal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_distal = np.empty(0, dtype=UINT_DTYPE)
        self._random = Random(seed)

        # These sparse matrices will hold the synapses for each segment.
        # Each row represents one segment on a cell, so each cell potentially has
        # 1 proximal segment and 1+len(lateralInputWidths) distal segments.
        self.proximal_connections = Connections(cellCount, self.connectedPermanenceProximal)
        self.internal_distal_connections = Connections(cellCount, self.connectedPermanenceDistal)

        self.useInertia = True

        # initialize proximal segments
        for cell in range(self.proximal_connections.numCells()):
            self.proximal_connections.createSegment(cell, 1)

    def compute(self, feedforwardInput=(),
                feedforwardGrowthCandidates=None, learn=True,
                corrPredictedInput=None):
        """
    Runs one time step of the column pooler algorithm.
    @param  feedforwardInput SDR
            Sorted indices of active feedforward input bits
    @param  feedforwardGrowthCandidates SDR or None
            Sorted indices of feedforward input bits that active cells may grow
            new synapses to. If None, the entire feedforwardInput is used.
    @param  learn (bool)
            If True, we are learning a new object
    @param corrPredictedInput SDR or None
           Sorted indices of predicted cells in the TM layer.
    """

        if feedforwardGrowthCandidates is None:
            feedforwardGrowthCandidates = feedforwardInput

        # inference step
        if not learn:
            self._computeInferenceMode(feedforwardInput)

        # learning step
        elif not self.onlineLearning:
            self._computeLearningMode(feedforwardInput,
                                      feedforwardGrowthCandidates)
        # online learning step
        else:
            if ((corrPredictedInput is not None) and
                    (corrPredictedInput.sparse.size > self.predictedInhibitionThreshold)):
                self._computeInferenceMode(corrPredictedInput)
                self._computeLearningMode(corrPredictedInput,
                                          feedforwardGrowthCandidates)
            elif not self.minSdrSize <= self.activeCells.sparse.size <= self.maxSdrSize:
                # If the pooler doesn't have a single representation, try to infer one,
                # before actually attempting to learn.
                self._computeInferenceMode(feedforwardInput)
                self._computeLearningMode(feedforwardInput,
                                          feedforwardGrowthCandidates)
            else:
                # If there isn't predicted input and we have a single SDR,
                # we are extending that representation and should just learn.
                self._computeLearningMode(feedforwardInput,
                                          feedforwardGrowthCandidates)

    def _computeLearningMode(self, feedforwardInput,
                             feedforwardGrowthCandidates):
        """
    Learning mode: we are learning a new object in an online fashion. If there
    is no prior activity, we randomly activate 'sdrSize' cells and create
    connections to incoming input. If there was prior activity, we maintain it.
    If we have a union, we simply do not learn at all.
    These cells will represent the object and learn distal connections to each
    other and to lateral cortical columns.
    Parameters:
    ----------------------------
    @param  feedforwardInput SDR
    @param  feedforwardGrowthCandidates SDR
    """
        # If there are not enough previously active cells, then we are no longer on
        # a familiar object.  Either our representation decayed due to the passage
        # of time (i.e. we moved somewhere else) or we were mistaken.  Either way,
        # create a new SDR and learn on it.
        # This case is the only way different object representations are created.
        # enforce the active cells in the output layer
        if self.activeCells.sparse.size < self.minSdrSize:
            active_cells = self._random.sample(np.arange(0, self.numberOfCells(), dtype=UINT_DTYPE), self.sdrSize)
            active_cells.sort()
            self.activeCells.sparse = active_cells

        self.prevActiveCells.sparse = self.activeCells.sparse

        # If we have a union of cells active, don't learn.  This primarily affects
        # online learning.
        if self.activeCells.sparse.size > self.maxSdrSize:
            return

        # Finally, now that we have decided which cells we should be learning on, do
        # the actual learning.
        if feedforwardInput.sparse.size > 0:
            # segment_id == cell_id, as we grow only one segment for cell
            self._learn(self.proximal_connections, self.activeCells.sparse, feedforwardInput,
                        feedforwardGrowthCandidates.sparse, self.num_potential_proximal, self.sampleSizeProximal,
                        self.numberOfInputs(),
                        self.initialProximalPermanence, self.synPermProximalInc, self.synPermProximalDec,
                        self.minThresholdProximal)

        if self.prevActiveCells.sparse.size > 0:
            learning_segments_distal, cells_to_grow_segments_distal = setCompare(self.active_segments_distal,
                                                                                 self.activeCells.sparse,
                                                                                 aKey=self.internal_distal_connections.mapSegmentsToCells(
                                                                                     self.active_segments_distal),
                                                                                 rightMinusLeft=True)
            self._learn(self.internal_distal_connections, learning_segments_distal, self.prevActiveCells,
                        self.prevActiveCells.sparse, self.num_potential_distal, self.sampleSizeDistal,
                        self.numberOfCells(),
                        self.initialDistalPermanence, self.synPermDistalInc, self.synPermDistalDec,
                        self.activationThresholdDistal)
            self._learn_on_new_segments(self.internal_distal_connections, cells_to_grow_segments_distal,
                                        self.prevActiveCells.sparse,
                                        self.sampleSizeDistal, self.numberOfCells(), self.initialDistalPermanence, 1)

    def _computeInferenceMode(self, feedforwardInput):
        """
    Inference mode: if there is some feedforward activity, perform
    spatial pooling on it to recognize previously known objects, then use
    lateral activity to activate a subset of the cells with feedforward
    support. If there is no feedforward activity, use lateral activity to
    activate a subset of the previous active cells.
    Parameters:
    ----------------------------
    @param  feedforwardInput: SDR
     """

        prevActiveCells = self.activeCells

        # Calculate the feedforward supported cells
        overlaps_proximal, num_potential_proximal = self.proximal_connections.computeActivityFull(feedforwardInput,
                                                                                                  False)
        self.num_potential_proximal = num_potential_proximal
        self.active_segments_proximal = np.flatnonzero(overlaps_proximal >= self.minThresholdProximal)
        feedforwardSupportedCells = self.proximal_connections.mapSegmentsToCells(self.active_segments_proximal)

        # Calculate the number of active segments on each cell
        numActiveSegmentsByCell = np.zeros(self.cellCount, dtype=UINT_DTYPE)
        cells_for_segments = self.proximal_connections.mapSegmentsToCells(self.active_segments_distal)
        cells, segment_counts = np.unique(cells_for_segments, return_counts=True)
        numActiveSegmentsByCell[cells] = segment_counts

        chosenCells = []

        # First, activate the FF-supported cells that have the highest number of
        # lateral active segments (as long as it's not 0)
        if len(feedforwardSupportedCells) == 0:
            pass
        else:
            numActiveSegsForFFSuppCells = numActiveSegmentsByCell[
                feedforwardSupportedCells]

            # This loop will select the FF-supported AND laterally-active cells, in
            # order of descending lateral activation, until we exceed the sdrSize
            # quorum - but will exclude cells with 0 lateral active segments.
            ttop = np.max(numActiveSegsForFFSuppCells)
            while ttop > 0 and len(chosenCells) < self.sdrSize:
                chosenCells = np.union1d(chosenCells,
                                         feedforwardSupportedCells[numActiveSegsForFFSuppCells >= ttop])
                ttop -= 1

        # If we haven't filled the sdrSize quorum, add in inertial cells.
        if len(chosenCells) < self.sdrSize:
            if self.useInertia:
                prevCells = np.setdiff1d(prevActiveCells.sparse, chosenCells)
                inertialCap = int(len(prevCells) * self.inertiaFactor)
                if inertialCap > 0:
                    numActiveSegsForPrevCells = numActiveSegmentsByCell[prevCells]
                    # We sort the previously-active cells by number of active lateral
                    # segments (this really helps).  We then activate them in order of
                    # descending lateral activation.
                    sortIndices = np.argsort(numActiveSegsForPrevCells)[::-1]
                    prevCells = prevCells[sortIndices]
                    numActiveSegsForPrevCells = numActiveSegsForPrevCells[sortIndices]

                    # We use inertiaFactor to limit the number of previously-active cells
                    # which can become active, forcing decay even if we are below quota.
                    prevCells = prevCells[:inertialCap]
                    numActiveSegsForPrevCells = numActiveSegsForPrevCells[:inertialCap]

                    # Activate groups of previously active cells by order of their lateral
                    # support until we either meet quota or run out of cells.
                    ttop = np.max(numActiveSegsForPrevCells)
                    while ttop >= 0 and len(chosenCells) < self.sdrSize:
                        chosenCells = np.union1d(chosenCells,
                                                 prevCells[numActiveSegsForPrevCells >= ttop])
                        ttop -= 1

        # If we haven't filled the sdrSize quorum, add cells that have feedforward
        # support and no lateral support.
        discrepancy = self.sdrSize - len(chosenCells)
        if discrepancy > 0:
            remFFcells = np.setdiff1d(feedforwardSupportedCells, chosenCells)

            # Inhibit cells proportionally to the number of cells that have already
            # been chosen. If ~0 have been chosen activate ~all of the feedforward
            # supported cells. If ~sdrSize have been chosen, activate very few of
            # the feedforward supported cells.

            # Use the discrepancy:sdrSize ratio to determine the number of cells to
            # activate.
            n = (len(remFFcells) * discrepancy) // self.sdrSize
            # Activate at least 'discrepancy' cells.
            n = max(n, discrepancy)
            # If there aren't 'n' available, activate all of the available cells.
            n = min(n, len(remFFcells))

            if len(remFFcells) > n:
                selected = self._random.sample(remFFcells, n)
                chosenCells = np.append(chosenCells, selected)
            else:
                chosenCells = np.append(chosenCells, remFFcells)

        chosenCells.sort()
        self.activeCells.sparse = chosenCells.astype(UINT_DTYPE)

    def activateDistalDendrites(self, learn):
        overlaps_distal, num_potential_distal = self.internal_distal_connections.computeActivityFull(self.activeCells,
                                                                                                     learn)
        self.num_potential_distal = num_potential_distal
        self.active_segments_distal = np.flatnonzero(overlaps_distal >= self.activationThresholdDistal)

    def numberOfInputs(self):
        """
    Returns the number of inputs into this layer
    """
        return self.inputWidth

    def numberOfCells(self):
        """
    Returns the number of cells in this layer.
    @return (int) Number of cells
    """
        return self.cellCount

    def getActiveCells(self):
        """
    Returns the indices of the active cells.
    @return (list) Indices of active cells.
    """
        return self.activeCells

    def reset(self):
        """
    Reset internal states. When learning this signifies we are to learn a
    unique new object.
    """
        self.activeCells.sparse = []
        self.prevActiveCells.sparse = []
        self.active_segments_proximal = np.empty(0, dtype=UINT_DTYPE)
        self.active_segments_distal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_proximal = np.empty(0, dtype=UINT_DTYPE)
        self.num_potential_distal = np.empty(0, dtype=UINT_DTYPE)

    def getUseInertia(self):
        """
    Get whether we actually use inertia  (i.e. a fraction of the
    previously active cells remain active at the next time step unless
    inhibited by cells with both feedforward and lateral support).
    @return (Bool) Whether inertia is used.
    """
        return self.useInertia

    def setUseInertia(self, useInertia):
        """
    Sets whether we actually use inertia (i.e. a fraction of the
    previously active cells remain active at the next time step unless
    inhibited by cells with both feedforward and lateral support).
    @param useInertia (Bool) Whether inertia is used.
    """
        self.useInertia = useInertia

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
        """
        Learn on specified segments
        :param connections: exemplar of Connections class
        :param learning_segments: list of segments' id
        :param active_cells: SDR
        :param winner_cells: SDR (cells to which connections will be grown)
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
                connections.growSynapses(segment, winner_cells, initial_permanence, self._random, max_new)

    def _learn(self, connections, learning_segments, active_cells, winner_cells, num_potential, sample_size,
               max_synapses_per_segment,
               initial_permanence, permanence_increment, permanence_decrement, segmentThreshold):
        """
        Learn on specified segments
        :param connections: exemplar of Connections class
        :param learning_segments: list of segments' id
        :param active_cells: SDR
        :param winner_cells: SDR (cells to which connections will be grown)
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
                connections.growSynapses(segment, winner_cells, initial_permanence, self._random, max_new)

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
            connections.growSynapses(new_segment, growth_candidates, initial_permanence, self._random,
                                     maxNew=num_new_synapses)


class UnionTemporalPooler(SpatialPooler):
    """
  Experimental Union Temporal Pooler Python implementation. The Union Temporal
  Pooler builds a "union SDR" of the most recent sets of active columns. It is
  driven by active-cell input and, more strongly, by predictive-active cell
  input. The latter is more likely to produce active columns. Such winning
  columns will also tend to persist longer in the union SDR.
  """

    def __init__(self,
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

        self.seed = kwargs['seed']

        if self.seed:
            self.rng = Random(kwargs['seed'])
        else:
            self.rng = Random()

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
        self._poolingActivation_tieBreaker = np.empty(self.getNumColumns(), dtype=REAL64_DTYPE)
        self._basalActivation_tieBreaker = np.empty(self.getNumColumns(), dtype=REAL64_DTYPE)
        self.rng.initializeReal64Array(self._poolingActivation_tieBreaker)
        self.rng.initializeReal64Array(self._basalActivation_tieBreaker)
        self._poolingActivation_tieBreaker *= _TIE_BREAKER_FACTOR
        self._basalActivation_tieBreaker *= _TIE_BREAKER_FACTOR

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

    def compute(self, input_active: SDR, correctly_predicted_input: SDR,
                learn: bool):
        """
    Computes one cycle of the Union Temporal Pooler algorithm.
    @param input_active (SDR) Input bottom up feedforward activity
    @param correctly_predicted_input (SDR) Represents correctly predicted input
    @param learn (bool) A boolean value indicating whether learning should be performed
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

    def getUnionSDR(self):
        return self._unionSDR


class TemporalDifferencePooler(SpatialPooler):
    def __init__(self, **kwargs):
        super(TemporalDifferencePooler, self).__init__(**kwargs)
        self._predictedCells = SDR(self.getNumColumns())

    def compute(self, input_active: SDR, input_predicted: SDR,
                learn: bool) -> SDR:
        """Computes one cycle of the Temporal Difference Pooler algorithm.
        @param input_active (SDR) Input bottom up feedforward activity
        @param input_predicted (SDR) Represents predicted input
        @param learn (bool) A boolean value indicating whether learning should be performed
        """
        assert input_active.dense.size == self.getNumInputs()
        assert input_predicted.dense.size == self.getNumInputs()
        self._updateBookeepingVars(learn)

        # minus phase
        if input_predicted.sparse.size > 0:
            overlapsPredicted = self.connections.computeActivity(input_predicted, learn)
            predictedCells = self._inhibitColumns(overlapsPredicted)
            self._predictedCells.sparse = predictedCells
        else:
            self._predictedCells.sparse = []

        # plus phase
        overlapsActive = self.connections.computeActivity(input_active, False)

        if learn:
            boostFactors = np.zeros(self.getNumColumns(), dtype=REAL_DTYPE)
            self.getBoostFactors(boostFactors)
            boostedOverlapsActive = boostFactors * overlapsActive
        else:
            boostedOverlapsActive = overlapsActive

        activeCells = self._inhibitColumns(boostedOverlapsActive)

        self._activeCells.sparse = activeCells

        if learn:
            if self._predictedCells.sparse.size > 0:
                # Temporal Diffrence Learning: deltaW = (x*y)_plus_phase - (x*y)_minus_phase
                correctly_predicted_cells = SDR(self.getNumColumns())
                not_predicted_cells = SDR(self.getNumColumns())
                not_active_cells = SDR(self.getNumColumns())
                not_predicted_input = SDR(self.getNumInputs())
                not_active_input = SDR(self.getNumInputs())

                correctly_predicted_cells.sparse = np.intersect1d(self._activeCells.sparse,
                                                                  self._predictedCells.sparse)
                not_predicted_cells.sparse = np.setdiff1d(self._activeCells.sparse,
                                                          self._predictedCells.sparse)
                not_active_cells.sparse = np.setdiff1d(self._predictedCells.sparse,
                                                       self._activeCells.sparse)

                not_predicted_input.sparse = np.setdiff1d(input_active.sparse,
                                                          input_predicted.spase)
                not_active_input.sparse = np.setdiff1d(input_predicted.spase,
                                                       input_active.sparse)

                self._adaptSynapses(not_predicted_input, correctly_predicted_cells,
                                    self.getSynPermActiveInc(), 0)
                self._adaptSynapses(not_active_input, correctly_predicted_cells,
                                    -self.getSynPermInactiveDec(), 0)

                self._adaptSynapses(input_active, not_predicted_cells,
                                    self.getSynPermActiveInc(), 0)

                self._adaptSynapses(input_predicted, not_active_cells,
                                    -self.getSynPermInactiveDec(), 0)
            else:
                # usual SP learning
                self._adaptSynapses(input_active, self._activeCells, self.getSynPermActiveInc(),
                                    self.getSynPermInactiveDec())

            # Homeostasis learning inherited from the spatial pooler
            self._updateDutyCycles(overlapsActive.astype(UINT_DTYPE), self._activeCells)
            self._bumpUpWeakColumns()
            self._updateBoostFactors()
            if self._isUpdateRound():
                self._updateInhibitionRadius()
                self._updateMinDutyCycles()

        return self._activeCells
