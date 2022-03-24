import copy

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.math import Random
from htm.bindings.sdr import SDR

from htm_rl.modules.htm.utils import (
    ExponentialDecayFunction, NoDecayFunction, LogisticExciteFunction,
    FixedExciteFunction
)

EPS = 1e-12
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 0.000001


# noinspection PyPep8Naming
class AblationUtp(SpatialPooler):
    """
      Experimental Union Temporal Pooler Python implementation. The Union Temporal
      Pooler builds a "union SDR" of the most recent sets of active columns. It is
      driven by active-cell input and, more strongly, by predictive-active cell
      input. The latter is more likely to produce active columns. Such winning
      columns will also tend to persist longer in the union SDR.
      """

    def __init__(
            self,
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
            second_boosting=True,
            **kwargs
    ):
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
        self.second_boosting = second_boosting

        super(AblationUtp, self).__init__(**kwargs)

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

    # noinspection PyMethodOverriding
    def compute(
            self, input_active: SDR, correctly_predicted_input: SDR, learn: bool
    ):
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
            if self.second_boosting:
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
            # noinspection PyArgumentList
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
        poolingActivationSubset = (
                poolingActivation[nonZeroCells] + self._poolingActivation_tieBreaker[nonZeroCells]
        )
        potentialUnionSDR = nonZeroCells[np.argsort(poolingActivationSubset)[::-1]]

        topCells = potentialUnionSDR[0: self._maxUnionCells]

        if max(self._poolingTimer) > self._minHistory:
            self._unionSDR.sparse = np.sort(topCells).astype(UINT_DTYPE)
        else:
            self._unionSDR.sparse = []

        return self._unionSDR

    def getUnionSDR(self):
        return self._unionSDR
