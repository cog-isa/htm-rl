import numpy as np
from abc import ABCMeta, abstractmethod
from htm.bindings.algorithms import SpatialPooler


class ExciteFunctionBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def excite(self, current, amount):
        pass


class LogisticExciteFunction(ExciteFunctionBase):
    """
    Implementation of a logistic activation function for activation updating.
    Specifically, the function has the following form:
    f(x) = (maxValue - minValue) / (1 + exp(-steepness * (x - xMidpoint) ) ) + minValue
    Note: The excitation rate is linear. The activation function is
    logistic.
    """

    def __init__(self, xMidpoint=5, minValue=10, maxValue=20, steepness=1):
        """
        @param xMidpoint: Controls where function output is half of 'maxValue,'
                          i.e. f(xMidpoint) = maxValue / 2
        @param minValue: Minimum value of the function
        @param maxValue: Controls the maximum value of the function's range
        @param steepness: Controls the steepness of the "middle" part of the
                          curve where output values begin changing rapidly.
                          Must be a non-zero value.
        """
        assert steepness != 0

        self._xMidpoint = xMidpoint
        self._maxValue = maxValue
        self._minValue = minValue
        self._steepness = steepness

    def excite(self, currentActivation, inputs):
        """
        Increases current activation by amount.
        @param currentActivation (numpy array) Current activation levels for each cell
        @param inputs            (numpy array) inputs for each cell
        """

        currentActivation += self._minValue + (self._maxValue - self._minValue) / (
                1 + np.exp(-self._steepness * (inputs - self._xMidpoint)))

        return currentActivation


class FixedExciteFunction(ExciteFunctionBase):
    """
      Implementation of a simple fixed excite function
      The function reset the activation level to a fixed amount
      """

    def __init__(self, targetExcLevel=10.0):
        """
    """
        self._targetExcLevel = targetExcLevel

    def excite(self, currentActivation, inputs):
        """
        Increases current activation by a fixed amount.
        @param currentActivation (numpy array) Current activation levels for each cell
        @param inputs            (numpy array) inputs for each cell
        """

        currentActivation += self._targetExcLevel

        return currentActivation


class DecayFunctionBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def decay(self, current, amount):
        pass


class NoDecayFunction(DecayFunctionBase):
    """
  Implementation of no decay.
  """

    def decay(self, current, amount=0):
        return current


class ExponentialDecayFunction(DecayFunctionBase):
    """
  Implementation of exponential decay.
  f(t) = exp(- lambda * t)
  lambda is the decay constant. The time constant is 1 / lambda
  """

    def __init__(self, time_constant=10.0):
        """
    @param (float) time_constant: positive exponential decay time constant.
    """
        assert not time_constant < 0
        self._lambda_constant = 1 / float(time_constant)

    def decay(self, initActivationLevel, timeSinceActivation):
        """
    @param initActivationLevel: initial activation level
    @param timeSinceActivation: time since the activation
    @return: activation level after decay
    """
        activationLevel = np.exp(-self._lambda_constant * timeSinceActivation) * initActivationLevel
        return activationLevel


class LogisticDecayFunction(DecayFunctionBase):
    """
  Implementation of logistic decay.
  f(t) = maxValue / (1 + exp(-steepness * (tMidpoint - t) ) )
  tMidpoint is when activation decays to half of its initial level
  steepness controls the steepness of the decay function around tMidpoint
  """

    def __init__(self, tMidpoint=10, steepness=.5):
        """
    @param tMidpoint: Controls where function output is half of 'maxValue,'
                      i.e. f(xMidpoint) = maxValue / 2
    @param steepness: Controls the steepness of the "middle" part of the
                      curve where output values begin changing rapidly.
                      Must be a non-zero value.
    """
        assert steepness != 0

        self._xMidpoint = tMidpoint
        self._steepness = steepness

    def decay(self, initActivationLevel, timeSinceActivation):
        """
    @param initActivationLevel: initial activation level
    @param timeSinceActivation: time since the activation
    @return: activation level after decay
    """

        activationLevel = initActivationLevel / (
                1 + np.exp(-self._steepness * (self._xMidpoint - timeSinceActivation)))

        return activationLevel


def get_receptive_field(sp: SpatialPooler, cell: int):
    receptive_field = np.zeros(sp.getInputDimensions())
    segment = sp.connections.getSegment(cell, 0)
    synapses = sp.connections.synapsesForSegment(segment)
    for synapse in synapses:
        if sp.connections.permanenceForSynapse(synapse) > sp.getSynPermConnected():
            presynaptic_cell = sp.connections.presynapticCellForSynapse(synapse)
            receptive_field[presynaptic_cell // receptive_field.shape[1],
                            presynaptic_cell % receptive_field.shape[1]] = 1
    return receptive_field
