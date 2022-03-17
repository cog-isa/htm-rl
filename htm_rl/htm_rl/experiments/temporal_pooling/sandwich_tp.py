import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR


class SandwichTp:
    def __init__(self, **kwargs):
        self.initial_pooling = kwargs['initial_pooling']
        self.pooling_decay = kwargs['pooling_decay']
        self.lower_sp = SpatialPooler(**kwargs['lower_sp_conf'])
        self.upper_sp = SpatialPooler(**kwargs['upper_sp_conf'])

        self._unionSDR = SDR(kwargs['upper_sp_conf']['columnDimensions'])
        self._unionSDR.dense = np.zeros(kwargs['upper_sp_conf']['columnDimensions'])
        self._pooling_activations = np.zeros(kwargs['upper_sp_conf']['inputDimensions'])

    def _pooling_decay_step(self):
        self._pooling_activations[self._pooling_activations != 0] -= self.pooling_decay
        self._pooling_activations = self._pooling_activations.clip(0, 1)


    def compute(self, active_neurons: SDR, predicted_neurons: SDR, learn: bool = True):
        self._pooling_decay_step()

        input_representation = SDR(self._pooling_activations.shape)
        self.lower_sp.compute(active_neurons, learn=learn, output=input_representation)
        self._pooling_activations[input_representation.sparse] += self.initial_pooling
        self._pooling_activations = self._pooling_activations.clip(0, 1)

        sdr_for_upper = SDR(self._pooling_activations.shape)
        sdr_for_upper.dense = self._pooling_activations != 0
        self.upper_sp.compute(sdr_for_upper, learn=learn, output=self._unionSDR)

    def getUnionSDR(self):
        return self._unionSDR

    def getNumInputs(self):
        return self.lower_sp.getNumInputs()

    def reset(self):
        self._pooling_activations = np.zeros(self._pooling_activations.shape)
        self._unionSDR = SDR(self._unionSDR.dense.shape)
        self._unionSDR.dense = np.zeros(self._unionSDR.dense.shape)