import numpy as np
from htm.bindings.sdr import SDR


class CustomUtp:
    _initial_pooling = 1
    _pooling_decay = 0.1

    def __init__(self,
                 inputDimensions,
                 columnDimensions,
                 initial_pooling,
                 pooling_decay,
                 permanence_inc,
                 permanence_dec,
                 sparsity,
                 active_weight,
                 predicted_weight,
                 receptive_field_sparsity,
                 activation_threshold,
                 history_length,
                 union_sdr_sparsity,
                 prev_perm_inc,
                 untemporal_learning_enabled=True,
                 union_learning_enabled=True,
                 history_learning_enabled=True,
                 limit_union_cells=True,
                 ):
        input_shape = inputDimensions
        output_shape = columnDimensions
        out_size = np.prod(output_shape)
        in_size = np.prod(input_shape)
        self._union_sdr = SDR(output_shape)
        self._pooling_activations = np.zeros(output_shape)
        self._shape = output_shape
        self._initial_pooling = initial_pooling
        self._pooling_decay = pooling_decay
        self._permanence_inc = permanence_inc
        self._permanence_dec = permanence_dec
        self.winners_num = int(np.ceil(sparsity * out_size))
        self.active_weight = active_weight
        self.predicted_weight = predicted_weight
        self.input_dimensions = input_shape
        self.field_size = int(receptive_field_sparsity * in_size)
        self.activation_threshold = activation_threshold
        self.history_length = history_length
        self.prev_perm_inc = prev_perm_inc

        self.connections = np.random.normal(self.activation_threshold, 0.1, (out_size, in_size))

        self.receptive_fields = np.zeros((out_size, in_size))
        self.set_receptive_fields()

        self.sensitivity = np.random.uniform(0, 1, (out_size, in_size))

        self.predict_history = []

        self.cells_in_union = int(np.prod(self._shape)*union_sdr_sparsity)

        self.untemporal_learning_enabled = untemporal_learning_enabled
        self.union_learning_enabled = union_learning_enabled
        self.history_learning_enabled = history_learning_enabled
        self.limit_union_cells = limit_union_cells

    def set_receptive_fields(self):
        for cell in self.receptive_fields:
            receptive_field = np.random.choice(np.prod(self.input_dimensions), self.field_size, replace=False)
            cell[receptive_field] = 1

    def pooling_decay_step(self):
        self._pooling_activations[self._pooling_activations != 0] -= self._pooling_decay
        self._pooling_activations = self._pooling_activations.clip(0, 1)

    def get_active_cells(self, data: np.ndarray) -> SDR:
        tie_breaker = np.random.normal(0, 0.01, self._shape)
        activations = (self.connections * self.sensitivity) @ data + tie_breaker

        most_active = np.argpartition(activations.flatten(), -self.winners_num)[-self.winners_num:]
        result_sdr = SDR(self._shape)
        result_sdr.sparse = most_active
        return result_sdr

    def adapt_synapses(self, input_: np.ndarray, output_: np.ndarray, inc, dec):
        self.connections[output_, :] -= dec

        active_synapses = np.ix_(output_, input_)
        self.connections[active_synapses] += dec
        self.connections[active_synapses] += inc
        self.connections = self.connections.clip(0, 1)

    def untemporal_learning(self, predicted_neurons, winners):
        self.adapt_synapses(
            input_=predicted_neurons.sparse,
            output_=winners,
            inc=self._permanence_inc,
            dec=self._permanence_dec
        )

    def union_learning(self, predicted_neurons):
        self.adapt_synapses(
            input_=predicted_neurons.sparse,
            output_=self.getUnionSDR().sparse,
            inc=self._permanence_inc,
            dec=0
        )

    def history_learning(self, predicted_neurons: SDR, winners: np.ndarray):
        for prev_predict in self.predict_history:
            self.adapt_synapses(
                input_=prev_predict.sparse,
                output_=winners,
                inc=self.prev_perm_inc,
                dec=0
            )
        if len(self.predict_history) == self.history_length:
            self.predict_history.pop(0)
        self.predict_history.append(predicted_neurons)

    def update_permanences(self, predicted_neurons: SDR, winners: np.ndarray):
        if self.untemporal_learning_enabled:
            self.untemporal_learning(predicted_neurons, winners)

        if self.union_learning_enabled:
            self.union_learning(predicted_neurons)

        if self.history_learning_enabled:
            self.history_learning(predicted_neurons, winners)

    def compute_continuous(self, active_neurons: SDR, predicted_neurons: SDR, learn: bool = True):
        weighted_input = active_neurons.dense * self.active_weight + predicted_neurons.dense * self.predicted_weight
        winners = self.get_active_cells(weighted_input)
        self.pooling_decay_step()
        self._pooling_activations[winners.sparse] += self._initial_pooling
        self._pooling_activations = self._pooling_activations.clip(0, 1)
        self._union_sdr.dense = self._pooling_activations != 0

        if learn:
            self.update_permanences(predicted_neurons, winners)

    def count_overlap(self, active_neurons: SDR, predicted_neurons: SDR) -> np.ndarray:
        active_synapses = np.logical_and(
            self.receptive_fields,
            self.connections > self.activation_threshold
        )

        cells_overlap = np.count_nonzero(np.logical_and(
            active_neurons.dense == 1,
            active_synapses
        ), axis=1) * self.active_weight + np.count_nonzero(np.logical_and(
            predicted_neurons.dense == 1,
            active_synapses
        ), axis=1) * self.predicted_weight

        return cells_overlap

    def choose_winners(self, overlap: np.ndarray):
        tie_breaker = np.random.normal(0, 0.01, np.prod(self._shape))
        overlap += tie_breaker
        winners = np.argpartition(overlap, -self.winners_num)[-self.winners_num:]
        return winners

    def update_union_sdr(self):
        if self.limit_union_cells:
            tie_breaker = np.random.normal(0, 0.001, np.prod(self._shape))
            self._union_sdr.dense = (self._pooling_activations + tie_breaker) > np.partition(
                (self._pooling_activations+tie_breaker).flatten(), -self.cells_in_union-1
            )[-self.cells_in_union-1]
        else:
            self._union_sdr.dense = self._pooling_activations != 0

    def compute(self, active_neurons: SDR, predicted_neurons: SDR, learn: bool = True):
        overlap = self.count_overlap(active_neurons, predicted_neurons)
        winners = self.choose_winners(overlap)

        self.pooling_decay_step()
        self._pooling_activations[winners] += self._initial_pooling
        self._pooling_activations = self._pooling_activations.clip(0, 1)

        self.update_union_sdr()

        if learn:
            self.update_permanences(predicted_neurons, winners)

    def getUnionSDR(self):
        return self._union_sdr

    def getNumInputs(self):
        return self.input_dimensions

    def reset(self):
        self._pooling_activations = np.zeros(self._pooling_activations.shape)
        self._union_sdr = SDR(self._union_sdr.dense.shape)
        self.predict_history = []
