from typing import Union

from htm.algorithms import TemporalMemory as HtmTemporalMemory
from htm.bindings.algorithms import ANMode


class TemporalMemory(HtmTemporalMemory):
    n_columns: int
    cells_per_column: int
    activation_threshold: int
    learning_threshold: int
    connected_permanence: float

    def __init__(
            self, n_columns, cells_per_column,
            initial_permanence, connected_permanence,
            activation_threshold: Union[int, float],
            learning_threshold: Union[int, float],
            max_new_synapse_count: Union[int, float],
            max_synapses_per_segment: Union[int, float],
            n_active_bits: int = None,
            **kwargs
    ):
        if n_active_bits is not None:
            activation_threshold = abs_or_relative(activation_threshold, n_active_bits)
            learning_threshold = abs_or_relative(learning_threshold, n_active_bits)
            max_new_synapse_count = abs_or_relative(max_new_synapse_count, n_active_bits)
            max_synapses_per_segment = abs_or_relative(max_synapses_per_segment, n_active_bits)
            # print(activation_threshold, learning_threshold, max_new_synapse_count, max_synapses_per_segment)

        super().__init__(
            columnDimensions=(n_columns, ),
            cellsPerColumn=cells_per_column,
            activationThreshold=activation_threshold,
            minThreshold=learning_threshold,
            initialPermanence=initial_permanence,
            connectedPermanence=connected_permanence,
            maxNewSynapseCount=max_new_synapse_count,
            maxSynapsesPerSegment=max_synapses_per_segment,
            anomalyMode=ANMode.DISABLED,
            **kwargs
        )
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence

    def __getstate__(self):
        # used to pickle object
        data = (
            self.n_columns, self.cells_per_column, self.activation_threshold,
            self.learning_threshold, self.initial_permanence, self.connected_permanence
        )
        return super().__getstate__(), data

    def __setstate__(self, state):
        # used to unpickle
        super_data, data = state

        super().__setstate__(super_data)

        (
            self.n_columns, self.cells_per_column, self.activation_threshold,
            self.learning_threshold, self.initial_permanence, self.connected_permanence
        ) = data

    @property
    def output_sdr_size(self):
        return self.n_columns


def abs_or_relative(value: Union[int, float], base: int):
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return int(base * value)
    else:
        ValueError(value)

