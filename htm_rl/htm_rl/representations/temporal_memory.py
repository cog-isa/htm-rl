from htm.bindings.algorithms import TemporalMemory as TM


class TemporalMemory(TM):
    n_columns: int
    cells_per_column: int
    activation_threshold: float
    learning_threshold: float
    connected_permanence: float

    def __init__(
            self, n_columns, cells_per_column, activation_threshold,
            learning_threshold, initial_permanence, connected_permanence
    ):
        super().__init__(
            columnDimensions=(n_columns, ),
            cellsPerColumn=cells_per_column,
            activationThreshold=activation_threshold,
            minThreshold=learning_threshold,
            initialPermanence=initial_permanence,
            connectedPermanence=connected_permanence,
        )
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence

    def __getstate__(self):
        data = (
            self.n_columns, self.cells_per_column, self.activation_threshold,
            self.learning_threshold, self.initial_permanence, self.connected_permanence
        )
        return super().__getstate__(), data

    def __setstate__(self, state):
        super_data, data = state

        super().__setstate__(super_data)

        (
            self.n_columns, self.cells_per_column, self.activation_threshold,
            self.learning_threshold, self.initial_permanence, self.connected_permanence
        ) = data

