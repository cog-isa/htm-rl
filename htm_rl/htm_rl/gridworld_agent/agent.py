from collections import defaultdict
from typing import List, Optional, Mapping

from htm.bindings.sdr import SDR

from htm_rl.representations.sar_sdr_encoder import SarSdrEncoder
from htm_rl.representations.sdr import SparseSdr
from htm_rl.representations.temporal_memory import TemporalMemory


class Agent:
    tm: TemporalMemory
    encoder: SarSdrEncoder
    _proximal_input_sdr: SDR

    def __init__(self, tm: TemporalMemory, encoder: SarSdrEncoder, formatter):
        self.tm = tm
        self.encoder = encoder
        self._proximal_input_sdr = SDR(self.tm.n_columns)
        self.format = formatter

    def reset(self, condition=True) -> None:
        if condition:
            self.tm.reset()

    def train_cycle(self, sequence: List[SparseSdr], print_enabled=False, reset_enabled=True):
        if reset_enabled:
            self.tm.reset()

        for proximal_input in sequence:
            self.train_one_step(proximal_input, print_enabled=print_enabled)

    def train_one_step(self, proximal_input: SparseSdr, print_enabled=False):
        self.activate_memory(
            proximal_input,
            learn_enabled=True, output_active_cells=False, print_enabled=print_enabled
        )
        self.depolarize_memory(learn_enabled=True, output_predictive_cells=False, print_enabled=print_enabled)

    def predict_cycle(self, initial_input: SparseSdr, n_steps, print_enabled=False, reset_enabled=True):
        if reset_enabled:
            self.tm.reset()

        proximal_input = initial_input
        for i in range(n_steps):
            predictive_cells = self.predict_one_step(proximal_input, print_enabled)
            proximal_input = self.columns_from_cells_sparse(predictive_cells)

            if print_enabled:
                sar_superposition = self.encoder.decode(proximal_input)
                print(self.format(sar_superposition))

    def predict_one_step(self, proximal_input: SparseSdr, print_enabled=False):
        self.activate_memory(
            proximal_input,
            learn_enabled=False, output_active_cells=False, print_enabled=print_enabled
        )
        predictive_cells = self.depolarize_memory(
            learn_enabled=False, output_predictive_cells=True, print_enabled=print_enabled
        )
        return predictive_cells

    def activate_memory(
            self, proximal_input: SparseSdr,
            learn_enabled: bool, output_active_cells: bool, print_enabled: bool
    ) -> Optional[SparseSdr]:
        self._proximal_input_sdr.sparse = proximal_input
        self.tm.compute(self._proximal_input_sdr, learn=learn_enabled)

        if output_active_cells or print_enabled:
            active_cells: SDR = self.tm.getActiveCells()
            if print_enabled:
                print(self._str_from_cells(active_cells, 'Active'))
            return active_cells.sparse

    def depolarize_memory(
            self, learn_enabled: bool, output_predictive_cells: bool, print_enabled: bool
    ) -> Optional[SparseSdr]:
        self.tm.activateDendrites(learn=learn_enabled)

        if output_predictive_cells or print_enabled:
            predictive_cells: SDR = self.tm.getPredictiveCells()
            if print_enabled:
                print(self._str_from_cells(predictive_cells, 'Predictive'))
            return predictive_cells.sparse

    def _str_from_cells(self, cells_sdr: SDR, name: str) -> str:
        flatten_cells_sparse_sdr = cells_sdr.sparse
        cells_sparse_sdr = [[] for _ in range(self.tm.cells_per_column)]
        for ind in flatten_cells_sparse_sdr:
            column, layer = divmod(ind, self.tm.cells_per_column)
            cells_sparse_sdr[layer].append(column)

        first_line = f'{self.encoder.format(cells_sparse_sdr[0])} {name}'
        substrings = [first_line]
        for layer_ind in range(1, self.tm.cells_per_column):
            substrings.append(self.encoder.format(cells_sparse_sdr[layer_ind]))
        return '\n'.join(substrings)

    def columns_from_cells_sparse(self, cells_sparse_sdr: SparseSdr) -> SparseSdr:
        # flatten active cells indices -> active column indices
        # active column - at least 1 active cell in it
        cpc = self.tm.cells_per_column
        return list(set(cell_ind // cpc for cell_ind in cells_sparse_sdr))

    def get_presynaptic_connections(self, active_presynaptic_cells: Optional[SparseSdr]) -> Mapping[int, List[int]]:
        if active_presynaptic_cells is None:
            active_presynaptic_cells = self.tm.getActiveCells().sparse

        active_presynaptic_cells = set(active_presynaptic_cells)
        tm, connections = self.tm, self.tm.connections

        def get_presynaptic_cells_for_segment(segment):
            return (
                connections.presynapticCellForSynapse(synapse)
                for synapse in connections.synapsesForSegment(segment)
                if connections.permanenceForSynapse(synapse) >= tm.connected_permanence
            )

        active_segments = (
            (connections.cellForSegment(segment), get_presynaptic_cells_for_segment(segment))
            for segment in self.tm.getActiveSegments()
        )
        presynaptic_connections = defaultdict(list)
        for postsynaptic_cell, presynaptic_cells in active_segments:
            presynaptic_connections[postsynaptic_cell].extend(
                presynaptic_cell
                for presynaptic_cell in presynaptic_cells
                if presynaptic_cell in active_presynaptic_cells
            )
        return presynaptic_connections
