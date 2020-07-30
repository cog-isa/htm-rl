import pickle
from collections import defaultdict
from typing import List, Callable, Dict, Set, Tuple

from htm.bindings.sdr import SDR

from htm_rl.common.base_sar import SarSuperposition, Sar
from htm_rl.common.int_sdr_encoder import BitRange
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import trace
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class LegacyMemory:
    tm: TemporalMemory
    encoder: SarSdrEncoder

    _proximal_input_sdr: SDR    # cached SDR

    def __init__(
            self, tm: TemporalMemory, encoder: SarSdrEncoder,
            sdr_formatter: Callable[[SparseSdr], str],
            sar_superposition_formatter: Callable[[SarSuperposition], str],
            collect_anomalies: bool = False
    ):
        self.tm = tm
        self.encoder = encoder
        self.format_sdr = sdr_formatter
        self.format_sar_superposition = sar_superposition_formatter
        self.anomalies = [] if collect_anomalies else None
        self._proximal_input_sdr = SDR(self.tm.n_columns)

    def reset(self):
        self.tm.reset()

    def train(self, sar: Sar, verbosity: int):
        proximal_input = self.encoder.encode(sar)
        self.process(proximal_input, learn=True, verbosity=verbosity)

    def predict_from_sar(self, initial_sar: Sar, n_steps, verbosity: bool):
        proximal_input = self.encoder.encode(initial_sar)
        for i in range(n_steps):
            _, depolarized_cells = self.process(proximal_input, learn=False, verbosity=verbosity)
            proximal_input = self.columns_from_cells(depolarized_cells)

    def process(
            self, proximal_input: SparseSdr, learn: bool, verbosity: int
    ) -> Tuple[SparseSdr, SparseSdr]:
        """
        Given new piece of proximal input data, processes it by sequentially activating cells
        and then depolarizes them, making prediction about next proximal input.

        :param proximal_input: column activations sparse SDR.
        :param learn: whether or not to force learning.
        :param verbose: whether or not to print debug traces.
        :return: tuple (active cells, depolarized cells) of sparse SDRs.
        """
        self.print_sar_superposition(verbosity, 2, proximal_input)

        active_cells = self.activate_cells(proximal_input, learn)
        self.print_cells(verbosity, 3, active_cells, 'Active')

        depolarized_cells = self.depolarize_cells(learn)
        self.print_cells(verbosity, 3, depolarized_cells, 'Predictive')

        return active_cells, depolarized_cells

    def activate_cells(self, proximal_input: SparseSdr, learn: bool) -> SparseSdr:
        """
        Given proximal input SDR, activates TM cells and [optionally] applies learning step.

        Learning is applied to correct/incorrect predictions - to active segments of depolarized cells.

        :param proximal_input: active columns sparse SDR.
        :param learn: whether or not to force learning.
        :return: active cells sparse SDR.
        """

        # sparse SDR can be any iterable, but HTM SDR object accepts only `list`
        if not isinstance(proximal_input, list):
            proximal_input = list(proximal_input)

        self._proximal_input_sdr.sparse = proximal_input
        self.tm.compute(self._proximal_input_sdr, learn=learn)

        # if anomalies tracking enabled, append this step
        if self.anomalies is not None:
            self.anomalies.append(1 - self.tm.anomaly)

        return self.tm.getActiveCells().sparse

    def depolarize_cells(self, learn: bool) -> SparseSdr:
        """
        Given the current state of active cells, activates cells' segments
        leading to cells depolarization.

        :param learn: whether or not TM should make a learning step too.
        :return: depolarized cells sparse SDR
        """
        self.tm.activateDendrites(learn=learn)
        return self.tm.getPredictiveCells().sparse

    def columns_from_cells(self, cells_sparse_sdr: SparseSdr) -> SparseSdr:
        """
        Converts cells sparse SDR to columns sparse SDR.

        :param cells_sparse_sdr: cells sparse SDR
        :return: columns sparse SDR
        """
        cpc = self.tm.cells_per_column
        return set(cell_ind // cpc for cell_ind in cells_sparse_sdr)

    def filter_cells_by_columns_range(self, cells: SparseSdr, columns_range: BitRange) -> SparseSdr:
        cpc = self.tm.cells_per_column
        left, right = columns_range
        # to cells range
        left, right = left * cpc, right * cpc
        return [cell for cell in cells if left <= cell < right]

    def active_segments(self, current_active_cells: SparseSdr) -> Dict[int, List[Set[int]]]:
        """
        Gets all active segments.

        Returned active segments is a dictionary: depolarized cell -> list of active segments.

        Each active segment is a sparse SDR of active presynaptic cells, which induced
        segment's activation. They stored as a `set`.

        :param current_active_cells: active cells sparse SDR.
        :return: active segments
        """
        tm, connections = self.tm, self.tm.connections
        active_cells = frozenset(current_active_cells)

        def get_presynaptic_cells_for_segment(segment):
            # take only _connected_ synapses
            return frozenset(
                connections.presynapticCellForSynapse(synapse)
                for synapse in connections.synapsesForSegment(segment)
                if connections.permanenceForSynapse(synapse) >= tm.connected_permanence
            )

        # active segment: (postsynaptic _depolarized_ cell; presynaptic !connected! cells)
        all_active_segments = (
            (connections.cellForSegment(segment), get_presynaptic_cells_for_segment(segment))
            for segment in tm.getActiveSegments()
        )
        # filter out synapses to _inactive_ cells, keeping only _active_ presynaptic cells
        active_segments = defaultdict(list)
        for depolarized_cell, active_segment in all_active_segments:
            # keep only synapses to active presynaptic cells
            active_presynaptic_cells = active_segment & active_cells

            assert len(active_presynaptic_cells) >= tm.activation_threshold
            active_segments[depolarized_cell].append(active_presynaptic_cells)

        # active segment: (postsynaptic _depolarized_ cell; presynaptic !active+connected! cells)
        return active_segments

    def print_cells(self, verbosity: int, req_level: int, cells_sparse_sdr: SparseSdr, mark: str = ''):
        """
        Prints cells sparse SDR layer by layer, each one on a separate line.
        :param verbosity: accepted verbosity level to print debug traces.
        :param req_level: required minimal verbosity level defined by caller.
        :param cells_sparse_sdr:
        :param mark: optional description mark to print near the first line
        """
        if req_level > verbosity:
            return

        cpc = self.tm.cells_per_column

        layerwise_sparse_sdrs = [[] for _ in range(cpc)]
        for cell_index in cells_sparse_sdr:
            column, layer = divmod(cell_index, cpc)
            layerwise_sparse_sdrs[layer].append(column)

        first_line = f'{self.format_sdr(layerwise_sparse_sdrs[0])} {mark}'
        lines = [first_line]
        for layer_ind in range(1, cpc):
            lines.append(self.format_sdr(layerwise_sparse_sdrs[layer_ind]))
        trace(verbosity, req_level, '\n'.join(lines))

    def print_sar_superposition(self, verbosity: int, req_level: int, proximal_input: SparseSdr):
        if req_level <= verbosity:
            sar_superposition = self.encoder.decode(proximal_input)
            trace(verbosity, req_level, self.format_sar_superposition(sar_superposition))

    def save_tm_state(self):
        """Saves TM state."""
        return pickle.dumps(self.tm)

    def restore_tm_state(self, tm_dump):
        """Restores saved TM state."""
        self.tm = pickle.loads(tm_dump)
