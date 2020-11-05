import pickle
from collections import defaultdict
from typing import List, Callable, Dict, Set, Tuple

from htm.bindings.sdr import SDR

from htm_rl.common.base_sa import SaSuperposition, Sa
from htm_rl.common.base_sar import Sar
from htm_rl.common.int_sdr_encoder import BitRange
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import trace
from htm_rl.htm_plugins.temporal_memory import TemporalMemory

from watcher.writer import Writer


class Memory:
    tm: TemporalMemory
    encoder: SaSdrEncoder

    _proximal_input_sdr: SDR    # cached SDR

    def __init__(
            self, tm: TemporalMemory, encoder: SaSdrEncoder,
            sdr_formatter: Callable[[SparseSdr], str],
            sa_superposition_formatter: Callable[[SaSuperposition], str],
            collect_anomalies: bool = False,
            start_indicator: Sa = None,
            output_file: str = None
    ):
        self.start_indicator = start_indicator
        self.tm = tm
        self.encoder = encoder
        self.format_sdr = sdr_formatter
        self.format_sa_superposition = sa_superposition_formatter
        self.anomalies = [] if collect_anomalies else None
        self._proximal_input_sdr = SDR(self.tm.n_columns)

        self.output_file = output_file
        if self.output_file is not None:
            self.writer = Writer(self.tm)
        else:
            self.writer = None

    def reset(self):
        self.tm.reset()
        if self.start_indicator is not None:
            self.train(self.start_indicator, verbosity=0)

    def train(self, sa: Sa, verbosity: int):
        proximal_input = self.encoder.encode(sa)
        self.process(proximal_input, learn=True, verbosity=verbosity)

    def predict_from_sa(self, initial_sa: Sa, n_steps, verbosity: int):
        proximal_input = self.encoder.encode(initial_sa)
        for i in range(n_steps):
            _, depolarized_cells = self.process(proximal_input, learn=False, verbosity=verbosity)
            proximal_input = self.columns_from_cells(depolarized_cells)

    def process(
            self, proximal_input: SparseSdr, learn: bool, verbosity: int
    ) -> Tuple[SparseSdr, SparseSdr]:
        """
        Given new piece of proximal input data, processes it by sequentially activating cells
        and then depolarizes them, making prediction about next proximal input.

        :param proximal_input: sparse SDR of column activations.
        :param learn: whether or not to force learning.
        :param verbosity: accepted verbosity level to print debug traces.
        :return: tuple (active cells, depolarized cells) of sparse SDRs.
        """
        self.print_sa_superposition(verbosity, 2, proximal_input)

        active_cells = self.activate_cells(proximal_input, learn)
        self.print_cells(verbosity, 3, active_cells, 'Active')

        depolarized_cells = self.depolarize_cells(learn)
        self.print_cells(verbosity, 3, depolarized_cells, 'Predictive')

        return active_cells, depolarized_cells

    def activate_cells(self, proximal_input: SparseSdr, learn: bool) -> SparseSdr:
        """
        Given proximal input SDR, activates TM cells and [optionally] applies learning step.

        Learning is applied to correct/incorrect predictions - to active segments of depolarized cells.

        :param proximal_input: sparse SDR of active columns.
        :param learn: whether or not to force learning.
        :return: sparse SDR of active cells.
        """

        # sparse SDR can be any iterable, but HTM SDR object accepts only `list`
        if not isinstance(proximal_input, list):
            proximal_input = list(proximal_input)

        self._proximal_input_sdr.sparse = proximal_input
        self.tm.compute(self._proximal_input_sdr, learn=learn)

        # if anomalies tracking enabled, append results from this step
        if self.anomalies is not None:
            self.anomalies.append(1 - self.tm.anomaly)

        return self.tm.getActiveCells().sparse

    def depolarize_cells(self, learn: bool) -> SparseSdr:
        """
        Given the current state of active cells, activates cells' segments
        leading to cells depolarization.

        :param learn: whether or not TM should make a learning step too.
        :return: sparse SDR of depolarized cells
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

    def filter_cells_by_columns(self, cells: SparseSdr, columns: SparseSdr) -> SparseSdr:
        columns = frozenset(columns)
        cpc = self.tm.cells_per_column
        return [
            cell
            for cell in cells
            if (cell // cpc) in columns
        ]

    def active_segments(self, active_cells: SparseSdr) -> Dict[int, List[Set[int]]]:
        """
        Gets all active segments.

        Returns active segments as a dictionary: depolarized cell -> list of active segments.

        Each active segment is a sparse SDR of active presynaptic cells, which induced
        the segment's activation. They stored as a `set`.

        :param active_cells: sparse SDR of active cells.
        :return: active segments
        """
        tm, connections = self.tm, self.tm.connections
        active_cells = frozenset(active_cells)

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

            if len(active_presynaptic_cells) >= tm.activation_threshold:
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

        layer_wise_sparse_sdrs = [[] for _ in range(cpc)]
        for cell_index in cells_sparse_sdr:
            column, layer = divmod(cell_index, cpc)
            layer_wise_sparse_sdrs[layer].append(column)

        first_line = f'{self.format_sdr(layer_wise_sparse_sdrs[0])} {mark}'
        lines = [first_line]
        for layer_ind in range(1, cpc):
            lines.append(self.format_sdr(layer_wise_sparse_sdrs[layer_ind]))
        trace(verbosity, req_level, '\n'.join(lines))

    def print_sa_superposition(
            self, verbosity: int, req_level: int, proximal_input: SparseSdr
    ):
        if req_level <= verbosity:
            sa_superposition = self.encoder.decode(proximal_input)
            trace(verbosity, req_level, self.format_sa_superposition(sa_superposition))

    def print_tm_state(self, verbosity: int, req_level: int,  label: str):
        if (self.output_file is None) or (verbosity < req_level):
            return

        self.writer.write(label)
        self.writer.save(self.output_file)

    def save_tm_state(self):
        """Saves TM state."""
        return pickle.dumps(self.tm)

    def restore_tm_state(self, tm_dump):
        """Restores saved TM state."""
        self.tm = pickle.loads(tm_dump)


