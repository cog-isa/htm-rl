import pickle
from typing import Tuple

import numpy as np
from htm import SDR

from htm_rl.agents.ucb.sparse_value_network import exp_decay, update_slice_lin_sum
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class RewardModel:
    rewards: np.ndarray
    learning_rate: Tuple[float, float]

    def __init__(self, shape, learning_rate: Tuple[float, float]):
        self.learning_rate = learning_rate
        self.rewards = np.zeros(shape, dtype=np.float)

    def update(self, s: SparseSdr, reward: float):
        update_slice_lin_sum(self.rewards, s, self.learning_rate[0], reward)

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)


class TransitionModel:
    tm: TemporalMemory

    _proximal_input_sdr: SDR    # cached SDR
    _tm_dump: bytes

    def __init__(
            self, tm: TemporalMemory, collect_anomalies: bool = False
    ):
        self.tm = tm
        self.anomalies = [] if collect_anomalies else None
        self._proximal_input_sdr = SDR(self.tm.n_columns)

    def reset_tracking(self):
        self.tm.reset()

    def process(self, proximal_input: SparseSdr, learn: bool) -> Tuple[SparseSdr, SparseSdr]:
        """
        Given new piece of proximal input data, processes it by sequentially activating cells
        and then depolarizes them, making prediction about next proximal input.

        :param proximal_input: sparse SDR of column activations.
        :param learn: whether or not to force learning.
        :return: tuple (active cells, depolarized cells) of sparse SDRs.
        """
        active_cells = self.activate_cells(proximal_input, learn)
        depolarized_cells = self.depolarize_cells(learn)
        return active_cells, depolarized_cells

    def activate_cells(self, proximal_input: SparseSdr, learn: bool) -> SparseSdr:
        """
        Given proximal input SDR, activates TM cells and [optionally] applies learning step.

        Learning is applied to already made predictions - to active segments of depolarized cells.

        :param proximal_input: sparse SDR of active columns.
        :param learn: whether or not to force learning.
        :return: sparse SDR of active cells.
        """
        self._proximal_input_sdr.sparse = proximal_input
        self.tm.compute(self._proximal_input_sdr, learn=learn)

        # if anomalies tracking enabled, append results from this step
        if self.anomalies is not None:
            self.anomalies.append(self.tm.anomaly)

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

    # def active_segments(self, active_cells: SparseSdr) -> Dict[int, List[Set[int]]]:
    #     """
    #     Gets all active segments.
    #
    #     Returns active segments as a dictionary: depolarized cell -> list of active segments.
    #
    #     Each active segment is a sparse SDR of active presynaptic cells, which induced
    #     the segment's activation. They stored as a `set`.
    #
    #     :param active_cells: sparse SDR of active cells.
    #     :return: active segments
    #     """
    #     tm, connections = self.tm, self.tm.connections
    #     active_cells = frozenset(active_cells)
    #
    #     def get_presynaptic_cells_for_segment(segment):
    #         # take only _connected_ synapses
    #         return frozenset(
    #             connections.presynapticCellForSynapse(synapse)
    #             for synapse in connections.synapsesForSegment(segment)
    #             if connections.permanenceForSynapse(synapse) >= tm.connected_permanence
    #         )
    #
    #     # active segment: (postsynaptic _depolarized_ cell; presynaptic !connected! cells)
    #     all_active_segments = (
    #         (connections.cellForSegment(segment), get_presynaptic_cells_for_segment(segment))
    #         for segment in tm.getActiveSegments()
    #     )
    #     # filter out synapses to _inactive_ cells, keeping only _active_ presynaptic cells
    #     active_segments = defaultdict(list)
    #     for depolarized_cell, active_segment in all_active_segments:
    #         # keep only synapses to active presynaptic cells
    #         active_presynaptic_cells = active_segment & active_cells
    #
    #         if len(active_presynaptic_cells) >= tm.activation_threshold:
    #             active_segments[depolarized_cell].append(active_presynaptic_cells)
    #
    #     # active segment: (postsynaptic _depolarized_ cell; presynaptic !active+connected! cells)
    #     return active_segments
    #
    # def print_cells(self, verbosity: int, req_level: int, cells_sparse_sdr: SparseSdr, mark: str = ''):
    #     """
    #     Prints cells sparse SDR layer by layer, each one on a separate line.
    #     :param verbosity: accepted verbosity level to print debug traces.
    #     :param req_level: required minimal verbosity level defined by caller.
    #     :param cells_sparse_sdr:
    #     :param mark: optional description mark to print near the first line
    #     """
    #     if req_level > verbosity:
    #         return
    #
    #     cpc = self.tm.cells_per_column
    #
    #     layer_wise_sparse_sdrs = [[] for _ in range(cpc)]
    #     for cell_index in cells_sparse_sdr:
    #         column, layer = divmod(cell_index, cpc)
    #         layer_wise_sparse_sdrs[layer].append(column)
    #
    #     first_line = f'{self.format_sdr(layer_wise_sparse_sdrs[0])} {mark}'
    #     lines = [first_line]
    #     for layer_ind in range(1, cpc):
    #         lines.append(self.format_sdr(layer_wise_sparse_sdrs[layer_ind]))
    #     trace(verbosity, req_level, '\n'.join(lines))
    #
    # def print_sa_superposition(
    #         self, verbosity: int, req_level: int, proximal_input: SparseSdr
    # ):
    #     if req_level <= verbosity:
    #         sa_superposition = self.encoder.decode(proximal_input)
    #         trace(verbosity, req_level, self.format_sa_superposition(sa_superposition))

    def save_tm_state(self):
        """Saves TM state."""
        self._tm_dump = pickle.dumps(self.tm)
        return self._tm_dump

    def restore_tm_state(self, tm_dump=None):
        """Restores saved TM state."""
        tm_dump = isnone(tm_dump, self._tm_dump)
        self.tm = pickle.loads(tm_dump)
