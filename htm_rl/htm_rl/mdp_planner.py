from typing import List, Tuple, Dict

import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM, Connections

from utils import isnone, sparse_to_dense


class SarEncoder:
    @staticmethod
    def encode(sar: Tuple[int, int, int]):
        s, a, r = sar
        return r * 100 + s * 10 + a

    @staticmethod
    def decode(x: int) -> Tuple[int, int, int]:
        r, sa = divmod(x, 100)
        s, a = divmod(sa, 10)
        return s, a, r

    @staticmethod
    def encode_arr(sars):
        return [SarEncoder.encode(sar) for sar in sars]

    @staticmethod
    def decode_arr(xs):
        return [SarEncoder.decode(x) for x in xs]


class DataEncoder:
    name: str
    n_values: int
    value_bits: int
    total_bits: int
    activation_threshold: int

    def __init__(self, name: str, n_vals: int, value_bits: int, activation_threshold: int = None):
        self.name = name
        self.value_bits = value_bits
        self.n_values = n_vals
        self.total_bits = n_vals * value_bits
        self.activation_threshold = isnone(activation_threshold, self.value_bits)

    def encode_sparse(self, x: int) -> List[int]:
        ind_from, ind_to = self._get_encode_range(x)
        return list(range(ind_from, ind_to))

    def encode_dense(self, x: int, arr: np.array = None) -> np.ndarray:
        ind_from, ind_to = self._get_encode_range(x)
        if arr is None:
            arr = np.zeros(self.total_bits, dtype=np.int8)
        arr[ind_from:ind_to] = 1
        return arr

    def _get_encode_range(self, x: int) -> Tuple[int, int]:
        assert x < self.n_values, f'Value must be in [0, {self.n_values}]; got {x}'
        ind_from = x * self.value_bits
        ind_to = ind_from + self.value_bits
        return ind_from, ind_to

    def str_from_dense(self, arr: np.ndarray):
        assert arr.ndim == 1, f'Expected 1-dim array; got {arr.ndim}'
        bit_packs_by_value = [
            ''.join(map(str, arr[i: i + self.value_bits]))
            for i in range(0, self.total_bits, self.value_bits)
        ]
        return ' '.join(bit_packs_by_value)

    def decode_dense(self, arr: np.ndarray) -> List[int]:
        decoded_values = []
        for i in range(0, self.total_bits, self.value_bits):
            s = np.count_nonzero(arr[i: i + self.value_bits])
            if s >= self.activation_threshold:
                decoded_values.append(i // self.value_bits)
        return decoded_values

    def decode_sparse(self, indices: List[int]) -> List[int]:
        activations = [0]*self.n_values
        for x in indices:
            activations[x // self.value_bits] += 1
        decoded_values = [i for i, act in enumerate(activations) if act >= self.activation_threshold]
        return decoded_values

    def __str__(self):
        name, n_values, value_bits = self.name, self.n_values, self.value_bits
        return f'DataEncoder("{name}", v{n_values} x b{value_bits})'


class DataMultiEncoder:
    encoders: Tuple[DataEncoder]
    dict: Dict[str, DataEncoder]
    total_bits: int
    macro_column_bits: int

    def __init__(self, encoders: Tuple[DataEncoder, ...]):
        self.encoders = encoders
        self.dict = {e.name: e for e in encoders}
        self.total_bits = sum(e.total_bits for e in encoders)
        self.macro_column_bits = sum(e.value_bits for e in encoders)

    def encode_dense(self, values: Tuple[int, ...]) -> np.array:
        assert len(values) == len(self.encoders)
        base_shift = 0
        encoded_vector = np.zeros(self.total_bits, dtype=np.int8)
        for x, encoder in zip(values, self.encoders):
            encoder.encode_dense(x, encoded_vector[base_shift : base_shift + encoder.total_bits])
            base_shift += encoder.total_bits
        return encoded_vector

    def encode_sparse(self, values: Tuple[int, ...]) -> List[int]:
        assert len(values) == len(self.encoders)
        base_shift = 0
        encoded_indices = []
        for x, encoder in zip(values, self.encoders):
            indices = encoder.encode_sparse(x)
            indices = self._apply_shift(indices, base_shift)

            encoded_indices.extend(indices)
            base_shift += encoder.total_bits
        return encoded_indices

    def decode_dense(self, arr: np.ndarray) -> List[Tuple[int, ...]]:
        base_shift = 0
        decoded_values = []
        for encoder in self.encoders:
            values = encoder.decode_dense(arr[base_shift: base_shift + encoder.total_bits])
            decoded_values.append(values)
            base_shift += encoder.total_bits

        return list(zip(*decoded_values))

    def decode_sparse(self, indices: List[int]) -> List[Tuple[int, ...]]:
        return self.decode_dense(sparse_to_dense(indices, self.total_bits))

    def str_from_dense(self, arr: np.ndarray) -> str:
        assert arr.ndim == 1 and arr.shape[0] == self.total_bits, \
            f'Array shape mismatch. Expected ({self.total_bits},); got {arr.shape}'
        base_shift = 0
        substrings = []
        for encoder in self.encoders:
            substrings.append(
                encoder.str_from_dense(arr[base_shift: base_shift + encoder.total_bits])
            )
            base_shift += encoder.total_bits
        return ' '.join(substrings)

    @staticmethod
    def _apply_shift(indices: List[int], shift: int):
        return [i + shift for i in indices]


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


class HtmAgent:
    tm: TemporalMemory
    encoder: DataMultiEncoder
    cached_input_sdr: SDR

    def __init__(self, tm: TemporalMemory, encoder: DataMultiEncoder):
        self.tm = tm
        self.encoder = encoder
        self.cached_input_sdr = None

    def train_cycle(self, sequence: List[np.array], print_enabled=False, learn=True, reset_enabled=True):
        if reset_enabled:
            self.tm.reset()

        x_sdr = self._get_or_create_cached_input_sdr()
        for x in sequence:
            x_sdr.dense = x
            self.tm.compute(x_sdr, learn=learn)
            if print_enabled:
                self.print_formatted_active_cells(self.tm.getActiveCells())

            self.tm.activateDendrites(learn)
            if print_enabled:
                self.print_formatted_predictive_cells(self.tm.getPredictiveCells())

    def train_one_step(self):
        ...

    def predict_cycle(self, start_from, n_steps, print_enabled=False, reset_enabled=True):
        if reset_enabled:
            self.tm.reset()

        x_sdr = self._get_or_create_cached_input_sdr()
        x_sdr.dense = start_from
        for i in range(n_steps):
            self.tm.compute(x_sdr, learn=False)
            if print_enabled:
                self.print_formatted_active_cells(self.tm.getActiveCells())

            self.tm.activateDendrites(False)
            prediction = self.tm.getPredictiveCells()
            if print_enabled:
                self.print_formatted_predictive_cells(prediction)

            next_x = self._active_columns_from_cells(prediction)
            print(SarEncoder.encode_arr(self.encoder.decode_dense(next_x)))
            x_sdr.dense = next_x

    def print_formatted_active_cells(self, active_cells_sdr: SDR):
        print(self._str_from_cells(active_cells_sdr, 'Active'))

    def print_formatted_predictive_cells(self, predictive_cells_sdr: SDR):
        print(self._str_from_cells(predictive_cells_sdr, 'Predictive'))

    def _str_from_cells(self, arr_sdr: SDR, name: str) -> str:
        arr: np.ndarray = arr_sdr.dense
        if arr.ndim == 1:
            arr = arr[np.newaxis]

        first_line = f'{self.encoder.str_from_dense(arr[:, 0])} {name}'
        substrings = [first_line]
        for layer_ind in range(1, self.tm.cells_per_column):
            substrings.append(self.encoder.str_from_dense(arr[:, layer_ind]))
        return '\n'.join(substrings)

    def _get_or_create_cached_input_sdr(self):
        if self.cached_input_sdr is None:
            self.cached_input_sdr = SDR(self.tm.n_columns)
        return self.cached_input_sdr

    def _active_columns_from_cells(self, cells_sdr) -> np.ndarray:
        cells: np.ndarray = cells_sdr.dense
        column_indices = cells.any(axis=1).nonzero()[0]

        return sparse_to_dense(column_indices, self.tm.n_columns)

    def _columns_from_cells_sparse(self, cells_sdr: SDR):
        # TODO: consider removing list(sorted())
        return list(sorted(set(self.tm.columnForCell(i) for i in cells_sdr.sparse)))

    def _columns_from_cells_dense(self, cells_sdr: SDR):
        dense_vec: np.ndarray = cells_sdr.dense
        return dense_vec.any(axis=1).nonzero().tolist()


#
# for cycle in range(len(seqs) * 40):
#     seq = seqs[np.random.choice(len(seqs))]
#     train_cycle(seq, tm, input_sdr, print_enabled=False, learn=True, reset_enabled=True)
#
#
# def get_columns_from_cells(cells_sdr, tm):
#     return list(sorted(set(tm.columnForCell(i) for i in cells_sdr.sparse)))
#
#
# def get_vals_from_cells(cols):
#     return list(sorted(set(col // col_size for col in cols)))
#
#
# def plan_to_val(start_val, end_val, max_steps, tm, input_sdr, print_enabled=False, reset_enabled=True):
#     if reset_enabled:
#         tm.reset()
#     val = start_val
#     end_sdr = SDR(input_size)
#     end_sdr.dense = encode_sdr_val(end_val)
#     connection_graph = []
#
#     input_sdr.dense = encode_sdr_val(val)
#     for i in range(max_steps):
#         tm.compute(input_sdr, learn=False)
#         active_cells = tm.getActiveCells()
#         if print_enabled:
#             print_formatted_active_cells(' ', active_cells)
#
#         overlap_with_end_val = end_sdr.getOverlap(input_sdr)
#         print(overlap_with_end_val)
#         if overlap_with_end_val >= activationThreshold:
#             break
#
#         tm.activateDendrites(False)
#         prediction = tm.getPredictiveCells()
#         if print_enabled:
#             print_formatted_predictive_cells(tm.anomaly, prediction)
#
#         # append layer to graph
#         active_segments = tm.getActiveSegments()
#         print(f'segments: {active_segments}')
#         cells_for_segments = [tm.connections.cellForSegment(segment) for segment in active_segments]
#         print(f'cells_se: {cells_for_segments}')
#         synapses = [
#             (
#             tm.connections.cellForSegment(segment), tm.connections.presynapticCellForSynapse(synapse), segment, synapse,
#             tm.connections.permanenceForSynapse(synapse))
#             for segment in active_segments
#             for synapse in tm.connections.synapsesForSegment(segment)
#             if tm.connections.permanenceForSynapse(synapse) >= connectedPermanence
#         ]
#         synapses_for_cells = defaultdict(list)
#         for synapse in synapses:
#             cell = synapse[0]
#             synapses_for_cells[cell].append(synapse)
#
#         connection_graph.append(synapses_for_cells)
#         # -----------------
#
#         next_activation_sparse = get_columns_from_cells(prediction, tm)
#         input_sdr.sparse = next_activation_sparse
#         print(get_vals_from_cells(get_columns_from_cells(prediction, tm)))
#
#     if overlap_with_end_val < activationThreshold:
#         return
#
#     end_columns = set(end_sdr.sparse)
#     active_cells = [
#         cell
#         for cell in active_cells.flatten().sparse
#         if tm.columnForCell(cell) in end_columns
#     ]
#     print(connection_graph[-1].keys())
#     backtracking_graph = [dict() for _ in connection_graph]
#
#     for t in range(len(connection_graph) - 1, -1, -1):
#         backtracking_graph[t] = {
#             cell: connection_graph[t][cell]
#             for cell in active_cells
#         }
#
#         print('===')
#
#         presynaptic_cells = {
#             connection[1]
#             for cell in active_cells
#             for connection in connection_graph[t][cell]
#         }
#         active_cells = list(presynaptic_cells)
#         print('---------------------')
#         print(active_cells)

