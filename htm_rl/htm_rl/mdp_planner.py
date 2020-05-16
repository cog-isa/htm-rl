from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM, Connections

from utils import isnone, sparse_to_dense


class SarEncoder:
    @staticmethod
    def encode(sar: Tuple[int, int, int]):
        s, a, r = sar
        return s * 100 + a * 10 + r

    @staticmethod
    def decode(x: int) -> Tuple[int, int, int]:
        s, ar = divmod(x, 100)
        a, r = divmod(ar, 10)
        return s, a, r

    @staticmethod
    def encode_arr(sars):
        return [SarEncoder.encode(sar) for sar in sars]

    @staticmethod
    def decode_arr(xs):
        return [SarEncoder.decode(x) for x in xs]

    @staticmethod
    def str_from_superposition(sars: List[List[int]]) -> str:
        return ' '.join(''.join(map(str, arr)) for arr in sars)

    @staticmethod
    def has_reward(sars: List[List[int]]) -> bool:
        return 1 in sars[2]


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

    def encode_dense(self, x: int, arr: np.ndarray = None) -> np.ndarray:
        ind_from, ind_to = self._get_encode_range(x)
        if arr is None:
            arr = np.zeros(self.total_bits, dtype=np.int8)
        arr[ind_from:ind_to] = 1
        return arr

    def _get_encode_range(self, x: int) -> Tuple[int, int]:
        if x is None:
            return 0, 0

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
    encoders: Tuple[DataEncoder, ...]
    dict: Dict[str, DataEncoder]
    total_bits: int
    macro_column_bits: int

    def __init__(self, encoders: Tuple[DataEncoder, ...]):
        self.encoders = encoders
        self.dict = {e.name: e for e in encoders}
        self.total_bits = sum(e.total_bits for e in encoders)
        self.macro_column_bits = sum(e.value_bits for e in encoders)

    def encode_dense(self, values: Tuple[int, ...]) -> np.ndarray:
        assert len(values) == len(self.encoders)
        base_shift = 0
        encoded_vector = np.zeros(self.total_bits, dtype=np.int8)
        for x, encoder in zip(values, self.encoders):
            encoder.encode_dense(x, encoded_vector[base_shift : base_shift + encoder.total_bits])
            base_shift += encoder.total_bits
        return encoded_vector

    def update_encoding_dense(self, encoded_vector, values: Tuple[int, ...]) -> np.ndarray:
        base_shift = 0
        for x, encoder in zip(values, self.encoders):
            if x is not None:
                l, r = base_shift, base_shift + encoder.total_bits
                encoded_vector[l:r] = 0
                encoder.encode_dense(x, encoded_vector[l:r])
            base_shift += encoder.total_bits
        return encoded_vector

    def encode_dense_state_all_actions(self, state: int, reward: int = 0) -> np.ndarray:
        encoded_vector = np.zeros(self.total_bits, dtype=np.int8)
        state_encoder, action_encoder, reward_encoder = self.encoders

        state_encoder.encode_dense(state, encoded_vector[0: state_encoder.total_bits])

        base_shift = state_encoder.total_bits
        encoded_vector[base_shift: base_shift + action_encoder.total_bits] = 1

        base_shift += action_encoder.total_bits
        reward_encoder.encode_dense(reward, encoded_vector[base_shift: base_shift + reward_encoder.total_bits])
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

    def decode_dense(self, arr: np.ndarray) -> List[List[int]]:
        base_shift = 0
        decoded_values = []
        for encoder in self.encoders:
            values = encoder.decode_dense(arr[base_shift: base_shift + encoder.total_bits])
            decoded_values.append(values)
            base_shift += encoder.total_bits

        return decoded_values

    def decode_sparse(self, indices: List[int]) -> List[List[int]]:
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

    def train_cycle(self, sequence: List[np.ndarray], print_enabled=False, learn=True, reset_enabled=True):
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

    def predict_cycle(self, start_value: np.ndarray, n_steps, print_enabled=False, reset_enabled=True):
        if reset_enabled:
            self.tm.reset()

        x_sdr = self._get_or_create_cached_input_sdr()
        x_sdr.dense = start_value
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

    def _columns_from_cells_sparse(self, cell_indices: List[int]):
        # TODO: consider removing list(sorted())
        return list(sorted(set(self.tm.columnForCell(i) for i in cell_indices)))

    def _columns_from_cells_dense(self, cells: np.ndarray):
        return cells.any(axis=1).nonzero().tolist()

    def _action_cells_dense(self, cells: np.ndarray) -> List[int]:
        state_encoder = self.encoder.encoders[0]
        action_encoder = self.encoder.encoders[1]
        cpc = self.tm.cells_per_column
        l = state_encoder.total_bits * cpc
        r = l + action_encoder.total_bits * cpc
        action_cells = cells.ravel()[l:r]
        indices = action_cells.nonzero()[0] + l
        return indices.tolist()

    def _action_cells_sparse(self, indices: List[int]) -> List[int]:
        state_encoder = self.encoder.encoders[0]
        action_encoder = self.encoder.encoders[1]
        cpc = self.tm.cells_per_column
        l = state_encoder.total_bits * cpc
        r = l + action_encoder.total_bits * cpc
        return [cell for cell in indices if l <= cell < r]

    def plan_to_value(
            self, start_value: np.ndarray, max_steps: int,
            print_enabled=False, reset_enabled=True
    ):
        if reset_enabled:
            self.tm.reset()

        connection_graph = []
        reward_reached = False
        x_sdr = self._get_or_create_cached_input_sdr()
        x_sdr.dense = start_value
        for i in range(max_steps):
            self.tm.compute(x_sdr, learn=False)
            active_cells = self.tm.getActiveCells()
            if print_enabled:
                self.print_formatted_active_cells(active_cells)

            active_columns = self._active_columns_from_cells(active_cells)
            value_superposition = self.encoder.decode_dense(active_columns)
            print(SarEncoder.str_from_superposition(value_superposition))
            if SarEncoder.has_reward(value_superposition):
                reward_reached = True
                break

            self.tm.activateDendrites(False)
            prediction = self.tm.getPredictiveCells()
            if print_enabled:
                self.print_formatted_predictive_cells(prediction)

            # append layer to graph
            active_cells_set = set(active_cells.flatten().sparse)
            active_segments = self.tm.getActiveSegments()
            # print(f'segments: {active_segments}')
            # cells_for_segments = [self.tm.connections.cellForSegment(segment) for segment in active_segments]
            # print(f'cells_se: {cells_for_segments}')
            synapses = [
                (
                    self.tm.connections.cellForSegment(segment),
                    self.tm.connections.presynapticCellForSynapse(synapse),
                    # segment, synapse,
                    # self.tm.connections.permanenceForSynapse(synapse)
                )
                for segment in active_segments
                for synapse in self.tm.connections.synapsesForSegment(segment)
                if self.tm.connections.permanenceForSynapse(synapse) >= self.tm.connected_permanence
            ]
            synapses = [synapse for synapse in synapses if synapse[1] in active_cells_set]
            print(len(synapses))

            synapses_for_cells = defaultdict(list)
            for synapse in synapses:
                cell = synapse[0]
                synapses_for_cells[cell].append(synapse)

            connection_graph.append(synapses_for_cells)
            # -----------------

            next_x = self._active_columns_from_cells(prediction)
            x_sdr.dense = next_x
            print()

        if not reward_reached:
            return

        print()
        print('Backward pass:')
        reward_columns = set(self.encoder.encode_sparse((None, None, 1)))
        active_cells = [
            cell
            for cell in active_cells.flatten().sparse
            if self.tm.columnForCell(cell) in reward_columns
        ]
        backtracking_graph = []
        active_cells_timeline = []
        actions = []

        for connections in connection_graph[::-1]:
            backtracking_graph.append({
                cell: connections[cell]
                for cell in active_cells
            })

            presynaptic_cells = {
                connection[1]
                for cell in active_cells
                for connection in connections[cell]
            }
            active_cells_timeline.append(presynaptic_cells)
            active_cells = list(presynaptic_cells)
            print('---------------------')

            active_column_indices = self._columns_from_cells_sparse(active_cells)
            value_superposition = self.encoder.decode_sparse(active_column_indices)
            actions.append(value_superposition[1])
            print(SarEncoder.str_from_superposition(value_superposition))

        backtracking_graph = backtracking_graph[::-1]
        active_cells_timeline = active_cells_timeline[::-1]

        print()
        print('Forward pass #2')

        # TODO: if reset_enabled=False, are we fucked up?
        if reset_enabled:
            self.tm.reset()

        actions = actions[::-1]
        print(actions)
        allowed_actions = actions[0]

        reward_reached = False
        actions = []
        x = start_value.copy()
        for i in range(max_steps):
            # get actions
            # value_superposition = self.encoder.decode_dense(x)
            action = np.random.choice(allowed_actions)
            print(allowed_actions, action)
            actions.append(action)

            x = self.encoder.update_encoding_dense(x, (None, action, None))
            x_sdr.dense = x

            self.tm.compute(x_sdr, learn=False)
            active_cells = self.tm.getActiveCells()
            if print_enabled:
                self.print_formatted_active_cells(active_cells)

            active_columns = self._active_columns_from_cells(active_cells)
            value_superposition = self.encoder.decode_dense(active_columns)
            print(SarEncoder.str_from_superposition(value_superposition))

            self.tm.activateDendrites(False)
            prediction = self.tm.getPredictiveCells()
            if print_enabled:
                self.print_formatted_predictive_cells(prediction)
            predictive_columns = self._active_columns_from_cells(prediction)
            predictive_value_superposition = self.encoder.decode_dense(predictive_columns)
            print(SarEncoder.str_from_superposition(predictive_value_superposition))
            if SarEncoder.has_reward(predictive_value_superposition):
                reward_reached = True
                break

            backtracking_prediction_cells = list(backtracking_graph[i].keys())

            presynaptic_action_cells_set = set(self._action_cells_dense(active_cells.dense))
            postsynaptic_action_cells_set = set(self._action_cells_sparse(backtracking_prediction_cells))

            # print(presynaptic_action_cells_set)
            allowed_action_cells = [
                cell
                for cell, connections in backtracking_graph[i].items()
                if cell in postsynaptic_action_cells_set and \
                   len([cn[1] for cn in connections if cn[1] in presynaptic_action_cells_set]) > 0
            ]
            allowed_actions = self.encoder.decode_sparse(
                self._columns_from_cells_sparse(allowed_action_cells)
            )[1]

            x = self._active_columns_from_cells(prediction)
            print()

        if reward_reached:
            print(f'OK: {actions}')
        else:
            print('FAIL')