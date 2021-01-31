from collections import deque
from typing import List, Mapping, Set, Tuple, Deque, Dict

from htm_rl.agent.memory import Memory
from htm_rl.common.base_sa import Sa
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import range_reverse, trace


# noinspection PyPep8Naming


class MctsPlanner:
    memory: Memory
    planning_horizon: int
    encoder: SaSdrEncoder

    # segments[time][cell] = segments = [ segment ] = [ [presynaptic_cell] ]
    _n_actions: int
    _clusters_merging_threshold: int

    def __init__(self, memory: Memory, n_actions: int, planning_horizon: int):
        self.memory = memory
        self.encoder = memory.encoder
        self.planning_horizon = planning_horizon
        self._n_actions = n_actions

        # TODO: WARN! Handcrafted threshold
        self._clusters_merging_threshold = self.memory.tm.activation_threshold - 2

    def predict_states(self, initial_sa: Sa, verbosity: int):
        predicted_states = []
        if self.planning_horizon == 0:
            return predicted_states

        trace(verbosity, 2, '\n======> Planning')

        active_segments = self._predict_one_step_forward(initial_sa, verbosity)
        action_outcomes = self._get_predicted_state_for_each_action(active_segments, verbosity)

        trace(verbosity, 2, '<====== Planning complete')
        return action_outcomes

    def _predict_one_step_forward(self, initial_sa: Sa, verbosity: int) -> Dict[int, List[Set[int]]]:
        """Gets active segments for one all-actions-step prediction."""
        trace(verbosity, 2, '===> Predict one step forward')

        # saves initial TM state at the start of planning.
        initial_tm_state = self.memory.save_tm_state()

        # to consider all possible prediction paths, prediction is started with all possible actions
        all_actions_sa = Sa(initial_sa.state, IntSdrEncoder.ALL)
        proximal_input = self.encoder.encode(all_actions_sa)

        active_cells, depolarized_cells = self.memory.process(
            proximal_input, learn=False, verbosity=verbosity
        )

        active_segments = self.memory.active_segments(active_cells)
        self.memory.restore_tm_state(initial_tm_state)

        trace(verbosity, 2, '<=== Obtained active segments for prediction')
        return active_segments

    def _get_predicted_state_for_each_action(
            self, active_segments: Dict[int, List[Set[int]]], verbosity: int
    ) -> List[SparseSdr]:
        """
        Split prediction, represented as active segments, by actions, i.e. which
        state is predicted for which action. Returns list of state SparseSdr, each
        state for every action.
        """
        trace(verbosity, 2, '===> Split state superposition by previous action')

        action_outcomes: List[List[int]] = [[] for _ in range(self._n_actions)]
        state_cells = self.memory.filter_cells_by_columns_range(
            active_segments, self.encoder.states_indices_range()
        )

        for cell in state_cells:
            for presynaptic_cells in active_segments[cell]:
                presynaptic_columns = self.memory.columns_from_cells(presynaptic_cells)
                # which state-action pair activates `cell` in a form of superposition
                initial_sa_superposition = self.encoder.decode(presynaptic_columns)
                # trace(verbosity, 2, presynaptic_columns)
                trace(verbosity, 3, initial_sa_superposition)
                # iterating here is redundant - it should be always only one action
                for action in initial_sa_superposition.action:
                    action_outcomes[action].append(cell)

        trace(verbosity, 2, action_outcomes)

        trace(verbosity, 2, '<===')
        return action_outcomes
