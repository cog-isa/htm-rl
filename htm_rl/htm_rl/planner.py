import pickle
from typing import List, Tuple, Mapping

import numpy as np

from agent import Agent
from representations.int_sdr_encoder import IntSdrEncoder
from representations.sar import Sar, str_from_sar_superposition, sar_superposition_has_reward
from representations.sdr import SparseSdr
from utils import range_reverse


class Planner:
    agent: Agent
    max_steps: int
    print_enabled: bool
    _presynaptic_connections_graph: List[Mapping[int, List[int]]]

    def __init__(self, agent: Agent, max_steps: int, print_enabled: bool = False):
        self.agent = agent
        self.max_steps = max_steps
        self.print_enabled = print_enabled
        self._presynaptic_connections_graph = []

    def plan_actions(self, initial_sar: Sar):
        self._store_initial_tm_state()

        reward_reached = self._make_predictions_to_reward(initial_sar)
        if not reward_reached:
            return None

        print(self.agent.tm.n_columns)
        starting_step_allowed_actions = self._backtrack_from_reward()

        planned_actions = self._remake_predictions_to_plan_actions(initial_sar, starting_step_allowed_actions)
        return planned_actions

    def _make_predictions_to_reward(self, initial_sar: Sar) -> bool:
        all_actions_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)
        proximal_input = self.agent.encoder.encode_sparse(all_actions_sar)
        reward_reached = False

        for i in range(self.max_steps):
            reward_reached = self._check_reward(proximal_input)
            if reward_reached:
                break

            active_cells = self._activate_cells(proximal_input)
            predictive_cells = self._depolarize_cells()

            proximal_input = self.agent.columns_from_cells_sparse(predictive_cells)

            # append layer of {postsynaptic_cell: [presynaptic_cell]} to connections graph
            presynaptic_connections = self.agent.get_presynaptic_connections(active_cells)
            self._presynaptic_connections_graph.append(presynaptic_connections)

        self._restore_initial_tm_state()
        return reward_reached

    def _activate_cells(self, proximal_input: SparseSdr) -> SparseSdr:
        return self.agent.activate_memory(
            proximal_input, learn_enabled=False, output_active_cells=True, print_enabled=self.print_enabled
        )

    def _depolarize_cells(self) -> SparseSdr:
        return self.agent.depolarize_memory(
            learn_enabled=False, output_predictive_cells=True, print_enabled=self.print_enabled
        )

    def _check_reward(self, sparse_sdr: SparseSdr) -> bool:
        sar_superposition = self.agent.encoder.decode_sparse(sparse_sdr)
        if self.print_enabled:
            print(str_from_sar_superposition(sar_superposition))
        return sar_superposition_has_reward(sar_superposition)

    def _backtrack_from_reward(self) -> SparseSdr:
        if self.print_enabled:
            print()
            print('Backward pass:')

        rewarding_cells = self._get_rewarding_cells_as_initial_for_backtracking()
        active_cells = rewarding_cells

        for t in range_reverse(self._presynaptic_connections_graph):
            # get backtracked connections and presynaptic cells
            presynaptic_connections, presynaptic_cells = self._backtrack_step_from_active_cells(
                active_cells,
                self._presynaptic_connections_graph[t]
            )

            # replace forward pass graph layer with backtracked one
            self._presynaptic_connections_graph[t] = presynaptic_connections
            # propagate presynaptic cells to the previous timestep
            active_cells = presynaptic_cells

            self._print_active_cells_superposition(active_cells)

        starting_step_allowed_actions = self._get_active_actions(active_cells)
        return starting_step_allowed_actions

    def _get_rewarding_cells_as_initial_for_backtracking(self):
        # Take cells from the last step of forward prediction phase, when the reward was found.
        all_final_active_cells = list(self._presynaptic_connections_graph[-1].keys())
        rewarding_columns_range = self.agent.encoder.get_rewarding_indices_range()

        return self._filter_cells_by_columns_range(all_final_active_cells, rewarding_columns_range)

    def _filter_cells_by_columns_range(self, cells: SparseSdr, columns_range: Tuple[int, int]) -> SparseSdr:
        cpc = self.agent.tm.cells_per_column
        l, r = columns_range
        # to cells range
        l, r = l * cpc, r * cpc
        return [cell for cell in cells if l <= cell < r]

    @staticmethod
    def _backtrack_step_from_active_cells(
            active_cells: SparseSdr, presynaptic_connections: Mapping[int, List[int]]
    ) -> Tuple[Mapping[int, List[int]], List[int]]:
        backtracked_presynaptic_connections = {
            cell: presynaptic_connections[cell]
            for cell in active_cells
        }
        presynaptic_cells = {
            connection
            for connections in backtracked_presynaptic_connections.values()
            for connection in connections
        }
        presynaptic_cells = list(presynaptic_cells)

        return backtracked_presynaptic_connections, presynaptic_cells

    def _print_active_cells_superposition(self, active_cells: SparseSdr):
        if self.print_enabled:
            active_columns = self.agent.columns_from_cells_sparse(active_cells)
            sar_superposition = self.agent.encoder.decode_sparse(active_columns)
            print(str_from_sar_superposition(sar_superposition))

    def _remake_predictions_to_plan_actions(
            self, initial_sar: Sar, starting_step_allowed_actions: List[int]
    ) -> List[int]:
        if self.print_enabled:
            print()
            print('Forward pass #2')

        reward_reached = False
        allowed_actions = starting_step_allowed_actions
        proximal_input = self.agent.encoder.encode_sparse(initial_sar)
        n_steps = len(self._presynaptic_connections_graph)
        planned_actions = []

        for i in range(n_steps):
            # choose action
            action = self._choose_action(allowed_actions)
            planned_actions.append(action)

            proximal_input = self._replace_actions_with_action(proximal_input, action)

            active_cells = self._activate_cells(proximal_input)
            self._print_active_cells_superposition(active_cells)

            predictive_cells = self._depolarize_cells()
            proximal_input = self.agent.columns_from_cells_sparse(predictive_cells)

            reward_reached = self._check_reward(proximal_input)
            if reward_reached:
                break

            allowed_action_cells = self.ground_backtracking_predictions_with_active_presynaptic_cells(
                active_cells, self._presynaptic_connections_graph[i]
            )
            allowed_actions = self._get_active_actions(allowed_action_cells)
            if self.print_enabled:
                print()

        self._restore_initial_tm_state()

        # return all planned actions or None
        if reward_reached:
            print(f'OK: {planned_actions}')
            return planned_actions

        print('FAIL')

    def ground_backtracking_predictions_with_active_presynaptic_cells(
            self, active_presynaptic_cells, backtracking_connections
    ):
        backtracked_postsynaptic_cells = list(backtracking_connections.keys())

        presynaptic_action_cells_set = set(self._filter_action_cells(active_presynaptic_cells))
        postsynaptic_action_cells_set = set(self._filter_action_cells(backtracked_postsynaptic_cells))

        allowed_action_cells = [
            postsynaptic_cell
            for postsynaptic_cell, presynaptic_cells in backtracking_connections.items()
            if (
                    postsynaptic_cell in postsynaptic_action_cells_set
                    and any(cell for cell in presynaptic_cells if cell in presynaptic_action_cells_set)
            )
        ]
        return allowed_action_cells

    def _filter_action_cells(self, cells: SparseSdr) -> SparseSdr:
        actions_columns_range = self.agent.encoder.get_actions_indices_range()
        return self._filter_cells_by_columns_range(cells, actions_columns_range)

    def _replace_actions_with_action(self, columns: SparseSdr, action: int):
        action_only_sar = Sar(state=None, action=action, reward=None)
        action_sdr = self.agent.encoder.encode_sparse(action_only_sar)

        l, r = self.agent.encoder.get_actions_indices_range()
        filtered_columns = [
            column
            for column in columns
            if 0 <= column < l or r <= column
        ]

        filtered_columns.extend(action_sdr)
        return filtered_columns

    def _choose_action(self, allowed_actions: List[int]) -> int:
        action = np.random.choice(allowed_actions)
        if self.print_enabled:
            print(allowed_actions, action)
        return action

    def _store_initial_tm_state(self):
        self._initial_tm_state = pickle.dumps(self.agent.tm)

    def _restore_initial_tm_state(self):
        self.agent.tm = pickle.loads(self._initial_tm_state)

    def _get_active_actions(self, active_cells) -> List[int]:
        active_columns = self.agent.columns_from_cells_sparse(active_cells)
        sar_superposition = self.agent.encoder.decode_sparse(active_columns)
        return sar_superposition.actions
