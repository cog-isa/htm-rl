import pickle
from collections import defaultdict
from typing import List, Tuple, Mapping, Set

import numpy as np

from htm_rl.agent import Agent
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.sdr import SparseSdr
from htm_rl.utils import range_reverse
from htm_rl.representations.sar import BaseSar as Sar


class Planner:
    agent: Agent
    max_steps: int
    print_enabled: bool

    # segments[time][cell] = segments = [ segment ] = [ [presynaptic_cell] ]
    _active_segments_timeline: List[Mapping[int, List[Set[int]]]]

    def __init__(self, agent: Agent, max_steps: int, print_enabled: bool = False):
        self.agent = agent
        self.max_steps = max_steps
        self.print_enabled = print_enabled
        self._active_segments_timeline = []

    def plan_actions(self, initial_sar: Sar):
        self._save_initial_tm_state()
        planning_succeeded = False

        if self.print_enabled:
            print()
            print('======> Planning')

        reward_reached = self._make_predictions_to_reward(initial_sar)
        if not reward_reached:
            return

        for backtrack in self._yield_successful_backtracks_from_reward():
            print('OK')
            # planned_actions = self._check_backtrack_correctly_predicts_reward(
            #     initial_sar, starting_step_allowed_actions
            # )
            # return planned_actions

        if self.print_enabled:
            print('<====== Planning complete')

    def _make_predictions_to_reward(self, initial_sar: Sar) -> bool:
        if self.print_enabled:
            print('===> Forward pass')

        all_actions_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)
        proximal_input = self.agent.encoder.encode(all_actions_sar)
        reward_reached = False
        self._active_segments_timeline = []

        for i in range(self.max_steps):
            reward_reached = self._check_reward(proximal_input)
            if reward_reached:
                break

            active_cells = self._activate_cells(proximal_input)
            self._depolarize_cells()

            # append active segments at time t = {depolarized cell: [ active segment ]}
            active_segments_t = self.agent.get_active_segments(active_cells)
            self._active_segments_timeline.append(active_segments_t)

            depolarized_cells = active_segments_t.keys()
            proximal_input = self.agent.columns_from_cells_sparse(depolarized_cells)

        self._restore_initial_tm_state()

        if self.print_enabled:
            if reward_reached:
                T = len(self._active_segments_timeline)
                print(f'<=== Predicting reward in {T} steps')
            else:
                print(f'<=== Predicting NO reward in {self.max_steps} steps')
            print()
        return reward_reached

    def _activate_cells(self, proximal_input: SparseSdr) -> SparseSdr:
        return self.agent.activate_memory(
            proximal_input, learn_enabled=False,
            output_active_cells=True, print_enabled=self.print_enabled
        )

    def _depolarize_cells(self) -> SparseSdr:
        return self.agent.depolarize_memory(
            learn_enabled=False, output_predictive_cells=True, print_enabled=self.print_enabled
        )

    def _check_reward(self, sparse_sdr: SparseSdr) -> bool:
        sar_superposition = self.agent.encoder.decode(sparse_sdr)
        if self.print_enabled:
            print(self.agent.format(sar_superposition))
        return sar_superposition.reward is not None and 1 in sar_superposition.reward

    def _yield_successful_backtracks_from_reward(self):
        if self.print_enabled:
            print('===> Backward pass')

        T = len(self._active_segments_timeline)
        for rewarding_segment in self._yield_rewarding_segments():
            # rewarding segment:
            #   - one of the active segments at time T-1
            #   - consists of a set of presynaptic [active at time T-1] cells
            #   - these active cells induce reward prediction [happening at time T]

            # start backtracking from time T-2:
            # when these presynaptic active cells were just a prediction (= depolarized)
            depolarized_cells = rewarding_segment
            backtracking_succeeded, segments = self._backtrack(depolarized_cells, T-2)
            if backtracking_succeeded:
                segments.append(rewarding_segment)
                yield segments

        if self.print_enabled:
            print('<=== Backward pass complete')

    def _yield_rewarding_segments(self):
        """TODO"""
        T = len(self._active_segments_timeline)

        # Take active_segments for time T-1, when the reward was found.
        depolarized_reward_cells = self._get_depolarized_reward_cells()
        self.agent.print_cells(depolarized_reward_cells, 'Reward == 1')

        # all segments, whose presynaptic cells can potentially induce desired reward == 1 prediction
        unique_potentially_rewarding_segments = self._get_unique_active_segments(
            depolarized_cells=depolarized_reward_cells,
            t=T-1
        )

        for potential_segment in unique_potentially_rewarding_segments:
            # imagine that they're active cells => how many "should be depolarized" cells they depolarize?
            n_depolarized_cells = self._count_induced_depolarization(
                active_cells=potential_segment,
                could_be_depolarized_cells=depolarized_reward_cells,
                t=T-1
            )

            print('>')
            self.agent.print_cells(
                potential_segment,
                f'n: {n_depolarized_cells} of {self.agent.encoder._encoders.reward.activation_threshold}'
            )

            # if number of depolarized cells < reward activation threshold
            #   ==> reward == 1 is not predicted
            if n_depolarized_cells >= self.agent.encoder._encoders.reward.activation_threshold:
                yield potential_segment
            print('<')

    def _backtrack(self, should_be_depolarized_cells, t):
        # goal:
        #   - find any active_segment whose presynaptic active cells induce depolarization we need
        #   - recursively check if we can backtrack from this segment

        if t < 0:
            return True, []

        # obviously, we should look only among active segments of these "should-be-depolarized" cells
        unique_potential_segments = self._get_unique_active_segments(should_be_depolarized_cells, t)
        print('candidates:')
        for potential_segment in unique_potential_segments:
            self.agent.print_cells(potential_segment)
        print()

        # check every potential segment
        for potential_segment in unique_potential_segments:
            # check depolarization induced by segment's presynaptic cells
            #   i.e. how many "should be depolarized" cells are in fact depolarized
            presynaptic_active_cells = potential_segment
            n_depolarized_cells = self._count_induced_depolarization(
                presynaptic_active_cells, should_be_depolarized_cells, t
            )

            self.agent.print_cells(
                presynaptic_active_cells,
                f'n: {n_depolarized_cells} of {self.agent.tm.activation_threshold}'
            )

            # if number of depolarized cells < threshold ==> they can't depolarize t+1 segment
            if n_depolarized_cells >= self.agent.tm.activation_threshold:
                backtracking_succeeded, activation_timeline = self._backtrack(
                    should_be_depolarized_cells=presynaptic_active_cells,
                    t=t - 1
                )
                if backtracking_succeeded:
                    activation_timeline.append(presynaptic_active_cells)
                    return True, activation_timeline

        return False, None

    def _get_unique_active_segments(self, depolarized_cells, t):
        """TODO"""
        active_segments_t = self._active_segments_timeline[t]
        return {
            cell_active_segment
            for cell in depolarized_cells
            for cell_active_segment in active_segments_t[cell]
        }

    def _count_induced_depolarization(self, active_cells, could_be_depolarized_cells, t):
        """TODO"""
        active_segments_t = self._active_segments_timeline[t]
        n_depolarized_cells = 0

        # look only among specified "could-be-depolarized" cells
        for could_be_depolarized_cell in could_be_depolarized_cells:
            cell_segments = active_segments_t[could_be_depolarized_cell]

            # for segment in cell_segments:
            #     self.agent.print_cells(segment, f' {could_be_depolarized_cell:2} presynaptics')

            any_cell_segments_activated = any(
                len(segment & active_cells) >= self.agent.tm.activation_threshold
                for segment in cell_segments
            )
            # the cell becomes depolarized if any of its segments becomes active
            if any_cell_segments_activated:
                n_depolarized_cells += 1

        return n_depolarized_cells

    def _get_depolarized_reward_cells(self) -> SparseSdr:
        # depolarized cells from the last step of forward prediction phase, when the reward was found
        all_final_depolarized_cells = list(self._active_segments_timeline[-1].keys())
        # tuple [l, r): range of columns related to reward == 1
        rewarding_columns_range = self.agent.encoder.get_rewarding_indices_range()

        depolarized_rewarding_cells = self._filter_cells_by_columns_range(
            all_final_depolarized_cells, rewarding_columns_range
        )
        return depolarized_rewarding_cells

    def _filter_cells_by_columns_range(
            self, cells: SparseSdr, columns_range: Tuple[int, int]
    ) -> SparseSdr:
        cpc = self.agent.tm.cells_per_column
        l, r = columns_range
        # to cells range
        l, r = l * cpc, r * cpc
        return [cell for cell in cells if l <= cell < r]

    def _backtrack_step_from_active_cells(
            self,
            active_cells: SparseSdr,
            t_step: int
    ):
        backtracked_presynaptic_connections = {
            cell: presynaptic_connections[cell]
            for cell in active_cells
        }
        presynaptic_cells = {
            presynaptic_cell
            for cell_segments in backtracked_presynaptic_connections.values()
            for segment in cell_segments
            for presynaptic_cell in segment
        }
        presynaptic_cells = list(presynaptic_cells)

        return backtracked_presynaptic_connections, presynaptic_cells

    def _print_active_cells_superposition(self, active_cells: SparseSdr):
        if self.print_enabled:
            active_columns = self.agent.columns_from_cells_sparse(active_cells)
            sar_superposition = self.agent.encoder.decode(active_columns)
            print(self.agent.format(sar_superposition))

    def _check_backtrack_correctly_predicts_reward(
            self, initial_sar: Sar, starting_step_allowed_actions: List[int]
    ) -> List[int]:
        if self.print_enabled:
            print()
            print('Forward pass #2')

        reward_reached = False
        allowed_actions = starting_step_allowed_actions
        proximal_input = self.agent.encoder.encode(initial_sar)
        n_steps = len(self._active_segments_timeline)
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
                active_cells, self._active_segments_timeline[i]
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
            for postsynaptic_cell, cell_segments in backtracking_connections.items()
            for segment in cell_segments
            if (
                    postsynaptic_cell in postsynaptic_action_cells_set
                    and any(cell for cell in segment if cell in presynaptic_action_cells_set)
            )
        ]
        return allowed_action_cells

    def _filter_action_cells(self, cells: SparseSdr) -> SparseSdr:
        actions_columns_range = self.agent.encoder.get_actions_indices_range()
        return self._filter_cells_by_columns_range(cells, actions_columns_range)

    def _replace_actions_with_action(self, columns: SparseSdr, action: int):
        action_only_sar = Sar(state=None, action=action, reward=None)
        action_sdr = self.agent.encoder.encode(action_only_sar)

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

    def _save_initial_tm_state(self):
        self._initial_tm_state = pickle.dumps(self.agent.tm)

    def _restore_initial_tm_state(self):
        self.agent.tm = pickle.loads(self._initial_tm_state)

    def _get_active_actions(self, active_cells) -> List[int]:
        active_columns = self.agent.columns_from_cells_sparse(active_cells)
        sar_superposition = self.agent.encoder.decode(active_columns)
        return sar_superposition.action
