import pickle
from typing import List, Mapping, Set

from htm_rl.agent.agent import Agent
from htm_rl.common.base_sar import Sar
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sdr import SparseSdr


# noinspection PyPep8Naming
class Planner:
    agent: Agent
    max_steps: int

    # segments[time][cell] = segments = [ segment ] = [ [presynaptic_cell] ]
    _active_segments_timeline: List[Mapping[int, List[Set[int]]]]

    def __init__(self, agent: Agent, max_steps: int):
        self.agent = agent
        self.max_steps = max_steps
        self._active_segments_timeline = []

    def plan_actions(self, initial_sar: Sar, trace: bool):
        self.print(trace, '\n======> Planning')

        # saves initial TM state at the start of planning.
        self._save_initial_tm_state()
        planned_actions = None

        reward_reached = self._make_predictions_to_reward(initial_sar, trace)
        if not reward_reached:
            return

        for activation_timeline in self._yield_successful_backtracks_from_reward(trace):
            planned_actions = self._check_backtrack_correctly_predicts_reward(
                initial_sar, activation_timeline, trace
            )
            if planned_actions is not None:
                break

        self.print(trace, '<====== Planning complete')
        return planned_actions

    def _make_predictions_to_reward(self, initial_sar: Sar, trace: bool) -> bool:
        self.print(trace, '===> Forward pass')

        # to consider all possible prediction paths, prediction is started with all possible actions
        all_actions_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)

        proximal_input = self.agent.encoder.encode(all_actions_sar)
        reward_reached = False
        active_segments_timeline = []

        for i in range(self.max_steps):
            reward_reached = self.agent.encoder.is_rewarding(proximal_input)
            if reward_reached:
                break

            active_cells, depolarized_cells = self.agent.process(proximal_input, learn=False, trace=trace)

            active_segments_t = self.agent.active_segments(active_cells)
            active_segments_timeline.append(active_segments_t)

            proximal_input = self.agent.columns_from_cells(depolarized_cells)
            self.print(trace, '')

        self._restore_initial_tm_state()
        self._active_segments_timeline = active_segments_timeline

        if reward_reached:
            T = len(active_segments_timeline)
            self.print(trace, f'<=== Predict reward in {T} steps')
        else:
            self.print(trace, f'<=== Predicting NO reward in {self.max_steps} steps')

        return reward_reached

    def _yield_successful_backtracks_from_reward(self, trace: bool):
        self.print(trace, '\n===> Backward pass')

        T = len(self._active_segments_timeline)
        for rewarding_segment in self._yield_rewarding_segments(trace):
            # rewarding segment:
            #   - one of the active segments at time T-1
            #   - consists of a set of presynaptic [active at time T-1] cells
            #   - these active cells induce reward prediction [happening at time T]

            # start backtracking from time T-2:
            # when these presynaptic active cells were just a prediction (= depolarized)
            depolarized_cells = rewarding_segment
            backtracking_succeeded, activation_timeline = self._backtrack(depolarized_cells, T-2, trace)
            if backtracking_succeeded:
                activation_timeline.append(rewarding_segment)
                yield activation_timeline

        self.print(trace, '<=== Backward pass complete')

    def _yield_rewarding_segments(self, trace: bool):
        """TODO"""
        T = len(self._active_segments_timeline)
        reward_activation_threshold = self.agent.encoder._encoders.reward.activation_threshold

        # Take active_segments for time T-1, when the reward was found.
        depolarized_reward_cells = self._get_depolarized_reward_cells()
        self.agent.print_cells(trace, depolarized_reward_cells, 'Reward == 1')

        # all segments, whose presynaptic cells can potentially induce desired reward == 1 prediction
        rewarding_segment_candidates = self._get_unique_active_segments(depolarized_reward_cells, T-1)
        for candidate_segment in rewarding_segment_candidates:
            # imagine that they're active cells => how many "should be depolarized" cells they depolarize?
            n_depolarized_cells = self._count_induced_depolarization(
                active_cells=candidate_segment,
                could_be_depolarized_cells=depolarized_reward_cells,
                t=T-1
            )

            print('>')
            self.agent.print_cells(
                trace, candidate_segment, f'n: {n_depolarized_cells} of {reward_activation_threshold}'
            )

            # if number of depolarized cells < reward activation threshold
            #   ==> reward == 1 is not predicted
            if n_depolarized_cells >= reward_activation_threshold:
                yield candidate_segment
            print('<')

    def _backtrack(self, should_be_depolarized_cells, t, trace):
        # goal:
        #   - find any active_segment whose presynaptic active cells induce depolarization we need
        #   - recursively check if we can backtrack from this segment

        if t < 0:
            return True, []

        # obviously, we should look only among active segments of these "should-be-depolarized" cells
        unique_potential_segments = self._get_unique_active_segments(should_be_depolarized_cells, t)
        self.print(trace, '\ncandidates:')
        for potential_segment in unique_potential_segments:
            self.agent.print_cells(trace, potential_segment)
        self.print(trace, '')

        # check every potential segment
        for potential_segment in unique_potential_segments:
            # check depolarization induced by segment's presynaptic cells
            #   i.e. how many "should be depolarized" cells are in fact depolarized
            presynaptic_active_cells = potential_segment
            n_depolarized_cells = self._count_induced_depolarization(
                presynaptic_active_cells, should_be_depolarized_cells, t
            )

            self.agent.print_cells(
                trace, presynaptic_active_cells,
                f'n: {n_depolarized_cells} of {self.agent.tm.activation_threshold}'
            )

            # if number of depolarized cells < threshold ==> they can't depolarize t+1 segment
            if n_depolarized_cells >= self.agent.tm.activation_threshold:
                backtracking_succeeded, activation_timeline = self._backtrack(
                    presynaptic_active_cells, t - 1, trace
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

        depolarized_rewarding_cells = self.agent.filter_cells_by_columns_range(
            all_final_depolarized_cells, rewarding_columns_range
        )
        return depolarized_rewarding_cells

    def _check_backtrack_correctly_predicts_reward(
            self, initial_sar: Sar, activation_timeline, trace: bool
    ) -> List[int]:
        self.print(trace, '\n===> Check backtracked activations')

        initial_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)
        proximal_input = self.agent.encoder.encode(initial_sar)
        T = len(self._active_segments_timeline)
        planned_actions = []

        for i in range(T):
            # choose action
            backtracked_activation = activation_timeline[i]
            backtracked_columns_activation = self.agent.columns_from_cells(backtracked_activation)
            backtracked_sar_superposition = self.agent.encoder.decode(backtracked_columns_activation)
            backtracked_actions = backtracked_sar_superposition.action
            # backtracked activation MUST CONTAIN only one action
            assert len(backtracked_actions) == 1
            action = backtracked_actions[0]
            planned_actions.append(action)

            proximal_input = self.agent.encoder.replace_action(proximal_input, action)

            active_cells, depolarized_cells = self.agent.process(proximal_input, learn=False, trace=trace)

            # current active cells MUST CONTAIN backtracked activation, i.e. the latter is a subset of it
            if not backtracked_activation <= set(active_cells):
                self.agent.print_cells(trace, active_cells, 'active cells')
                self.agent.print_cells(trace, backtracked_activation, 'backtracked')
            assert backtracked_activation <= set(active_cells)

            proximal_input = self.agent.columns_from_cells(depolarized_cells)
            self.print(trace, '')

        reward_reached = self.agent.encoder.is_rewarding(proximal_input)
        self._restore_initial_tm_state()

        if reward_reached:
            self.print(trace, f'<=== Checking complete with SUCCESS: {planned_actions}')
        else:
            self.print(trace, f'<=== Checking complete with NO success')

        # return all planned actions or None
        if reward_reached:
            return planned_actions

    def ground_backtracking_predictions_with_active_presynaptic_cells(
            self, active_presynaptic_cells, backtracking_connections
    ):
        backtracked_postsynaptic_cells = list(backtracking_connections.keys())
        actions_columns_range = self.agent.encoder.get_actions_indices_range()

        presynaptic_action_cells = set(
            self.agent.filter_cells_by_columns_range(active_presynaptic_cells, actions_columns_range)
        )
        postsynaptic_action_cells = set(
            self.agent.filter_cells_by_columns_range(backtracked_postsynaptic_cells, actions_columns_range)
        )

        allowed_action_cells = [
            postsynaptic_cell
            for postsynaptic_cell, cell_segments in backtracking_connections.items()
            for segment in cell_segments
            if (
                    postsynaptic_cell in postsynaptic_action_cells
                    and any(cell for cell in segment if cell in presynaptic_action_cells)
            )
        ]
        return allowed_action_cells

    def _save_initial_tm_state(self):
        """Saves TM state."""
        self._initial_tm_state = pickle.dumps(self.agent.tm)

    def _restore_initial_tm_state(self):
        """Restores saved TM state."""
        self.agent.tm = pickle.loads(self._initial_tm_state)

    @staticmethod
    def print(print_condition, str_to_print: str):
        if print_condition:
            print(str_to_print)
