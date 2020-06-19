import pickle
from collections import defaultdict
from typing import List, Mapping, Set

from htm_rl.agent.agent import Agent
from htm_rl.common.base_sar import Sar
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import trace, range_reverse


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

    def plan_actions(self, initial_sar: Sar, verbose: bool):
        trace(verbose, '\n======> Planning')

        # saves initial TM state at the start of planning.
        self._save_initial_tm_state()
        planned_actions = None

        reward_reached = self._make_predictions_to_reward(initial_sar, verbose)
        if not reward_reached:
            return

        for activation_timeline in self._yield_successful_backtracks_from_reward(verbose):
            planned_actions = self._check_backtrack_correctly_predicts_reward(
                initial_sar, activation_timeline, verbose
            )
            if planned_actions is not None:
                break

        trace(verbose, '<====== Planning complete')
        return planned_actions

    def _make_predictions_to_reward(self, initial_sar: Sar, verbose: bool) -> bool:
        trace(verbose, '===> Forward pass')

        # to consider all possible prediction paths, prediction is started with all possible actions
        all_actions_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)

        proximal_input = self.agent.encoder.encode(all_actions_sar)
        reward_reached = False
        active_segments_timeline = []

        for i in range(self.max_steps):
            reward_reached = self.agent.encoder.is_rewarding(proximal_input)
            if reward_reached:
                break

            active_cells, depolarized_cells = self.agent.process(proximal_input, learn=False, verbose=verbose)

            active_segments_t = self.agent.active_segments(active_cells)
            active_segments_timeline.append(active_segments_t)

            proximal_input = self.agent.columns_from_cells(depolarized_cells)
            trace(verbose, '')

        self._restore_initial_tm_state()
        self._active_segments_timeline = active_segments_timeline

        if reward_reached:
            T = len(active_segments_timeline)
            trace(verbose, f'<=== Predict reward in {T} steps')
        else:
            trace(verbose, f'<=== Predicting NO reward in {self.max_steps} steps')

        return reward_reached

    def _yield_successful_backtracks_from_reward(self, verbose: bool):
        trace(verbose, '\n===> Backward pass')

        T = len(self._active_segments_timeline)
        for rewarding_segment in self._yield_rewarding_segments(verbose):
            # rewarding segment:
            #   - one of the active segments at time T-1
            #   - consists of a set of presynaptic [active at time T-1] cells
            #   - these active cells induce reward prediction [happening at time T]

            # start backtracking from time T-2:
            # when these presynaptic active cells were just a prediction (= depolarized)

            depolarized_cells = rewarding_segment
            backtracking_succeeded, activation_timeline = self._backtrack(depolarized_cells, T-2, verbose)
            if backtracking_succeeded:
                activation_timeline.append(rewarding_segment)
                yield activation_timeline

        trace(verbose, '<=== Backward pass complete')

    def _yield_rewarding_segments(self, verbose: bool):
        """TODO"""
        T = len(self._active_segments_timeline)
        reward_activation_threshold = self.agent.encoder._encoders.reward.activation_threshold

        # Take active_segments for time T-1, when the reward was found.
        depolarized_reward_cells = self._get_depolarized_reward_cells()
        self.agent.print_cells(verbose, depolarized_reward_cells, 'Reward == 1')

        # all segments, whose presynaptic cells can potentially induce desired reward == 1 prediction
        rewarding_segment_candidates = self._get_backtracking_candidates(
            should_be_depolarized_cells=depolarized_reward_cells,
            sufficient_activation_threshold=reward_activation_threshold,
            t=T - 1,
            verbose=verbose
        )

        for candidate_segment, n_induced_depolarization in rewarding_segment_candidates:
            trace(verbose, '>')
            self.agent.print_cells(
                verbose, candidate_segment, f'n: {n_induced_depolarization} of {reward_activation_threshold}'
            )
            yield candidate_segment
            trace(verbose, '<')

    def _backtrack(self, should_be_depolarized_cells, t, verbose):
        # goal:
        #   - find any active_segment whose presynaptic active cells induce depolarization we need
        #   - recursively check if we can backtrack from this segment

        if t < 0:
            return True, []

        trace(verbose)
        self.agent.print_sar_superposition(verbose, self.agent.columns_from_cells(should_be_depolarized_cells))

        # obviously, we should look only among active segments of these "should-be-depolarized" cells
        unique_potential_segments = self._get_backtracking_candidates(
            should_be_depolarized_cells=should_be_depolarized_cells,
            sufficient_activation_threshold=self.agent.tm.activation_threshold,
            t=t, verbose=verbose
        )

        # check every candidate
        for candidate, n_induced_depolarization in unique_potential_segments:
            self.agent.print_cells(
                verbose, candidate, f'n: {n_induced_depolarization} of {self.agent.tm.activation_threshold}'
            )

            backtracking_succeeded, activation_timeline = self._backtrack(
                candidate, t - 1, verbose
            )
            if backtracking_succeeded:
                activation_timeline.append(candidate)
                return True, activation_timeline

        return False, None

    def _get_backtracking_candidates(
            self, should_be_depolarized_cells, sufficient_activation_threshold: int, t: int, verbose: bool
    ):
        """Find any active_segment whose presynaptic active cells induce depolarization we need."""

        # obviously, we should look only among active segments of these "should-be-depolarized" cells
        unique_potential_segments = self._get_active_cell_clusters(should_be_depolarized_cells, t)

        trace(verbose, 'candidates:')
        candidates = []
        n_all_candidate = 0
        for candidate_segment in unique_potential_segments:
            n_depolarized_cells = self._count_induced_depolarization(
                active_cells=candidate_segment,
                depolarization_candidates=should_be_depolarized_cells,
                t=t
            )
            n_all_candidate += 1
            if n_depolarized_cells > 4:
                self.agent.print_cells(
                    verbose, candidate_segment,
                    f'n: {n_depolarized_cells} of {sufficient_activation_threshold}'
                )
                trace(verbose, '----')
            if n_depolarized_cells >= sufficient_activation_threshold:
                candidates.append((candidate_segment, n_depolarized_cells))
        trace(verbose, f'total:   {n_all_candidate}')

        return candidates

    def _get_active_cell_clusters(self, depolarized_cells, t):
        """TODO"""
        active_segments_t = self._active_segments_timeline[t]
        # clusters = defaultdict(set)
        # for cell in depolarized_cells:
        #     for cell_active_segment in active_segments_t[cell]:
        #         cluster = frozenset(self.agent.columns_from_cells(cell_active_segment))
        #         clusters[cluster].update(cell_active_segment)

        clusters = []
        for cell in depolarized_cells:
            for cell_active_segment in active_segments_t[cell]:
                cols = frozenset(self.agent.columns_from_cells(cell_active_segment))
                clusters.append((cols, cell_active_segment))

        while self._merge_clusters(clusters):
            ...

        column_clusters, cell_clusters = zip(*clusters)

        # print(column_clusters)
        return cell_clusters

    def _merge_clusters(self, clusters: List):
        initial_len = len(clusters)

        for i in range_reverse(clusters):
            cluster_cols, cluster_cells = clusters[i]
            for j in range(i):
                cols, cells = clusters[j]
                is_subset = cluster_cols <= cols or cluster_cols >= cols
                enough_intersection = len(cluster_cols & cols) >= self.agent.tm.activation_threshold - 2
                if is_subset or enough_intersection:
                    # merge i into j
                    clusters[j] = (cluster_cols | cols, cluster_cells | cells)
                    # remove i by swapping it with the last and then popping
                    clusters[i] = clusters[-1]
                    clusters.pop()
                    break

        return len(clusters) < initial_len

    def _count_induced_depolarization(self, active_cells, depolarization_candidates, t):
        """TODO"""
        active_segments_t = self._active_segments_timeline[t]

        # look only among specified "could-be-depolarized" cells
        depolarized_cells = []
        for depolarization_candidate_cell in depolarization_candidates:
            cell_segments = active_segments_t[depolarization_candidate_cell]

            # for segment in cell_segments:
            #     self.agent.print_cells(segment, f' {depolarization_candidate_cell:2} presynaptics')

            any_cell_segments_activated = any(
                len(segment & active_cells) >= self.agent.tm.activation_threshold
                for segment in cell_segments
            )
            # the cell becomes depolarized if any of its segments becomes active
            if any_cell_segments_activated:
                depolarized_cells.append(depolarization_candidate_cell)

        depolarized_columns = self.agent.columns_from_cells(depolarized_cells)
        return len(depolarized_columns)

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
            self, initial_sar: Sar, activation_timeline, verbose: bool
    ) -> List[int]:
        trace(verbose, '\n===> Check backtracked activations')

        initial_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)
        proximal_input = self.agent.encoder.encode(initial_sar)
        T = len(self._active_segments_timeline)
        planned_actions = []

        for i in range(T):
            # choose action
            backtracked_activation = activation_timeline[i]
            backtracked_columns_activation = self.agent.columns_from_cells(backtracked_activation)

            # current active cells MUST CONTAIN backtracked activation, i.e. the latter is a subset of it
            if not set(backtracked_columns_activation) <= set(proximal_input):
                trace(verbose, f'{self.agent.format_sdr(proximal_input)} proximal_input')
                trace(verbose, f'{self.agent.format_sdr(backtracked_columns_activation)} backtracked')
                # assert False

            backtracked_sar_superposition = self.agent.encoder.decode(backtracked_columns_activation)
            backtracked_actions = backtracked_sar_superposition.action
            # backtracked activation MUST CONTAIN only one action
            assert len(backtracked_actions) == 1
            action = backtracked_actions[0]
            planned_actions.append(action)

            proximal_input = self.agent.encoder.replace_action(proximal_input, action)

            active_cells, depolarized_cells = self.agent.process(proximal_input, learn=False, verbose=verbose)

            proximal_input = self.agent.columns_from_cells(depolarized_cells)
            trace(verbose, '')

        reward_reached = self.agent.encoder.is_rewarding(proximal_input)
        self._restore_initial_tm_state()

        if reward_reached:
            trace(verbose, f'<=== Checking complete with SUCCESS: {planned_actions}')
        else:
            trace(verbose, f'<=== Checking complete with NO success')

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
