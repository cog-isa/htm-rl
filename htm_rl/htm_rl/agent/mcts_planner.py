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
        predicted_states, goal_state = self._backtrack(reached_goals, verbosity)

        trace(verbosity, 2, '<====== Planning complete')
        return predicted_states, goal_state

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

    def _split_states_for_actions(
            self, active_segments: Dict[int, List[Set[int]]], verbosity: int
    ) -> List[SparseSdr]:
        """
        Split prediction, represented as active segments, by actions, i.e. which
        state is predicted for which action. Returns list of state SparseSdr, each
        state for every action.
        """
        action_outcomes: List[SparseSdr] = [[] for _ in range(self._n_actions)]
        state_cells = self.memory.filter_cells_by_columns_range(
            active_segments, self.encoder.states_indices_range()
        )

        for cell in state_cells:
            for presynaptic_cells in active_segments[cell]:
                presynaptic_columns = self.memory.columns_from_cells(presynaptic_cells)
                # which state-action pair activates `cell` in a form of superposition
                initial_sa_superposition = self.encoder.decode(presynaptic_columns)
                # iterating here is redundant - it should be always only one action
                for action in initial_sa_superposition.action:
                    action_outcomes[action].append(cell)

        return action_outcomes


    def _backtrack(self, reached_goal_states: List[int], verbosity: int):
        """
        Performs recursive backtracking for each possible pathway starting from reached goal states.

        :param verbosity:
        :return: iterator on all successful backtracks. Each backtrack is forward view activation timeline.
            Activation timeline is a list of active cells sparse SDR at each previous timestep t.
        """
        trace(verbosity, 2, '\n===> Backward pass')
        planned_actions, goal_state = [], None

        T = len(self._active_segments_timeline)
        if not reached_goal_states or T == 0:
            return planned_actions, goal_state

        depolarized_cells = self._active_segments_timeline[T-1].keys()
        for goal_state in reached_goal_states:
            goal_state_columns_sdr = self.encoder.encode(Sa(goal_state, None))
            active_goal_cells = self.memory.filter_cells_by_columns(
                cells=depolarized_cells, columns=goal_state_columns_sdr
            )

            planning_successful, planned_actions = self._backtrack_from_state(
                desired_depolarization=active_goal_cells, t=T-1, verbosity=verbosity
            )
            if planning_successful:
                break

        trace(verbosity, 2, '<=== Backward pass complete')
        return planned_actions, goal_state

    def _backtrack_from_state(self, desired_depolarization: SparseSdr, t: int, verbosity: int) -> Tuple:
        """
        Performs recursive backtracking for each possible pathway.

        :param verbosity: accepted level of verbosity to trace
        :return: Tuple (whether backtracking was successful, forward view activation timeline).
            Activation timeline is a list of active cells sparse SDR at each previous timestep t.
        """

        if t < 0:
            return True, []

        trace(verbosity, 3)
        self.memory.print_sa_superposition(
            verbosity, 3, self.memory.columns_from_cells(desired_depolarization)
        )

        # Gets all presynaptic cells clusters, each can induce desired depolarization
        candidate_cell_clusters = self._get_backtracking_candidate_clusters(
            desired_depolarization=desired_depolarization,
            sufficient_activation_threshold=self.encoder.state_activation_threshold,
            t=t, verbosity=verbosity
        )

        # For each candidate cluster tries backtracking to the beginning
        for candidate_cluster, n_induced_depolarization in candidate_cell_clusters:
            self.memory.print_cells(
                verbosity, 3, candidate_cluster,
                f'n: {n_induced_depolarization} of {self.memory.tm.activation_threshold}'
            )

            states_sdr, actions_sdr = self._split_sa_cells(candidate_cluster)
            action_columns = self.memory.columns_from_cells(actions_sdr)
            actions = self.encoder.decode(action_columns)
            if len(actions.action) != 1:
                trace(verbosity, 2, f'Actions: {actions}')
                continue

            # actions has length == 1
            action = actions.action[0]

            backtracking_state = states_sdr
            backtracking_succeeded, policy = self._backtrack_from_state(
                desired_depolarization=backtracking_state, t=t-1, verbosity=verbosity
            )
            if backtracking_succeeded:
                policy.append(action)
                return True, policy

        return False, []

    def _get_backtracking_candidate_clusters(
            self, desired_depolarization: SparseSdr,
            sufficient_activation_threshold: int, t: int, verbosity: int
    ):
        """Gets presynaptic active cell clusters, which sufficient to induce desired depolarization."""

        # Gets active presynaptic cell clusters
        all_candidate_cell_clusters = self._get_active_presynaptic_cell_clusters(desired_depolarization, t)

        trace(verbosity, 3, 'candidates:')
        # Backtracking candidate clusters are clusters that induce sufficient depolarization among desired
        backtracking_candidate_clusters = []
        for cell_cluster, n_containing_segments in all_candidate_cell_clusters:
            cluster_for_backtracking = self._check_candidate_cluster(
                cell_cluster, n_containing_segments, desired_depolarization, t,
                sufficient_activation_threshold, verbosity
            )

            if cluster_for_backtracking is not None:
                backtracking_candidate_clusters.append(cluster_for_backtracking)
        trace(verbosity, 3, f'total:   {len(backtracking_candidate_clusters)}')

        return backtracking_candidate_clusters

    def _check_candidate_cluster(
            self, cell_cluster: Set[int], n_containing_segments: int,
            desired_depolarization: SparseSdr, t: int,
            sufficient_activation_threshold: int, verbosity: int
    ):
        if n_containing_segments < 5:   # obviously noise clusters
            return

        active_cells = cell_cluster
        n_depolarized_cells = n_containing_segments     # just an estimate!

        if n_containing_segments >= sufficient_activation_threshold - 2:
            # expecting insufficient depolarization, hence get precise depolarization
            n_depolarized_cells = self._count_induced_depolarization(
                active_cells, desired_depolarization, t
            )

            if n_depolarized_cells != n_containing_segments:
                # hypothesis: they should be equal every time, hence we never reach this branch!
                trace(verbosity, 2, f'{n_containing_segments} -> {n_depolarized_cells}')
                self.memory.print_cells(
                    verbosity, 2, active_cells,
                    f'n: {n_depolarized_cells} of {sufficient_activation_threshold}'
                )

        self.memory.print_cells(
            verbosity, 3, active_cells,
            f'n: {n_depolarized_cells} of {sufficient_activation_threshold}'
        )
        trace(verbosity, 3, '----')

        if n_depolarized_cells >= sufficient_activation_threshold:
            return active_cells, n_depolarized_cells

    def _get_active_presynaptic_cell_clusters(self, depolarized_cells, t):
        """
        Gets active presynaptic cell clusters of given postsynaptic depolarized cells.

        Given set of depolarized cells, it takes their active segments and do cells clustering.

        :param depolarized_cells: depolarized postsynaptic cells
        :param t: time t at which active cells are considered
        :return: list of cell clusters. Each cluster is a tuple:
            - set of active cells at time t
            - number of segments merged into cluster
        """
        active_segments_t = self._active_segments_timeline[t]
        cell_clusters = []
        for cell in depolarized_cells:
            for cell_active_segment in active_segments_t[cell]:
                cell_clusters.append((cell_active_segment, 1))

        # merge cell clusters greedy if they have sufficient intersection
        while self._merge_clusters(cell_clusters):
            ...

        return cell_clusters

    def _merge_clusters(self, clusters: List):
        """
        Merges cell cluster pairs if they have sufficient intersection.
        :param clusters: list of clusters. Cluster is a tuple(cell_cluster, n_merged_segments)
        :return: whether any clusters were successfully merged
        """
        initial_n_clusters = len(clusters)
        for j in range_reverse(clusters):
            for i in range(len(clusters)):
                if i == j:
                    continue
                merged_cluster = self._merge_two_cell_clusters(clusters, i, j)
                if merged_cluster is not None:
                    clusters[i] = merged_cluster
                    # remove cluster j by swapping it with the last and then popping
                    clusters[j] = clusters[-1]
                    clusters.pop()
                    break

        return len(clusters) < initial_n_clusters

    def _merge_two_cell_clusters(self, clusters, i, j):
        """Merges two cell clusters if they have sufficient columns intersection."""

        cluster_i, n_merged_segments_i = clusters[i]
        cluster_j, n_merged_segments_j = clusters[j]

        sufficient_intersection = min(
            # if one of them is subset of another
            len(cluster_i), len(cluster_j),
            # or it's almost the activation threshold
            self._clusters_merging_threshold
        )

        intersection = cluster_i & cluster_j
        if len(intersection) >= sufficient_intersection:
            # merge them by taking a union of them, also sum up their merge counters
            return cluster_i | cluster_j, n_merged_segments_i + n_merged_segments_j
        return None

    def _count_induced_depolarization(
            self, active_cells: Set, desired_depolarization: SparseSdr, t: int
    ) -> int:
        """
        Gets the fracture of desired depolarization induced by given active cells. The number
        of activated columns is counted.

        :param active_cells: a set of active cells
        :param desired_depolarization: sparse SDR of depolarization candidate cells that should be checked
        :param t: time t at which active cells are active
        :return: number of depolarized columns.
        """
        active_segments_t = self._active_segments_timeline[t]

        induced_depolarization = []
        for depolarization_candidate_cell in desired_depolarization:
            cell_segments = active_segments_t[depolarization_candidate_cell]

            any_cell_segments_activated = any(
                len(segment & active_cells) >= self.memory.tm.activation_threshold
                for segment in cell_segments
            )
            # the cell becomes depolarized if any of its segments becomes active
            if any_cell_segments_activated:
                induced_depolarization.append(depolarization_candidate_cell)

        depolarized_columns = self.memory.columns_from_cells(induced_depolarization)
        return len(depolarized_columns)

    def _split_sa_cells(self, cells_sparse_sdr: SparseSdr) -> Tuple[SparseSdr, SparseSdr]:
        states_columns_range = self.encoder.states_indices_range()
        states_only_sparse_sdr = self.memory.filter_cells_by_columns_range(
            cells_sparse_sdr, states_columns_range
        )
        actions_columns_range = self.encoder.actions_indices_range()
        actions_only_sparse_sdr = self.memory.filter_cells_by_columns_range(
            cells_sparse_sdr, actions_columns_range
        )
        return states_only_sparse_sdr, actions_only_sparse_sdr

    def _extract_action_from_backtracked_activation(self, backtracked_activation: SparseSdr) -> int:
        backtracked_columns_activation = self.memory.columns_from_cells(backtracked_activation)
        backtracked_sar_superposition = self.encoder.decode(backtracked_columns_activation)
        backtracked_actions = backtracked_sar_superposition.action

        # backtracked activation MUST CONTAIN only one action
        assert len(backtracked_actions) == 1
        action = backtracked_actions[0]
        return action