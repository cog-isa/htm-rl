from collections import deque
from typing import List, Mapping, Set, Tuple, Deque
from random import sample

from htm_rl.agent.memory import Memory
from htm_rl.common.base_sa import Sa
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import range_reverse, trace


# noinspection PyPep8Naming


class Planner:
    memory: Memory
    planning_horizon: int
    encoder: SaSdrEncoder
    episode_goal_memory: 'GoalMemory'
    inter_episode_goal_memory: 'GoalMemory'
    alpha: float

    # segments[time][cell] = segments = [ segment ] = [ [presynaptic_cell] ]
    _active_segments_timeline: List[Mapping[int, List[Set[int]]]]
    _clusters_merging_threshold: int

    def __init__(self, memory: Memory, planning_horizon: int, goal_memory_size: int, alpha: float):
        self.alpha = alpha
        self.memory = memory
        self.encoder = memory.encoder
        self.planning_horizon = planning_horizon
        self.episode_goal_memory = GoalMemory(goal_memory_size, memory.tm.n_columns,
                                              alpha, memory.tm.activation_threshold)
        self.inter_episode_goal_memory = GoalMemory(goal_memory_size, memory.tm.n_columns,
                                                    alpha, memory.tm.activation_threshold)

        self._active_segments_timeline = []
        self._initial_tm_state = None

        # TODO: WARN! Handcrafted threshold
        self._clusters_merging_threshold = self.memory.tm.activation_threshold

    def plan_actions(self, initial_sa: Sa, verbosity: int):
        planned_actions, goal_state = [], None
        if self.planning_horizon == 0 or not self.episode_goal_memory:
            return planned_actions, goal_state

        # TODO: check if it should be there
        self.episode_goal_memory.remove(initial_sa.state in self.episode_goal_memory)

        trace(verbosity, 2, '\n======> Planning')

        # saves initial TM state at the start of planning.
        self._initial_tm_state = self.memory.save_tm_state()

        reached_goals = self._predict_to_goals(initial_sa, verbosity)
        planned_actions, goal_state = self._backtrack(reached_goals, verbosity)

        trace(verbosity, 2, '<====== Planning complete')
        return planned_actions, goal_state

    def _predict_to_goals(self, initial_sa: Sa, verbosity: int) -> List['SparseSDRUnion']:
        trace(verbosity, 2, '===> Forward pass')

        # to consider all possible prediction paths, prediction is started with all possible actions
        all_actions_sa = Sa(initial_sa.state, IntSdrEncoder.ALL)

        proximal_input = self.encoder.encode(all_actions_sa)
        reached_goals = []
        active_segments_timeline = []

        self.memory.print_tm_state(verbosity, 1,
                                   f'forward planning initial state: {all_actions_sa.state} action: {all_actions_sa.action}')

        for i in range(self.planning_horizon):
            active_cells, depolarized_cells = self.memory.process(
                proximal_input, learn=False, verbosity=verbosity
            )

            active_segments_t = self.memory.active_segments(active_cells)
            active_segments_timeline.append(active_segments_t)

            proximal_input = self.memory.columns_from_cells(depolarized_cells)
            state_superposition_sdr, _ = self.encoder.split_sa(proximal_input)
            reached_goals = self._reached_goals(state_superposition_sdr)
            trace(verbosity, 3, '')

            self.memory.print_tm_state(verbosity, 1,
                                       f'forward planning step: {i}')

            if reached_goals or not proximal_input:
                break

        self.memory.print_tm_state(verbosity, 1,
                                   f'forward planning end',
                                   save_on_disk=True)

        self.memory.restore_tm_state(self._initial_tm_state)
        self._active_segments_timeline = active_segments_timeline

        if reached_goals:
            T = len(active_segments_timeline)
            trace(verbosity, 2, f'<=== Predict reaching goals {reached_goals} in {T} steps')
        else:
            trace(verbosity, 2, f'<=== Predicting NO goals in {self.planning_horizon} steps')

        return reached_goals

    def _backtrack(self, reached_goal_states: List['SparseSDRUnion'], verbosity: int):
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
            goal_state_columns_sdr = goal_state.union
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
        if n_containing_segments < sufficient_activation_threshold * 0.5:   # obviously noise clusters ??
            return

        active_cells = cell_cluster
        # n_depolarized_cells = n_containing_segments     # just an estimate!

        # if n_containing_segments >= sufficient_activation_threshold - 2:
        #     # expecting insufficient depolarization, hence get precise depolarization
        n_depolarized_cells = self._count_induced_depolarization(
            active_cells, desired_depolarization, t
        )

        trace(verbosity, 2, f'containing segments {n_containing_segments} -> depolarized cells {n_depolarized_cells}')
            # if n_depolarized_cells != n_containing_segments:
            #     # seems like n_depolarized_cells > n_containing_segments is normal situation
            #     trace(verbosity, 2, f'{n_containing_segments} -> {n_depolarized_cells}')
            #     self.memory.print_cells(
            #         verbosity, 2, active_cells,
            #         f'n: {n_depolarized_cells} of {sufficient_activation_threshold}'
            #     )

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

    def _reached_goals(self, proximal_input):
        return proximal_input in self.episode_goal_memory

    def add_goal(self, goal_state):
        self.episode_goal_memory.add(goal_state)
        self.inter_episode_goal_memory.add(goal_state)

    def remove_goal(self, goal_state):
        self.episode_goal_memory.remove(goal_state)

    def restore_goal_list(self):
        self.episode_goal_memory = self.inter_episode_goal_memory.copy()


class SparseSDRUnion:
    alpha: float
    size: int

    def __init__(self, sparse_sdr: SparseSdr, alpha: float, size: int):
        self.alpha = alpha
        self.size = size
        self.union = set(SparseSdr)

    def __len__(self):
        return self.size

    def add(self, sparse_sdr: SparseSdr):
        self.union = self.union.difference(sample(self.union, round(self.alpha * len(self.union))))
        self.union.add(SparseSdr)


class GoalMemory:
    threshold: int
    alpha: float
    goals: List[SparseSDRUnion]

    def __init__(self, n_goals, goal_size, alpha, threshold):
        self.threshold = threshold
        self.alpha = alpha
        self.goals = list()
        self.goal_size = goal_size
        self.n_goals = n_goals

    def add(self, sparse_sdr: SparseSdr):
        # filter columns that don't correspond to goal
        for goal in self.goals:
            if len(goal.union.intersection(sparse_sdr)) >= self.threshold:
                goal.add(sparse_sdr)
                break
        else:
            self.goals.append(SparseSDRUnion(sparse_sdr, self.alpha, self.goal_size))
            if len(self.goals) > self.n_goals:
                self.goals.pop(0)

    def remove(self, goals: List[SparseSDRUnion]):
        for goal in goals:
            self.goals.remove(goal)

    def copy(self) -> 'GoalMemory':
        cp = GoalMemory(self.n_goals, self.goal_size,
                        self.alpha, self.threshold)
        cp.goals = self.goals
        return cp

    def __contains__(self, sparse_sdr: SparseSdr) -> List[SparseSDRUnion]:
        # returns all goals that constitute sparse_sdr
        subset = list()
        for goal in self.goals:
            if len(goal.union.intersection(sparse_sdr)) >= self.threshold:
                subset.append(goal)
        return subset

    def __len__(self):
        return len(self.goals)



class CircularSet:
    max_size: int

    _set: Set[int]
    _deque: Deque[int]

    def __init__(self, max_size: int):
        self.max_size = max_size

        self._set = set()
        self._deque = deque(maxlen=max_size)

    def __len__(self):
        return len(self._set)

    def add(self, item):
        # check if it's already exist
        # Speed up by checking against the last added element first
        if self._deque and item == self._deque[-1] or item in self._set:
            return

        if len(self._deque) == self.max_size:
            # full => remove from set first
            self._set.remove(self._deque[0])

        self._deque.append(item)
        self._set.add(item)

    def remove(self, item):
        self._set.remove(item)
        self._deque.remove(item)

    def copy(self) -> 'CircularSet':
        cp = CircularSet(self.max_size)
        cp._deque = self._deque.copy()
        cp._set = self._set.copy()
        return cp

    def __contains__(self, item):
        return item in self._set