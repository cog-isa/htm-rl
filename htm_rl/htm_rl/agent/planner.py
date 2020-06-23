import pickle
from typing import List, Mapping, Set, Tuple

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
        if reward_reached:
            for activation_timeline in self._backtrack_from_reward(verbose):
                planning_successful, planned_actions = self._check_activation_timeline_leads_to_reward(
                    initial_sar, activation_timeline, verbose
                )
                if planning_successful:
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

    def _backtrack_from_reward(self, verbose: bool):
        """
        Performs recursive backtracking for each possible pathway.

        :param verbose:
        :return: iterator on all successful backtracks. Each backtrack is forward view activation timeline.
            Activation timeline is a list of active cells sparse SDR at each previous timestep t.
        """
        trace(verbose, '\n===> Backward pass')

        T = len(self._active_segments_timeline)
        reward_activation_threshold = self.agent.encoder._encoders.reward.activation_threshold

        # All depolarized cells from the last step of forward prediction phase, when the reward was found
        all_final_depolarized_cells = list(self._active_segments_timeline[T-1].keys())

        # Take only cells corresponding to reward == 1. Backtracking is started from these cells.
        depolarized_reward_cells = self._get_reward_cells(all_final_depolarized_cells)
        self.agent.print_cells(verbose, depolarized_reward_cells, 'Reward == 1')

        # Gets all presynaptic cells clusters, each can induce desired reward == 1 depolarization
        rewarding_candidate_cell_cluster = self._get_backtracking_candidate_clusters(
            desired_depolarization=depolarized_reward_cells,
            sufficient_activation_threshold=reward_activation_threshold,
            t=T-1,
            verbose=verbose
        )

        # For each rewarding candidate cluster tries backtracking to the beginning
        for candidate_cluster, n_induced_depolarization in rewarding_candidate_cell_cluster:
            trace(verbose, '>')
            self.agent.print_cells(
                verbose, candidate_cluster,
                f'n: {n_induced_depolarization} of {reward_activation_threshold}'
            )
            backtracking_succeeded, activation_timeline = self._backtrack(candidate_cluster, T-2, verbose)
            trace(verbose, '<')

            if backtracking_succeeded:
                # yield successful activation timeline
                activation_timeline.append(candidate_cluster)
                yield activation_timeline

        trace(verbose, '<=== Backward pass complete')

    def _backtrack(self, desired_depolarization: SparseSdr, t: int, verbose: bool) -> Tuple:
        """
        Performs recursive backtracking for each possible pathway.

        :param verbose:
        :return: Tuple (whether backtracking was successful, forward view activation timeline).
            Activation timeline is a list of active cells sparse SDR at each previous timestep t.
        """

        if t < 0:
            return True, []

        trace(verbose)
        self.agent.print_sar_superposition(verbose, self.agent.columns_from_cells(desired_depolarization))

        # Gets all presynaptic cells clusters, each can induce desired depolarization
        candidate_cell_cluster = self._get_backtracking_candidate_clusters(
            desired_depolarization=desired_depolarization,
            sufficient_activation_threshold=self.agent.tm.activation_threshold,
            t=t, verbose=verbose
        )

        # For each candidate cluster tries backtracking to the beginning
        for candidate_cluster, n_induced_depolarization in candidate_cell_cluster:
            self.agent.print_cells(
                verbose, candidate_cluster,
                f'n: {n_induced_depolarization} of {self.agent.tm.activation_threshold}'
            )

            backtracking_succeeded, activation_timeline = self._backtrack(candidate_cluster, t-1, verbose)
            if backtracking_succeeded:
                activation_timeline.append(candidate_cluster)
                return True, activation_timeline

        return False, None

    def _get_backtracking_candidate_clusters(
            self, desired_depolarization: SparseSdr,
            sufficient_activation_threshold: int, t: int, verbose: bool
    ):
        """Gets presynaptic active cell clusters, which sufficient to induce desired depolarization."""

        # Gets active presynaptic cell clusters (clusterization by their columns)
        all_candidate_cell_clusters = self._get_active_presynaptic_cell_clusters(desired_depolarization, t)

        trace(verbose, 'candidates:')
        # Backtracking candidate clusters are clusters that induce sufficient depolarization among desired
        backtracking_candidate_clusters = []
        for active_cells in all_candidate_cell_clusters:
            n_depolarized_cells = self._count_induced_depolarization(
                active_cells, desired_depolarization, t
            )
            print_filter = n_depolarized_cells > 4
            self.agent.print_cells(
                print_filter and verbose, active_cells,
                f'n: {n_depolarized_cells} of {sufficient_activation_threshold}'
            )
            trace(print_filter and verbose, '----')

            if n_depolarized_cells >= sufficient_activation_threshold:
                backtracking_candidate_clusters.append((active_cells, n_depolarized_cells))
        trace(verbose, f'total:   {len(backtracking_candidate_clusters)}')

        return backtracking_candidate_clusters

    def _get_active_presynaptic_cell_clusters(self, depolarized_cells, t):
        """
        Gets active presynaptic cell clusters of given postsynaptic depolarized cells.

        Given set of depolarized cells, it takes their active segments and do column clusterization.

        :param depolarized_cells: depolarized postsynaptic cells
        :param t: time t at which active cells are considered
        :return: list of cell clusters. Each cluster is a set of active cells at time t.
        """
        active_segments_t = self._active_segments_timeline[t]
        clusters = []
        for cell in depolarized_cells:
            for cell_active_segment in active_segments_t[cell]:
                cols = frozenset(self.agent.columns_from_cells(cell_active_segment))
                clusters.append((cols, cell_active_segment))

        # greedy merge cell clusters if they have sufficient columns intersection
        while self._merge_clusters(clusters):
            ...

        # unzip list of tuples to tuple of lists
        column_clusters, cell_clusters = zip(*clusters)
        return cell_clusters

    def _merge_clusters(self, clusters: List):
        """
        Merges cell cluster pairs if they have sufficient columns intersection.
        :param clusters: list of clusters. Each cluster is Tuple(columns, cells)
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

        columns_i, cells_i = clusters[i]
        columns_j, cells_j = clusters[j]

        sufficient_intersection = min(
            # if one of them is subset of the other
            len(columns_i), len(columns_j),
            # TODO: WARN! Handcrafted threshold
            self.agent.tm.activation_threshold - 2
        )

        intersection = columns_i & columns_j
        if len(intersection) >= sufficient_intersection:
            # merge them by taking a union of them
            return columns_i | columns_j, cells_i | cells_j
        return None

    def _count_induced_depolarization(
            self, active_cells: Set, desired_depolarization: SparseSdr, t: int
    ) -> int:
        """
        Gets the fracture of desired depolarization induced by given active cells.

        :param active_cells: a set of active cells
        :param desired_depolarization: sparse SDR of depolarization candidate cells that should be checked
        :param t: time t at which active cells are active
        :return: number of depolarized cells
        """
        active_segments_t = self._active_segments_timeline[t]

        induced_depolarization = []
        for depolarization_candidate_cell in desired_depolarization:
            cell_segments = active_segments_t[depolarization_candidate_cell]

            any_cell_segments_activated = any(
                len(segment & active_cells) >= self.agent.tm.activation_threshold
                for segment in cell_segments
            )
            # the cell becomes depolarized if any of its segments becomes active
            if any_cell_segments_activated:
                induced_depolarization.append(depolarization_candidate_cell)

        depolarized_columns = self.agent.columns_from_cells(induced_depolarization)
        # noinspection PyTypeChecker
        return len(depolarized_columns)

    def _get_reward_cells(self, cells_sparse_sdr: SparseSdr) -> SparseSdr:
        """
        Filters cells by keeping only cells from the reward == 1 columns.
        """

        # BitRange [l, r): range of columns corresponding to reward == 1
        rewarding_columns_range = self.agent.encoder.rewarding_indices_range()

        # filter cells by that range
        depolarized_rewarding_cells = self.agent.filter_cells_by_columns_range(
            cells_sparse_sdr, rewarding_columns_range
        )
        return depolarized_rewarding_cells

    def _check_activation_timeline_leads_to_reward(
            self, initial_sar: Sar, activation_timeline, verbose: bool
    ) -> Tuple[bool, List[int]]:
        trace(verbose, '\n===> Check backtracked activations')

        initial_sar = Sar(initial_sar.state, IntSdrEncoder.ALL, initial_sar.reward)
        proximal_input = self.agent.encoder.encode(initial_sar)
        T = len(self._active_segments_timeline)
        planned_actions = []

        for i in range(T):
            action = self._extract_action_from_backtracked_activation(activation_timeline[i])
            # set action to proximal input
            proximal_input = self.agent.encoder.replace_action(proximal_input, action)
            planned_actions.append(action)

            active_cells, depolarized_cells = self.agent.process(proximal_input, learn=False, verbose=verbose)

            proximal_input = self.agent.columns_from_cells(depolarized_cells)
            trace(verbose, '')

        reward_reached = self.agent.encoder.is_rewarding(proximal_input)
        self._restore_initial_tm_state()

        if reward_reached:
            trace(verbose, f'<=== Checking complete with SUCCESS: {planned_actions}')
            return True, planned_actions
        else:
            trace(verbose, f'<=== Checking complete with NO success')
            return False, planned_actions

    def _extract_action_from_backtracked_activation(self, backtracked_activation: SparseSdr) -> int:
        backtracked_columns_activation = self.agent.columns_from_cells(backtracked_activation)
        backtracked_sar_superposition = self.agent.encoder.decode(backtracked_columns_activation)
        backtracked_actions = backtracked_sar_superposition.action

        # backtracked activation MUST CONTAIN only one action
        assert len(backtracked_actions) == 1
        action = backtracked_actions[0]
        return action

    def _save_initial_tm_state(self):
        """Saves TM state."""
        self._initial_tm_state = pickle.dumps(self.agent.tm)

    def _restore_initial_tm_state(self):
        """Restores saved TM state."""
        self.agent.tm = pickle.loads(self._initial_tm_state)
