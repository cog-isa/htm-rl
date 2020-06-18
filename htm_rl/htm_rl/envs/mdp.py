from typing import Type

from gym.utils import seeding

from htm_rl.common.sar_sdr_encoder import SarSuperposition


class Mdp:
    def __init__(self, transitions, initial_state=None, seed=None):
        """
        Defines an MDP. Compatible with gym Env.
        :param transitions: (s, a) -> s_next
            A dictionary of dictionaries: Dict[state][action] = next_state
            If a Dict[state] is None, it is considered terminal and rewarding (reward = 1).
        :param initial_state: a state where agent starts or a callable() -> state
            By default, picks initial state at random.

        Example:
        transitions = {
            0: {0: 4, 1: 1},
            1: {0: 1, 1: 2},
            2: {0: 2, 1: 3},
            3: {0: 3, 1: 0},
            4: None
        ]
        rewarding terminal state: 4
        """

        self._initial_state = initial_state
        self.transitions = transitions
        self.n_states = len(transitions)
        self.n_actions = max(len(edges) for edges in transitions.values() if edges is not None)
        self.np_random, _ = seeding.np_random(seed)
        self._current_state = self._get_initial_state()

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return self.transitions[state] is None

    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = self._get_initial_state()
        return self._current_state

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(self._current_state), 'Episode is finished!'

        next_state = self.transitions[self._current_state][action]
        assert next_state is not None

        reward = self._reward(next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        return next_state, reward, is_done, {}

    def render(self):
        state = self._current_state
        print(f'State: {state}')

    def _get_initial_state(self) -> int:
        state = None
        if self._initial_state is None:
            while state is None or self.is_terminal(state):
                state = self.np_random.choice(self.n_states)
        elif 0 <= self._initial_state < self.n_states:
            state = self._initial_state
        elif callable(self._initial_state):
            state = self._initial_state()
        return state

    def _reward(self, state):
        """ return the reward you get for taking action in state and landing on next_state"""
        return 1 if self.is_terminal(state) else 0


class GridworldMdpGenerator:
    """
    Gridworld MDP environment generator.
    Grid cells doesn't have to be squares - e.g. you can set them as ranges or hexagons. Check out init params.
    """
    _n_of_cell_edges: int

    def __init__(self, n_of_cell_edges: int = 4):
        assert n_of_cell_edges >= 2 and n_of_cell_edges % 2 == 0
        self._n_of_cell_edges = n_of_cell_edges

    def generate_env(
            self, env_type: Type[Mdp], initial_state, cell_transitions, allow_clockwise_action=False, seed=None
    ) -> Mdp:
        """
        Generates MDP/POMDP based on a grid world with 2 [or 3] allowed actions:
            - 0: move forward
            - 1: turn counter-clockwise
            - [optional] you can allow 3rd action 2: turn clockwise

        By convenience directions for square grid: 0 - right, 1 - up, 2 - left, 3 - down
            , i.e. from 0 to 3 counter-clockwise, starting from right.
        :param env_type: type of environment. It may be MDP or
        :param initial_state: can be tuple (cell, view_direction)
        :param cell_transitions: List[(cell, view_direction, next_cell)]
            Last transition should be on of the terminal transitions.
            Because last transition's destination will be set as terminal cell.
        :param allow_clockwise_action:  by default only counter-clockwise turn is allowed
        :param seed: optional seed for a random generator
        :return: Gym's environment-like object
        """
        transitions = self._generate_transitions(cell_transitions, allow_clockwise_action)

        if initial_state is not None and initial_state[1] is not None:
            cell, view_direction = initial_state
            initial_state = self._state(cell, view_direction)

        return env_type(transitions, initial_state, seed)

    def _generate_transitions(self, cell_transitions, add_clockwise_action=False):
        """
        Generates transitions for MDP based on a grid world with 2 [or 3] allowed actions:
            - move forward
            - turn counter-clockwise
            - [optional] you can allow 3rd action - turn clockwise

        By convenience directions: 0 - right, 1 - up, 2 - left, 3 - down
            , i.e. [0, 3] counter-clockwise.
        :param cell_transitions: List[(cell, direction, next_cell)].
            Last transition must be to the terminal rewarding cell.
        :param add_clockwise_action: by default only counter-clockwise turn is allowed
        :return: Dict[state][action] = next_state
        """
        terminal_cell = cell_transitions[-1][2]
        n_full_cells = terminal_cell

        n_states = n_full_cells * self._n_of_cell_edges + 1  # n_cells full cells + one cell w/ only one terminal state
        terminal_state = n_states - 1
        transitions = {terminal_state: None}
        for cell in range(n_full_cells):
            transitions.update(self._generate_separate_cell(cell, add_clockwise_action))

        for cell, direction, next_cell in cell_transitions:
            self._link_cells(transitions, cell, direction, next_cell, next_cell == terminal_cell)
        return transitions

    def _generate_separate_cell(self, cell, allow_clockwise_action=False):
        """ Generates transitions as if the cell was the only one."""
        transitions = dict()
        for view_direction in range(self._n_of_cell_edges):
            s0 = self._state(cell, view_direction)
            if s0 not in transitions:
                transitions[s0] = dict()

            # forward move keeps you in the same state
            transitions[s0][0] = s0
            # turn counter-clockwise
            transitions[s0][1] = self._state(cell, view_direction + 1)
            # [optional] turn clockwise
            if allow_clockwise_action:
                transitions[s0][2] = self._state(cell, view_direction - 1 + self._n_of_cell_edges)
        return transitions

    def _link_cells(self, transitions, c0, view_direction, c1, c1_is_terminal):
        s0 = self._state(c0, view_direction)
        if c1_is_terminal:
            # direction doesn't matter, also only forward link
            s1 = self._state(c1, 0)
            transitions[s0][0] = s1
        else:
            # forward link
            s1 = self._state(c1, view_direction)
            transitions[s0][0] = s1

            # backward link
            back_view_direction = self._back_view_direction(view_direction)
            s0 = self._state(c0, back_view_direction)
            s1 = self._state(c1, back_view_direction)
            transitions[s1][0] = s0

    def _state(self, cell, view_direction):
        return self._n_of_cell_edges * cell + (view_direction % self._n_of_cell_edges)

    def _back_view_direction(self, view_direction):
        return (view_direction + self._n_of_cell_edges // 2) % self._n_of_cell_edges


class SarSuperpositionFormatter:
    @staticmethod
    def format(sar: SarSuperposition) -> str:
        return '  '.join(
            '.'.join(map(str, superposition))
            for superposition in sar
        )
