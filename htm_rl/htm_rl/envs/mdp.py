from gym.utils import seeding


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
        self.np_random, _ = seeding.np_random(seed)
        self._current_state = self._get_initial_state()

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

    def get_all_states(self):
        """ return a tuple of all possible states """
        return tuple(range(self.n_states))

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return self.transitions[state] is None

    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        return 1 if self.is_terminal(next_state) else 0

    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = self._get_initial_state()
        return self._current_state

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(self._current_state), 'Episode is finished!'

        next_state = self.transitions[self._current_state][action]
        assert next_state is not None

        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        return next_state, reward, is_done, {}

    def render(self):
        state = self._current_state
        print(f'State: {state}')


def generate_gridworld_mdp(initial_state, cell_transitions, add_clockwise_action=False, seed=None):
    """
    Generates MDP base on a grid world with 2 [or 3] allowed actions:
        - move forward
        - turn counter-clockwise
        - [optional] you can allow 3rd action - turn clockwise

    By convenience directions: 0 - right, 1 - up, 2 - left, 3 - down
        , i.e. [0, 3] counter-clockwise.
    :param initial_state: can be tuple (cell, direction)
    :param cell_transitions: List[(cell, direction, next_cell)]
    :param add_clockwise_action:  by default only counter-clockwise turn is allowed
    :param seed: optional seed for a random generator
    :return: Gym's environment-like object
    """
    transitions = _generate_transitions(cell_transitions, add_clockwise_action)

    cell, direction = initial_state
    initial_state = cell * 4 + direction

    return Mdp(transitions, initial_state, seed)


def _generate_transitions(cell_transitions, add_clockwise_action=False):
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

    n_states = n_full_cells * 4 + 1  # n_cells full cells + one cell w/ only one terminal state
    terminal_state = n_states - 1
    transitions = {terminal_state: None}
    for cell in range(n_full_cells):
        transitions.update(_generate_separate_cell(cell, add_clockwise_action))

    for cell, direction, next_cell in cell_transitions:
        _link_cells(transitions, cell, direction, next_cell, next_cell == terminal_cell)
    return transitions


def _generate_separate_cell(cell, add_clockwise_action=False):
    """ Generates transitions as if the cell was the only one."""
    transitions = dict()
    base = cell * 4
    for i in range(4):
        s0 = base + i
        if s0 not in transitions:
            transitions[s0] = dict()
        transitions[s0][0] = s0                     # forward move keeps you in the same state
        transitions[s0][1] = base + ((i + 1) % 4)   # counter-clockwise turn
        if add_clockwise_action:                    # optional clockwise turn
            transitions[s0][2] = base + ((i - 1 + 4) % 4)
    return transitions


def _link_cells(transitions, c0, direction, c1, c1_is_terminal):
    # forward
    s0 = c0 * 4 + direction
    if c1_is_terminal:
        s1 = c1 * 4
        transitions[s0][0] = s1
        return

    s1 = c1 * 4 + direction
    transitions[s0][0] = s1

    # backward
    s0 = c0 * 4 + ((direction + 2) % 4)
    s1 = c1 * 4 + ((direction + 2) % 4)
    transitions[s1][0] = s0


def _link_to_terminal_cell(transitions, c0, direction, c1):
    # forward
    s0 = c0 * 4 + direction

