from abc import abstractmethod

from htm_rl.common.sar_sdr_encoder import SarSuperposition
from htm_rl.envs.mdp import Mdp


class AbstractPomdp(Mdp):
    def __init__(self, transitions, initial_state=None, seed=None):
        """
        Defines a POMDP. Compatible with gym Env.
        Instead of a full state an agent 'sees' only some observation. What exactly is that observation
        decides each non-abstract implementation of this class.

        I.e. this class provides only a basic skeleton of transformation MDP into POMDP and leaves
        the method `_observation` to be overridden.

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

        super().__init__(transitions, initial_state, seed)

    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = self._get_initial_state()
        return self._observation(self._current_state)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(self._current_state), 'Episode is finished!'

        next_state = self.transitions[self._current_state][action]
        assert next_state is not None
        self._current_state = next_state

        observation = self._observation(next_state)
        reward = self._reward(next_state)
        is_done = self.is_terminal(next_state)
        return observation, reward, is_done, {}

    def render(self):
        observation = self._observation(self._current_state)
        print(f'Observation: {observation}')

    @abstractmethod
    def _observation(self, state):
        ...


class ForwardCellPomdp(AbstractPomdp):
    ForwardMove = 0

    def __init__(self, transitions, initial_state=None, seed=None):
        """
        Defines a POMDP. Compatible with gym Env.
        Instead of a full state an agent 'sees' only a next cell, i.e. a next state if it moves forward.

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

        super().__init__(transitions, initial_state, seed)

    def _observation(self, state):
        if self.is_terminal(state):
            return state
        else:
            return self.transitions[state][self.ForwardMove]


class NumberOfForwardCellsPomdp(AbstractPomdp):
    ForwardMove = 0

    def __init__(self, transitions, initial_state=None, seed=None):
        """
        Defines a POMDP. Compatible with gym Env.
        Instead of a full state an agent 'sees' only a number of cells before an agent,
        i.e. a number of cells an agent can pass if he moves forward.

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

        super().__init__(transitions, initial_state, seed)

    def _observation(self, state):
        n_cells_before_agent = 0
        while not (self.is_terminal(state) or self.transitions[state][self.ForwardMove] == state):
            state = self.transitions[state][self.ForwardMove]
            n_cells_before_agent += 1
        return n_cells_before_agent


class SarSuperpositionFormatter:
    @staticmethod
    def format(sar: SarSuperposition) -> str:
        return '  '.join(
            '.'.join(map(str, superposition))
            for superposition in sar
        )
