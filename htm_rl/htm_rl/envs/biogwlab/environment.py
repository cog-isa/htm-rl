from typing import Dict, List

from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.environment_state import EnvironmentState


class BioGwLabEnvironment:

    output_sdr_size: int
    state: EnvironmentState

    _initial_state: EnvironmentState
    _actions: List[str]
    _episode_max_steps: int

    def __init__(
            self,
            environment: Dict,
            episode_max_steps: int = None,
            actions: List[str] = None,
    ):
        self._actions = isnone(actions, EnvironmentState.supported_actions.copy())
        self.ensure_all_actions_supported(self._actions)

        self._initial_state = EnvironmentState.make(**environment)

        self.state = self._initial_state.make_copy()
        self._episode_max_steps = isnone(episode_max_steps, 2 * self.state.n_cells)

    def observe(self):
        return self.state.observe()

    def act(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        if self._is_terminal():
            self._reset()
            return

        action = self._actions[action]
        self.state.act(action)

    def _is_terminal(self):
        """ Returns True if position is terminal or False if it isn't """
        steps_exceed = self.state.episode_step >= self._episode_max_steps
        return steps_exceed or self.state.is_terminal()

    @property
    def n_actions(self):
        return len(self._actions)

    @property
    def output_sdr_size(self):
        return self.state.output_sdr_size

    def _reset(self):
        """ Resets environment."""
        self.state = self._initial_state.make_copy()

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in EnvironmentState.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'

    @classmethod
    def _induce_if_agent_view_enabled(cls, actions):
        agent_view_dependent_actions = {'move forward', 'turn left', 'turn right'}
        return any(agent_view_dependent_actions & set(actions))
