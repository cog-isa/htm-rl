from typing import Dict, List

from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.environment_state import EnvironmentState


class BioGwLabEnvironment:
    supported_actions = [
        'stay',
        'move left', 'move up', 'move right', 'move down',
        'move forward', 'turn left', 'turn right'
    ]

    output_sdr_size: int
    state: EnvironmentState

    _initial_state: EnvironmentState
    _actions: List[str]
    _episode_max_steps: int

    def __init__(
            self,
            environment: Dict,
            state_encoder: Dict,
            episode_max_steps: int = None,
            actions: List[str] = None,
    ):
        if actions is not None:
            self.ensure_all_actions_supported(actions)
        else:
            actions = self.supported_actions

        self._actions = actions
        agent_view_enabled = self.is_agent_view_enabled(self._actions)

        self._initial_state = EnvironmentState.make(
            agent_view_enabled=agent_view_enabled,
            **environment
        )
        self.state = self._initial_state.make_copy()
        self._position_encoder = IntBucketEncoder(n_values=self.state.n_cells, **state_encoder)
        if agent_view_enabled:
            self._view_direction_encoder = IntBucketEncoder(n_values=4, **state_encoder)
            self._sdr_concatenator = SdrConcatenator([
                self._position_encoder, self._view_direction_encoder
            ])
            self.output_sdr_size = self._sdr_concatenator.output_sdr_size
        else:
            self.output_sdr_size = self._position_encoder.output_sdr_size

        self._episode_max_steps = isnone(episode_max_steps, 2 * self.state.n_cells)

    def observe(self):
        reward, obs, is_first = self.state.observe()
        if isinstance(obs, tuple):
            position_fl, view_direction = obs
            position = self._position_encoder.encode(position_fl)
            view_direction = self._view_direction_encoder.encode(view_direction)
            obs = self._sdr_concatenator.concatenate(position, view_direction)
        else:
            position_fl = obs
            position = self._position_encoder.encode(position_fl)
            obs = position

        return reward, obs, is_first

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

    def _reset(self):
        """ Resets environment."""
        self.state = self._initial_state.make_copy()

    @classmethod
    def is_agent_view_enabled(cls, actions):
        agent_view_dependent_actions = {'move forward', 'turn left', 'turn right'}
        return agent_view_dependent_actions & set(actions)

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in cls.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'


