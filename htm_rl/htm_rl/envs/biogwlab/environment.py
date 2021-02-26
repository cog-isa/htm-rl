from typing import Dict, List

from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.common.utils import clip, isnone
from htm_rl.envs.biogwlab.environment_state import EnvironmentState
from htm_rl.envs.biogwlab.generation.env_generator import EnvironmentGenerator


class BioGwLabEnvironment:
    supported_actions = ['stay', 'move left', 'move up', 'move right', 'move down']

    state: EnvironmentState

    _initial_state: EnvironmentState
    _actions: List[str]
    _episode_max_steps: int
    _generator: EnvironmentGenerator
    _move_dynamics: 'MoveDynamics'

    def __init__(
            self,
            generator: Dict,
            state_encoder: Dict,
            move_dynamics: Dict,
            food_reward: float,
            seed: int,
            episode_max_steps: int = None,
            actions: List[str] = None,
    ):
        if actions is not None:
            self.ensure_all_actions_supported(actions)
        else:
            actions = self.supported_actions

        self._actions = actions
        self._generator = EnvironmentGenerator(seed=seed, **generator)
        self._move_dynamics = MoveDynamics(**move_dynamics)
        self._food_reward = food_reward

        self._initial_state = self._generator.generate()
        self.state = self._initial_state.make_copy()
        self._state_encoder = IntBucketEncoder(n_values=self.state.n_cells, **state_encoder)

        self._episode_max_steps = isnone(episode_max_steps, 2 * self.state.n_cells)

    def observe(self):
        reward = self.state.reward
        obs = self._get_agent_observation()
        is_first = self.state.episode_step == 0
        return reward, obs, is_first

    def act(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        if self._is_terminal():
            self._reset()
            return

        state = self.state
        action = self._actions[action]
        if action == 'stay':
            state.agent_position, success, reward = self._move_dynamics.stay(state)
        else:
            direction = action[5:]  # cut "move "
            direction = state.directions[direction]
            state.agent_position, success, reward = self._move_dynamics.move(state, direction)

        reward += self._collect()
        self.state.reward = reward
        self.state.episode_step += 1

    def _collect(self):
        state = self.state
        i, j = state.agent_position
        reward = 0
        if state.food_mask[i, j]:
            state.food_mask[i, j] = False
            state.n_foods -= 1
            reward += self._food_reward
        return reward

    def _get_agent_observation(self):
        position = self.state.agent_position
        position_fl = position[0] * self.state.shape[1] + position[1]
        encoded_position = self._state_encoder.encode(position_fl)
        return encoded_position

    def _is_terminal(self):
        """ Returns True if position is terminal or False if it isn't """
        steps_exceed = self.state.episode_step >= self._episode_max_steps
        no_foods = self.state.n_foods <= 0
        return steps_exceed or no_foods

    @property
    def n_actions(self):
        return len(self._actions)

    @property
    def output_sdr_size(self):
        return self._state_encoder.output_sdr_size

    def _reset(self):
        """ Resets environment."""
        self.state = self._initial_state.make_copy()

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in cls.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'


class MoveDynamics:
    action_cost: float
    action_weight: Dict[str, float]

    def __init__(self, action_cost: float, action_weight: Dict[str, float]):
        self.action_cost = action_cost
        self.action_weight = action_weight

    def stay(self, state):
        success = True
        reward = self.action_weight['stay'] * self.action_cost
        return state.agent_position, success, reward

    def move(self, state, direction):
        new_position, success = self._try_move(state, direction)
        reward = self.action_weight['move'] * self.action_cost
        return new_position, success, reward

    @staticmethod
    def _try_move(state: EnvironmentState, move_direction):
        i, j = state.agent_position

        i += move_direction[0]
        j += move_direction[1]

        i = clip(i, state.shape[0])
        j = clip(j, state.shape[1])
        success = (i, j) != state.agent_position and not state.obstacle_mask[i, j]
        new_position = (i, j)
        if not success:
            new_position = state.agent_position
        return new_position, success
