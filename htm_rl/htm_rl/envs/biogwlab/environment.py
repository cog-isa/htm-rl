from typing import Dict, List

from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab.environment_state import EnvironmentState
from htm_rl.envs.biogwlab.generation.env_generator import EnvironmentGenerator


class BioGwLabEnvironment:
    supported_actions = {'stay', 'move left', 'move up', 'move right', 'move down'}

    state: EnvironmentState

    _init_state: EnvironmentState
    _actions: List[str]
    _episode_max_steps: int
    _generator: EnvironmentGenerator
    _move_dynamics: 'MoveDynamics'

    _episode_step: int

    def __init__(
            self, generator: Dict,
            episode_max_steps: int, seed: int,
            state_encoder: Dict,
            food_reward: float,
            move_dynamics: Dict,
            actions: List[str] = None,
    ):
        if actions is not None:
            self.ensure_all_actions_supported(actions)
        else:
            actions = list(self.supported_actions)

        self._actions = actions
        self._episode_max_steps = episode_max_steps
        self._generator = EnvironmentGenerator(seed=seed, **generator)
        self._move_dynamics = MoveDynamics(**move_dynamics)
        self._episode_step = 0
        self._food_reward = food_reward

        self._init_state = self._generator.generate()
        self.state = self._init_state.make_copy()
        self._state_encoder = IntBucketEncoder(n_values=self.state.n_cells, **state_encoder)

    @property
    def n_actions(self):
        return len(self._actions)

    @property
    def output_sdr_size(self):
        return self._state_encoder.output_sdr_size

    def reset(self):
        """ Resets environment and returns initial state."""
        self._episode_step = 0
        self.state = self._init_state.make_copy()
        return self._get_agent_state()

    def is_terminal(self):
        """ Returns True if position is terminal or False if it isn't """
        if self._episode_step >= self._episode_max_steps:
            return True
        return self.state.n_foods <= 0

    def act(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(), 'Episode is finished!'

        state = self.state
        action = self._actions[action]
        move_dynamics = self._move_dynamics
        if action == 'stay':
            new_position, success, reward = move_dynamics.stay(state)
        else:
            direction = action[5:]  # cut "move "
            direction = state.directions[direction]
            new_position, success, reward = move_dynamics.move(state, direction)

        state.agent_position = new_position
        reward += self._collect()
        self._episode_step += 1

        # form output
        agent_state = self._get_agent_state()
        is_done = self.is_terminal()
        return agent_state, reward, is_done, {}

    def _collect(self):
        state = self.state
        i, j = state.agent_position
        reward = 0
        if state.food_mask[i, j]:
            state.food_mask[i, j] = False
            state.n_foods -= 1
            reward += self._food_reward
        return reward

    def _get_agent_state(self):
        position = self.state.agent_position
        position_fl = position[0] * self.state.shape[1] + position[1]
        encoded_position = self._state_encoder.encode(position_fl)
        return encoded_position

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
