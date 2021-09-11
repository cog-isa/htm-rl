import pickle
from typing import Optional, List

import numpy as np
from numpy.random import Generator

from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import exp_decay, isnone, DecayingValue
from htm_rl.htm_plugins.spatial_pooler import SpatialPooler, SpatialPoolerWrapper


class Dreamer:
    n_actions: int
    sa_encoder: SaEncoder

    train: bool

    transition_model: TransitionModel
    tm_checkpoint: Optional[bytes]
    reward_model: RewardModel

    enabled: bool
    enter_prob_alpha: DecayingValue
    enter_prob_threshold: float
    prediction_depth: int
    n_prediction_rollouts: tuple[int, int]

    _episode: int
    _rng: Generator

    # noinspection PyMissingConstructor,PyPep8Naming
    def __init__(
            self,
            seed: int,
            n_actions: int,
            agent,
            state_encoder,
            action_encoder,
            sa_encoder: dict,
            transition_model: dict,
            reward_model: dict,
            enter_prob_alpha: DecayingValue,
            enter_prob_threshold: float,
            prediction_depth: int,
            n_prediction_rollouts: tuple[int, int],
            enabled: bool = True,
            first_dreaming_episode: int = None,
            last_dreaming_episode: int = None,
    ):
        self.n_actions = n_actions
        self.agent = agent
        self.sa_encoder = sa_encoder

        self.transition_model = transition_model
        self.tm_checkpoint = None
        self.reward_model = reward_model

        self.enabled = enabled
        self.enter_prob_alpha = enter_prob_alpha
        self.enter_prob_threshold = enter_prob_threshold
        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = isnone(first_dreaming_episode, 0)
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)

        self.train = True
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._episode = 0

    def on_wake_step(self, state, s, action, reward):
        s = self.sa_encoder.encode_state(state, learn=True)
        self.reward_model.update(s, reward)

        sa = self.sa_encoder.encode_sa(s, action, learn=True)
        self.transition_model.process(
            self.transition_model.preprocess(s, sa), learn=True
        )

    def on_new_episode(self):
        self._episode += 1
        if self.train:
            self.transition_model.reset()
            self.reward_model.decay_learning_factors()
            self.enter_prob_alpha = exp_decay(self.enter_prob_alpha)

    def can_dream(self, reward):
        if not self.enabled:
            return False
        if not (self.first_dreaming_episode <= self._episode < self.last_dreaming_episode):
            return False
        if reward > .2:
            # reward condition prevents useless planning when we've already
            # found the goal
            return False
        return True

    def decide_to_dream(self, td_error):
        if self._td_error_based_dreaming(td_error):
            return True
        return False

    def dream(self, starting_state):
        self._put_into_dream()

        starting_state_len = len(starting_state)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth >= 2.2 * i_rollout
        ):
            self._on_new_rollout(i_rollout)
            state = starting_state
            depth = 0
            for depth in range(self.prediction_depth):
                if len(state) < .6 * starting_state_len:
                    break
                state, a = self._move_in_dream(state)

            i_rollout += 1
            sum_depth += depth ** .6
            depths.append(depth)

        # if depths: print(depths)
        self._wake()

    def _put_into_dream(self):
        self._save_tm_checkpoint()

    def _on_new_rollout(self, i_rollout):
        if i_rollout > 0:
            self._restore_tm_checkpoint()

    def _wake(self):
        self._restore_tm_checkpoint()

    def _move_in_dream(self, state: SparseSdr):
        reward = self.reward_model.rewards[state].mean()
        a = self.agent.make_action(state)
        self.agent.reinforce(reward)

        if reward > .2:
            # found goal ==> should stop rollout
            next_s = []
            return next_s, a

        # encode (s,a)
        _, s_sa_next_superposition = self.transition_model.process(
            self._current_sa_sdr, learn=False
        )
        s_sa_next_superposition = self.transition_model.columns_from_cells(
            s_sa_next_superposition
        )
        next_s = self.sa_encoder.decode_state(s_sa_next_superposition)
        return next_s, a

    def _td_error_based_dreaming(self, td_error):
        max_abs_td_error = 2.
        dreaming_prob_boost = self.enter_prob_alpha[0]
        dreaming_prob = (dreaming_prob_boost * abs(td_error) - self.enter_prob_threshold)
        dreaming_prob = np.clip(dreaming_prob / max_abs_td_error, 0., 1.)
        return self._rng.random() < dreaming_prob

    def _save_tm_checkpoint(self) -> bytes:
        """Saves TM state."""
        self.tm_checkpoint = pickle.dumps(self.transition_model)
        return self.tm_checkpoint

    def _restore_tm_checkpoint(self, tm_checkpoint: bytes = None):
        """Restores saved TM state."""
        tm_checkpoint = isnone(tm_checkpoint, self.tm_checkpoint)
        self.transition_model = pickle.loads(tm_checkpoint)

    @property
    def name(self):
        return 'dreaming_double'


class DreamerSaEncoder:
    state_encoder: SpatialPooler
    state_decoder: list

    action_encoder: IntBucketEncoder
    sa_concatenator: SdrConcatenator

    def __init__(self, state_encoder, action_encoder):
        self.state_encoder = SpatialPoolerWrapper(state_encoder)
        self.state_decoder = []
        self.action_encoder = action_encoder
        self.sa_concatenator = SdrConcatenator(input_sources=[
            self.state_encoder,
            self.action_encoder
        ])

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        s = self.state_encoder.compute(state, learn=learn)
        self.state_decoder.append((state, s))
        return s

    def encode_action(self, action: int) -> SparseSdr:
        return self.action_encoder.encode(action)

    def encode_sa(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        pass

    def decode_state(self, s: SparseSdr) -> SparseSdr:
        pass

    @property
    def output_sdr_size(self):
        return self.sa_concatenator.output_sdr_size