import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.q.sa_encoders import SpSaEncoder
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.qmb.transition_models import make_transition_model
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator
from htm_rl.common.utils import exp_decay, isnone, DecayingValue
from htm_rl.htm_plugins.spatial_pooler import SpatialPoolerWrapper
from htm_rl.modules.empowerment import Memory


class DreamerSaEncoder(SpSaEncoder):
    sa_sp: None

    state_clusters: Memory
    state_decoder: list[SparseSdr]

    # noinspection PyMissingConstructor
    def __init__(
            self, state_encoder, action_encoder: IntBucketEncoder,
            clusters_similarity_threshold: float
    ):
        self.state_encoder = SpatialPoolerWrapper(state_encoder)
        self.state_clusters = Memory(
            size=self.state_encoder.output_sdr_size,
            threshold=clusters_similarity_threshold
        )
        self.state_decoder = []

        self.action_encoder = action_encoder
        self.s_a_concatenator = SdrConcatenator(input_sources=[
            self.state_encoder,
            self.action_encoder
        ])
        self.sa_sp = None  # it isn't needed for a Dreamer

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        if not isinstance(state, np.ndarray):
            state = np.array(list(state))
        s = super(DreamerSaEncoder, self).encode_state(state, learn=learn)

        self._add_to_decoder(state, s)
        return s

    def encode_action(self, action: int, learn: bool) -> SparseSdr:
        return super(DreamerSaEncoder, self).encode_action(action, learn=learn)

    def decode_s_to_state(self, s: SparseSdr) -> SparseSdr:
        similarity_with_clusters = self.state_clusters.similarity(s)
        i_state_cluster = np.argmax(similarity_with_clusters)
        return self.state_decoder[i_state_cluster]

    @property
    def output_sdr_size(self):
        return self.s_a_concatenator.output_sdr_size

    def _add_to_decoder(self, state: SparseSdr, s: SparseSdr):
        similarity_with_clusters = self.state_clusters.similarity(s)
        i_state_cluster = np.argmax(similarity_with_clusters)
        if i_state_cluster < len(self.state_decoder):
            assert np.all(state == self.state_decoder[i_state_cluster])
        else:
            self.state_decoder.append(state.copy())


class Dreamer:
    n_actions: int
    sa_encoder: DreamerSaEncoder

    train: bool

    # TM learns s, a -> s'
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
        self.sa_encoder = DreamerSaEncoder(
            state_encoder, action_encoder, **sa_encoder
        )

        self.transition_model = make_transition_model(
            sa_encoder=self.sa_encoder,
            transition_model_config=transition_model
        )
        self.tm_checkpoint = None
        self.reward_model = RewardModel(
            self.sa_encoder.output_sdr_size, **reward_model
        )

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

    def on_wake_step(self, state, action, reward):
        s = self.sa_encoder.encode_state(state, learn=False)
        self.reward_model.update(s, reward)

        s_a = self.sa_encoder.encode_s_a(s, action)
        self.transition_model.process(s, learn=True)
        self.transition_model.process(s_a, learn=False)

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

        starting_s = self.sa_encoder.encode_state(starting_state, learn=False)
        starting_s_len = len(starting_s)
        i_rollout = 0
        sum_depth = 0
        depths = []
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth >= 2.2 * i_rollout
        ):
            self._on_new_rollout(i_rollout)
            state, s = starting_state, starting_s
            depth = 0
            for depth in range(self.prediction_depth):
                if len(s) < .6 * starting_s_len:
                    break
                state, s, a = self._move_in_dream(state, s)

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

    def _move_in_dream(self, state: SparseSdr, s: SparseSdr):
        action = self.agent.make_action(state)
        reward = self.reward_model.rewards[s].mean()
        self.agent.reinforce(reward)

        if reward > .2:
            # found goal ==> should stop rollout
            next_state, s_next = [], []
            return next_state, s_next, action

        s_a = self.sa_encoder.encode_s_action(s, action, learn=False)
        self.transition_model.process(s, learn=False)
        _, s_next_cells = self.transition_model.process(s_a, learn=False)
        s_next = self.transition_model.columns_from_cells(s_next_cells)

        next_state = self.sa_encoder.decode_s_to_state(s_next)
        return next_state, s_next, action

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
