import pickle
from typing import Optional

import numpy as np
from numpy.random import Generator

from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.qmb.transition_models import make_transition_model
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay, isnone, DecayingValue
from htm_rl.modules.dreaming.sa_encoder import DreamerSaEncoder


class Dreamer:
    """
    Dreamer object, that can substitute with itself the environment for an agent,
    such that the agent <--> env interaction is replaced with the
    agent <--> imaginary env (learned transition + reward models).
    """
    n_actions: int
    sa_encoder: DreamerSaEncoder

    # working mode (turning it off disables learning rates decay)
    train: bool

    # TM learns s_a -> s'
    transition_model: TransitionModel
    tm_checkpoint: Optional[bytes]
    reward_model: RewardModel

    # turning it off disables possibility of dreaming
    enabled: bool
    # the probability of entering the dreaming mode
    enter_prob_alpha: DecayingValue

    # max prediction depth
    prediction_depth: int
    # min, max number of rollouts
    n_prediction_rollouts: tuple[int, int]

    _episode: int
    _rng: Generator

    def __init__(
            self,
            # --- specify this
            seed: int,
            n_actions: int,  # env.n_actions
            agent,           # HIMA agent
            state_encoder,   # state --> s encoder
            action_encoder,  # action --> a encoder
            # ----------------
            # --- from config
            sa_encoder: dict,
            transition_model: dict,
            reward_model: dict,
            enter_prob_alpha: DecayingValue,
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
        self.prediction_depth = prediction_depth
        self.n_prediction_rollouts = n_prediction_rollouts
        self.first_dreaming_episode = isnone(first_dreaming_episode, 0)
        self.last_dreaming_episode = isnone(last_dreaming_episode, 999999)

        self.train = True
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._episode = 0

    def on_wake_step(self, state: SparseSdr, reward: float, action: int):
        """
        Callback that should be called after each agent's step.
        It's used to learn transition and reward models.

        # TODO: you can pass already encoded state (s) and/or action (a)
        # I'd just need to adapt this code to remove unnecessary encoding steps.

        :param state: raw state (observation) sparse sdr
        :param reward: reward value given for getting to the state `state`
        :param action: action index
        """
        s = self.sa_encoder.encode_state(state, learn=False)
        self.reward_model.update(s, reward)

        s_a = self.sa_encoder.encode_s_action(s, action, learn=False)
        self.transition_model.process(s, learn=True)
        self.transition_model.process(s_a, learn=False)

    def on_new_episode(self):
        """
        Callback that should be called after the end of the episode.
        It's used to reset transition memory, increment episodes counter
        and decay some hyperparams.
        """
        self._episode += 1
        self.transition_model.reset()
        if self.train:
            self.reward_model.decay_learning_factors()
            self.enter_prob_alpha = exp_decay(self.enter_prob_alpha)

    def can_dream(self, reward: float) -> bool:
        """
        Checks whether the dreaming is possible.
        """
        if not self.enabled:
            return False
        if not (self.first_dreaming_episode <= self._episode < self.last_dreaming_episode):
            return False
        if reward > .2:
            # reward condition prevents useless planning when we've already
            # found the goal
            return False
        return True

    def decide_to_dream(self) -> bool:
        """
        Informs whether the dreaming should be activated.

        # TODO: add anomaly-based decision
        """
        if self._random_based_dreaming():
            return True
        return False

    def dream(self, starting_state: SparseSdr):
        """
        Switches to dreaming mode and performs several imaginary trajectory
        rollouts. Dreaming starts from `starting_state`, which is the current
        raw observation, for which the agent isn't acted yet.

        # TODO: you probably want to support callback that should be called
        # after each rollout to reset agent's state to the initial starting state
        """

        self._put_into_dream()

        starting_s = self.sa_encoder.encode_state(starting_state, learn=False)
        starting_s_len = len(starting_s)
        i_rollout = 0
        sum_depth = 0
        depths = []
        # loop over separate rollouts
        while i_rollout < self.n_prediction_rollouts[0] or (
                i_rollout < self.n_prediction_rollouts[1]
                and sum_depth >= 2.2 * i_rollout
        ):
            self._on_new_rollout(i_rollout)
            state, s = starting_state, starting_s
            depth = 0
            # loop over one rollout's trajectory states
            for depth in range(self.prediction_depth):
                if len(s) < .7 * starting_s_len:
                    # predicted pattern is too vague
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

    def _random_based_dreaming(self):
        dreaming_prob = self.enter_prob_alpha[0]
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
