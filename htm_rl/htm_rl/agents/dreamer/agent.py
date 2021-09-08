from typing import Tuple, Optional

import numpy as np

from htm_rl.agents.dreamer.dreaming_double import DreamingDouble
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_sum


class DreamerAgent(QModelBasedAgent):
    td_error_decay: float
    cum_td_error: float
    force_dreaming: bool
    dreamer: DreamingDouble

    prediction_depth: int
    n_prediction_rollouts: Tuple[int, int]
    dream_length: Optional[int]

    def __init__(
            self,
            seed: int,
            td_error_decay: float,
            dreaming: dict,
            **qmb_agent_config
    ):
        super(DreamerAgent, self).__init__(seed=seed, **qmb_agent_config)

        self.td_error_decay = td_error_decay
        self.cum_td_error = 0.
        self.dreamer = DreamingDouble(seed, self, **dreaming)
        self.force_dreaming = False

    @property
    def name(self):
        return 'dreamer'

    def on_new_episode(self):
        super(DreamerAgent, self).on_new_episode()
        if self.train:
            self.dreamer.on_new_episode()

    def act(self, reward: float, state: SparseSdr, first: bool):
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        im_reward = self._get_im_reward()

        prev_sa_sdr = self._current_sa_sdr
        s = self.sa_encoder.encode_state(state, learn=True and train)
        actions_sa_sdr = self.sa_encoder.encode_actions(s, learn=True and train)

        if train and not first:
            self.reward_model.update(s, reward)
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward+im_reward,
                next_actions_sa_sdr=actions_sa_sdr
            )
            self._try_dreaming(
                prev_sa=prev_sa_sdr, r=reward+im_reward, s=s,
                next_actions_sa_sdr=actions_sa_sdr, eval_r=reward
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]
        if train:
            self.transition_model.process(
                self.transition_model.preprocess(s, chosen_sa_sdr),
                learn=True and train
            )
        if train and self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        self._step += 1
        return action

    def _try_dreaming(self, prev_sa, r, s, next_actions_sa_sdr, eval_r):
        if self.force_dreaming:
            dream = True
            self.force_dreaming = False
        elif self.dreamer.can_dream(eval_r):
            action_values = self.Q.values(next_actions_sa_sdr)
            greedy_action = np.argmax(action_values)
            greedy_sa_sdr = next_actions_sa_sdr[greedy_action]
            td_error = self.Q.td_error(prev_sa, r, greedy_sa_sdr)

            self.cum_td_error = exp_sum(self.cum_td_error, self.td_error_decay, td_error)
            dream = self.dreamer.decide_to_dream(self.cum_td_error)
        else:
            dream = False

        if dream:
            # print('-- dream --')
            self.dreamer.dream(s, prev_sa)
