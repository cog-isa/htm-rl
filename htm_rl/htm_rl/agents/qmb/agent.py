from typing import Optional

from htm_rl.agents.q.agent import QAgent
from htm_rl.agents.qmb.anomaly_model import AnomalyModel
from htm_rl.agents.qmb.reward_model import RewardModel
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.agents.qmb.transition_models import make_transition_model
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay, DecayingValue


class QModelBasedAgent(QAgent):
    transition_model: TransitionModel
    reward_model: RewardModel
    anomaly_model: AnomalyModel

    im_weight: DecayingValue

    def __init__(
            self,
            reward_model: dict,
            transition_model: dict,
            anomaly_model: dict,
            im_weight: DecayingValue,
            **q_agent_config,
    ):
        super(QModelBasedAgent, self).__init__(**q_agent_config)

        self.transition_model = make_transition_model(
            self.sa_encoder, transition_model
        )
        self.reward_model = RewardModel(self.sa_encoder.output_sdr_size, **reward_model)
        self.anomaly_model = AnomalyModel(
            cells_sdr_size=self.sa_encoder.s_output_sdr_size,
            n_actions=self.n_actions,
            **anomaly_model
        )
        self.im_weight = im_weight

    @property
    def name(self):
        return 'qmb'

    def on_new_episode(self):
        super(QModelBasedAgent, self).on_new_episode()
        if self.train:
            self.transition_model.reset()
            self.reward_model.decay_learning_factors()
            self.anomaly_model.decay_learning_factors()
            self.im_weight = exp_decay(self.im_weight)

    def act(self, reward: float, state: SparseSdr, first: bool) -> Optional[int]:
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        im_reward = self._get_im_reward()

        prev_sa_sdr = self._current_sa_sdr
        s = self.sa_encoder.encode_state(state, learn=True and train)
        actions_sa_sdr = self._encode_s_actions(s, learn=True and train)

        if train and not first:
            self.reward_model.update(s, reward)
            self.E_traces.update(prev_sa_sdr)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward+im_reward,
                next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]
        if train:
            self._update_transition_model(s, action, learn=True and train)
        if train and self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        self._step += 1
        return action

    def _get_im_reward(self) -> float:
        if self.train and self.im_weight[0] > 0.:
            x = (1 - self.transition_model.recall) ** 2
            return self.im_weight[0] * x
        return 0.

    def _update_transition_model(self, s: SparseSdr, action: int, learn: bool):
        # learn transition and anomaly for (s,a) -> s'
        self.transition_model.process(s, learn=learn)
        anomaly = self.transition_model.recall
        self.anomaly_model.update(action, s, anomaly)

        # activate (s',a')
        s_a = self.sa_encoder.concat_s_action(s, action, learn=False)
        self.transition_model.process(s_a, learn=False)
