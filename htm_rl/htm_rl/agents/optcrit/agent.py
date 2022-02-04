#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from htm.bindings.sdr import SDR
from torch import nn

from htm_rl.agents.optcrit.config import Config
from htm_rl.agents.dqn.network import to_np, FCBody
from htm_rl.agents.optcrit.network import OptionCriticNet
from htm_rl.agents.dqn.replay import DequeStorage
from htm_rl.common.utils import softmax


class OptionCriticAgent:
    def __init__(self, config: Config):
        self.config = config
        self.network = config.network_fn()
        self.cr_optimizer = config.cr_optimizer_fn(self.network.parameters())
        self.ac_optimizer = config.ac_optimizer_fn(self.network.parameters())

        self._is_first = None
        self._action = None
        self._option = None
        self._state = None
        self._state_sdr = SDR(self.config.state_dim)
        self.replay = DequeStorage(config.cr_batch_size)
        self._total_steps = 0

        self._rng = np.random.default_rng(self.config.seed)

    def act(self):
        with torch.no_grad():
            prediction = self.network(self._state)

        option = self.sample_option(
            q_options=to_np(prediction['q']),
            beta_options=to_np(prediction['beta'])
        )

        probs = to_np(prediction['pi'][option])
        if self.config.ac_eps_greedy is not None and self.config.ac_eps_greedy > .0:
            # eps-soft: at-least eps prob for each action
            eps = self.config.ac_eps_greedy / self.config.action_dim
            probs = np.clip(probs, a_min=eps, a_max=1.)
            probs /= probs.sum()

        action = self._rng.choice(self.config.action_dim, p=probs)

        self._action = action
        self._option = option
        return action

    def observe(self, next_state, reward, is_first):
        next_state = self.to_dense(next_state)
        if self._state is None:
            self._state = next_state
            return

        s, o, a, r, s_next = self._state, self._option, self._action, reward, next_state
        self.replay.feed({'s': s, 'o': o, 'a': a, 'r': r, 's_next': s_next, 'done': is_first})
        self._total_steps += 1

        self.train_critic_step()
        self.train_actor_step()

        self._state = next_state

    def train_actor_step(self):
        replay_buffer_filled = self.replay.is_full or self.replay.sub_size >= self.config.ac_train_schedule
        train_scheduled = self._total_steps % self.config.ac_train_schedule == 0

        if not (replay_buffer_filled and train_scheduled):
            return

        # uniformly sample a batch of transitions
        batch_size = min(self.config.ac_train_schedule, self.replay.sub_size)
        i_batch = -np.arange(batch_size) + self.replay.sub_size - 1
        i_batch = i_batch[::-1]
        batch = self.replay.extract(
            keys=['s', 'o', 'a', 'r', 's_next', 'done'],
            indices=i_batch
        )

        self.ac_optimizer.zero_grad()
        loss = self.compute_actor_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.ac_optimizer.step()

    def train_critic_step(self):
        replay_buffer_filled = self.replay.is_full or self.replay.sub_size >= 4 * self.config.cr_batch_size
        train_scheduled = self._total_steps % self.config.cr_train_schedule == 0

        if not (replay_buffer_filled and train_scheduled):
            return

        # uniformly sample a batch of transitions
        i_batch = self._rng.choice(self.replay.sub_size, self.config.cr_batch_size)
        batch = self.replay.extract(
            keys=['s', 'o', 'a', 'r', 's_next', 'done'],
            indices=i_batch
        )

        self.cr_optimizer.zero_grad()
        loss = self.compute_critic_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.cr_optimizer.step()

    def compute_actor_loss(self, batch):
        s, o, a, r, s_n, done = batch.s, batch.o, batch.a, batch.r, batch.s_next, batch.done
        # [batch_size, 1]
        o = o.long().unsqueeze(-1)
        a = a.long().unsqueeze(-1)
        # [batch_size, 1, n_actions]
        o_a = o.unsqueeze(-1).expand(*o.size(), self.config.action_dim)
        gamma = self.config.discount

        pred = self.network(s)
        pi_o = pred['pi'].gather(1, o_a).squeeze(1)
        log_pi_o = pred['log_pi'].gather(1, o_a).squeeze(1)
        beta_s_o = pred['beta'].gather(1, o).squeeze(-1)

        with torch.no_grad():
            Q_s = pred['q']
            Q_s_o = Q_s.gather(1, o).squeeze(-1)
            V_s = Q_s.max(-1)[0]
            beta_advantage = Q_s_o - V_s + self.config.termination_regularizer

        beta_loss = beta_s_o * beta_advantage

        with torch.no_grad():
            pred_next = self.network(s_n)
            beta_sn_o = pred_next['beta'].gather(1, o).squeeze(-1)
            Q_sn_o = pred_next['q'].gather(1, o).squeeze(-1)
            V_sn = pred_next['q'].max(-1)[0]

            U_sn_o = (1 - beta_sn_o) * Q_sn_o + beta_sn_o * V_sn

            Q_s_o_a = r + gamma * U_sn_o * (1 - done)
            advantage = Q_s_o_a - Q_s_o
            entropy = -(pi_o * log_pi_o).sum(-1)

        w_entropy = self.config.entropy_weight
        log_pi_o_a = log_pi_o.gather(1, a).squeeze(-1)

        pi_loss = -(log_pi_o_a * advantage + w_entropy * entropy)

        return (pi_loss + beta_loss).mean()

    def compute_critic_loss(self, batch):
        s, o, a, r, s_n, done = batch.s, batch.o, batch.a, batch.r, batch.s_next, batch.done
        o = o.long().unsqueeze(-1)
        gamma = self.config.discount

        with torch.no_grad():
            pred_next = self.network(s_n)

        beta_sn_o = pred_next['beta'].gather(1, o).squeeze(-1)
        Q_sn_o = pred_next['q'].gather(1, o).squeeze(-1)
        V_sn = pred_next['q'].max(-1)[0]

        U_sn_o = (1 - beta_sn_o) * Q_sn_o + beta_sn_o * V_sn
        td_target = r + gamma * U_sn_o * (1 - done)

        Q_s_o = self.network(s)['q'].gather(1, o).squeeze(-1)
        td_error = Q_s_o - td_target

        return td_error.pow(2).mul(.5).mean()

    def flush_replay(self):
        self.replay.reset()

    def sample_option(self, q_options: np.ndarray, beta_options: np.ndarray):
        if self._option is not None:
            if self._rng.random() >= beta_options[self._option]:
                # do not terminate the current option
                return self._option

        # have to select new option
        probs = softmax(q_options, self.config.cr_softmax_temp)

        if self.config.cr_eps_greedy is not None and self.config.cr_eps_greedy > 0:
            # eps-soft: at-least eps prob for each action
            eps = self.config.cr_eps_greedy / self.config.action_dim
            probs = np.clip(probs, a_min=eps, a_max=1.)
            probs /= probs.sum()

        option = self._rng.choice(self.config.num_options, p=probs)
        return option

    def to_dense(self, sparse_sdr):
        self._state_sdr.sparse = sparse_sdr
        return self._state_sdr.dense


def make_agent(_config):
    config = Config()
    config.merge(_config)

    # Sanitize int config values because they might come as floats during sweep hyperparam search!
    if config.cr_replay_buffer_size is not None:
        config.cr_replay_buffer_size = int(config.cr_replay_buffer_size)

    config.cr_optimizer_fn = lambda params: torch.optim.RMSprop(
        params, config.cr_learning_rate, weight_decay=config.w_regularization
    )
    config.ac_optimizer_fn = lambda params: torch.optim.RMSprop(
        params, config.ac_learning_rate, weight_decay=config.w_regularization
    )
    config.network_fn = lambda: OptionCriticNet(
        FCBody(
            config.state_dim,
            hidden_units=tuple(config.hidden_units),
            gates=config.hidden_act_f,
            w_scale=config.w_scale
        ),
        config.action_dim,
        num_options=config.num_options,
        softmax_temp=config.ac_softmax_temp
    )
    agent = OptionCriticAgent(config)
    return agent
