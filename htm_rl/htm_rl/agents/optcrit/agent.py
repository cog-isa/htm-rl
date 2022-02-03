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
            eps = self.config.ac_eps_greedy
            probs = np.clip(probs, a_min=eps, a_max=1.)
            probs /= probs.sum()

        action = self._rng.choice(self.config.action_dim, p=probs)

        self._action = action
        self._option = option
        return action

    def observe(self, next_state, reward, is_first):
        next_state = self.to_dense(next_state)
        if is_first:
            self._state = next_state
            self._option = None
            return

        s, o, a, r, s_next = self._state, self._option, self._action, reward, next_state
        self.replay.feed({'s': s, 'o': o, 'a': a, 'r': r, 's_next': s_next})
        self._total_steps += 1

        self.train_critic_step()
        self.train_actor_step(s, o, a, r, s_next)

        # self._is_first = is_first
        self._state = next_state

        # self.train_step()
        #
        # self._storage.feed({
        #     'prev_option': self._option,
        #     'reward': reward,
        #     'mask': 1 - is_first,
        #     'ret': None,
        #     'advantage': None,
        #     'beta_advantage': None,
        # })

    def train_step(self):
        storage = self.replay
        if not storage.is_full:
            return

        with torch.no_grad():
            prediction = self.network(self._state)
            q_options = prediction['q']
            betas = prediction['beta'][self._option]
            ret = (1 - betas) * q_options[self._option] + betas * torch.max(q_options)

        config = self.config
        for i in reversed(range(config.batch_size)):
            r, q = storage['reward'][i], storage['q'][i]
            ret = r + config.discount * storage['mask'][i] * ret
            adv = ret - q[storage['option'][i]]
            storage['ret'][i] = ret
            storage['advantage'][i] = adv

            probs = torch.softmax(q, dim=-1)
            v = q.dot(probs)
            q_s = q[storage['prev_option'][i]]
            storage['beta_advantage'][i] = q_s - v + config.termination_regularizer

        entries = storage.extract(
            ['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage', 'entropy', 'option', 'action', 'init_state', 'prev_option']
        )

        q_loss = (entries.q.gather(0, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(
            entries.log_pi.gather(0, entries.action) * entries.advantage.detach()
            + config.entropy_weight * entries.entropy
        )
        pi_loss = pi_loss.mean()
        beta_loss = (
                entries.beta.gather(0, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)
        ).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
        self.replay.reset()

    def train_actor_step(self, s, o, a, r, s_n):
        self.ac_optimizer.zero_grad()
        pi_loss, beta_loss = self.compute_actor_loss(s, o, a, r, s_n)

        (pi_loss + beta_loss).backward()
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
            keys=['s', 'o', 'a', 'r', 's_next'],
            indices=i_batch
        )

        self.cr_optimizer.zero_grad()
        loss = self.compute_critic_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.cr_optimizer.step()

    def compute_actor_loss(self, s, o, a, r, s_n):
        pred = self.network(s)
        pi = pred['pi'][o]
        log_pi = pred['log_pi'][o]
        beta_s_o = pred['beta'][o]

        with torch.no_grad():
            Q_s = pred['q']
            Q_s_o = Q_s[o]
            V_s = Q_s.max()
            beta_advantage = Q_s_o - V_s + self.config.termination_regularizer

        beta_loss = beta_s_o * beta_advantage

        gamma = self.config.discount
        with torch.no_grad():
            pred_next = self.network(s_n)
            beta_sn_o = pred_next['beta'][o]
            Q_sn_o = pred_next['q'][o]
            V_sn = pred_next['q'].max()

            U_sn_o = (1 - beta_sn_o) * Q_sn_o + beta_sn_o * V_sn

            Q_s_o_a = r + gamma * U_sn_o
            advantage = Q_s_o_a - Q_s_o
            entropy = -(pi * log_pi).sum()

        w_entropy = self.config.entropy_weight
        pi_loss = -log_pi[a] * advantage - w_entropy * entropy

        return pi_loss, beta_loss

    def compute_critic_loss(self, batch):
        s, o, a, r, s_n = batch.s, batch.o, batch.a, batch.r, batch.s_next
        o = o.long().unsqueeze(-1)
        gamma = self.config.discount

        with torch.no_grad():
            pred_next = self.network(s_n)

        beta_sn_o = pred_next['beta'].gather(1, o).squeeze()
        Q_sn_o = pred_next['q'].gather(1, o).squeeze()
        V_sn = pred_next['q'].max(-1)[0]

        U_sn_o = (1 - beta_sn_o) * Q_sn_o + beta_sn_o * V_sn
        td_target = r + gamma * U_sn_o

        Q_s_o = self.network(s)['q'].gather(1, o).squeeze()
        td_error = Q_s_o - td_target

        # advantage and beta advantage
        return td_error.pow(2).mean() / 2

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
            eps = self.config.cr_eps_greedy
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
    config.gradient_clip = 5
    agent = OptionCriticAgent(config)
    return agent
