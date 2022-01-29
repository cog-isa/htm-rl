#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from htm.bindings.sdr import SDR
from torch import nn

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network_bodies import FCBody
from htm_rl.agents.dqn.network_heads import OptionCriticNet
from htm_rl.agents.dqn.replay import Storage, DequeStorage


class OptionCriticAgent:
    def __init__(self, config):
        self.config = config
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self._is_first = None
        self._current_option = 0
        self._state = None
        self._state_sdr = SDR(self.config.state_dim)
        self._storage = DequeStorage(config.batch_size)

    def act(self):
        prediction = self.network(self._state)
        option = self.sample_option(prediction)

        # print(prediction['pi'])
        prediction['pi'] = prediction['pi'][option]
        prediction['log_pi'] = prediction['log_pi'][option]
        dist = torch.distributions.Categorical(probs=prediction['pi'])
        action = dist.sample()
        entropy = dist.entropy()

        self._storage.feed(prediction)
        self._storage.feed({
            'option': option,
            'entropy': entropy,
            'action': action,
            'init_state': self._is_first,
        })
        self._current_option = option
        return action

    def observe(self, next_state, reward, is_first):
        next_state = self.to_dense(next_state)

        self._is_first = is_first
        self._state = next_state

        self.train_step()

        self._storage.feed({
            'prev_option': self._current_option,
            'reward': reward,
            'mask': 1 - is_first,
            'ret': None,
            'advantage': None,
            'beta_advantage': None,
        })

    def train_step(self):
        storage = self._storage
        if not storage.is_full:
            return

        with torch.no_grad():
            prediction = self.network(self._state)
            q_options = prediction['q']
            betas = prediction['beta'][self._current_option]
            ret = (1 - betas) * q_options[self._current_option] + betas * torch.max(q_options)

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
        pi_loss = -(entries.log_pi.gather(0,
                                          entries.action) * entries.advantage.detach()) - config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()
        beta_loss = (
                entries.beta.gather(0, entries.prev_option) * entries.beta_advantage.detach() * (1 - entries.init_state)
        ).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
        self._storage.reset()

    def flush_replay(self):
        self._storage.reset()

    def sample_option(self, prediction):
        with torch.no_grad():
            q_option = prediction['q']
            # pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            # greedy_option = q_option.argmax(dim=-1, keepdim=True)
            # prob = 1 - epsilon + epsilon / q_option.size(1)
            # prob = torch.zeros_like(pi_option).add(prob)
            # pi_option.scatter_(1, greedy_option, prob)
            pi_option = torch.softmax(q_option/self.config.hl_softmax_temp, dim=-1)
            # pi_option = torch.softmax(q_option, dim=-1)

            if self._is_first:
                dist = torch.distributions.Categorical(probs=pi_option)
                option = dist.sample()
            else:
                mask = torch.zeros_like(q_option)
                mask[self._current_option] = 1
                beta = prediction['beta']
                pi_hat_option = (1 - beta) * mask + beta * pi_option
                dist = torch.distributions.Categorical(probs=pi_hat_option)
                option = dist.sample()

        return option

    def to_dense(self, sparse_sdr):
        self._state_sdr.sparse = sparse_sdr
        return self._state_sdr.dense


def make_agent(_config):
    config = Config()
    config.merge(_config)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.learning_rate)
    config.network_fn = lambda: OptionCriticNet(
        FCBody(config.state_dim, hidden_units=tuple(config.hidden_units)),
        config.action_dim,
        num_options=config.num_options,
        softmax_temp=config.softmax_temp
    )
    config.gradient_clip = 5
    agent = OptionCriticAgent(config)
    return agent
