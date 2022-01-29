#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network_utils import BaseNet, layer_init, NoisyLinear
from htm_rl.agents.dqn.utils import tensor


class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_head(phi)
        return dict(q=q)


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return dict(q=q)


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class RainbowNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body, noisy_linear):
        super(RainbowNet, self).__init__()
        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, num_atoms)
            self.fc_advantage = NoisyLinear(body.feature_dim, action_dim * num_atoms)
        else:
            self.fc_value = layer_init(nn.Linear(body.feature_dim, num_atoms))
            self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x):
        phi = self.body(tensor(x))
        value = self.fc_value(phi).view((-1, 1, self.num_atoms))
        advantage = self.fc_advantage(phi).view(-1, self.action_dim, self.num_atoms)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return dict(quantile=quantiles)


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options, softmax_temp):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.softmax_temp = softmax_temp
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        # theta = torch.softmax()
        beta = torch.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi).view(self.num_options, self.action_dim)
        log_pi = torch.log_softmax(pi/self.softmax_temp, dim=-1)
        pi = torch.softmax(pi/self.softmax_temp, dim=-1)
        return {
            'q': q,
            'beta': beta,
            'log_pi': log_pi,
            'pi': pi
        }


class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2
