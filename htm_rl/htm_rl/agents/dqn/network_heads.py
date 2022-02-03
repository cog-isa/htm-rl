#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import torch.nn as nn

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network_bodies import layer_init
from htm_rl.agents.dqn.utils import tensor


class VanillaNet(nn.Module):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_head(phi)
        return dict(q=q)


class OptionCriticNet(nn.Module):
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
