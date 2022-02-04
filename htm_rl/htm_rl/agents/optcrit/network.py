import torch
from torch import nn as nn

from htm_rl.agents.dqn.network import to_tensor, layer_init


class OptionCriticNet(nn.Module):
    def __init__(self, body, action_dim, num_options, softmax_temp):
        super(OptionCriticNet, self).__init__()
        self.q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.softmax_temp = softmax_temp
        self.body = body

    def forward(self, x):
        x = self.body(to_tensor(x))

        q = self.q(x)
        beta = torch.sigmoid(self.beta(x))
        pi = self.pi(x)
        pi = pi.view(-1, self.num_options, self.action_dim).squeeze(0)

        log_pi = torch.log_softmax(pi/self.softmax_temp, dim=-1)
        pi = torch.softmax(pi/self.softmax_temp, dim=-1)
        return {
            'q': q,
            'beta': beta,
            'log_pi': log_pi,
            'pi': pi
        }
