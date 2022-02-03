import torch
from torch import nn as nn

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network import to_tensor, layer_init


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
        phi = self.body(to_tensor(x))
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