#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import torch.nn as nn

from htm_rl.common.utils import isnone


class FCBody(nn.Module):
    # orig 64, 64
    def __init__(self, state_dim, hidden_units, w_scale, gates=None, enable_sparse_init=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + tuple(hidden_units)
        self.layers = nn.ModuleList([
            layer_init(
                nn.Linear(dim_in, dim_out),
                'sparse' if dim_in > 100 and enable_sparse_init else 'xavier',
                w_scale
            )
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])

        gates = isnone(gates, [None]*len(hidden_units))
        self.gates = [self._gate(gate, torch.relu) for gate in gates[:len(hidden_units)]]
        self.feature_dim = dims[-1]

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer, gate in zip(self.layers, self.gates):
            x = gate(layer(x))
        return x

    def _gate(self, gate, default_gate):
        if gate is None:
            return default_gate
        elif gate == 'relu':
            return torch.relu
        elif gate == 'tanh':
            return torch.tanh
        else:
            raise KeyError(f'{gate} is unsupported activation function')


def layer_init(layer, init_type='xavier', w_scale=1e-2):
    if init_type == 'xavier':
        nn.init.xavier_uniform_(layer.weight.data, w_scale)
    elif init_type == 'sparse':
        nn.init.sparse_(layer.weight.data, .5, w_scale)
    else:
        nn.init.constant_(layer.weight.data, w_scale)

    nn.init.constant_(layer.bias.data, w_scale)
    return layer
