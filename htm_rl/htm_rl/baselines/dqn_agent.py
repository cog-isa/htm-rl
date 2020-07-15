import numpy as np
import torch
import torch.nn as nn

from htm_rl.common.utils import trace


class DqnAgent:
    def __init__(self, n_states, n_actions, epsilon, gamma, lr):
        self.qn = DqnAgentNetwork((n_states, ), n_actions)
        self.opt = torch.optim.Adam(self.qn.parameters(), lr=lr)

        self.epsilon = epsilon
        self.gamma = gamma
        self._n_states = n_states
        self._n_actions = n_actions
        self._last_state = None
        self._last_action = None

    def reset(self):
        pass

    def make_step(self, state, reward, is_done, verbose):
        if self._last_state is not None:
            s, a, r, ns = self._last_state, self._last_action, reward, state
            self._train(s, a, r, ns, is_done)

        trace(verbose, f'\nState: {state}; reward: {reward}')
        action = self._make_action(state)
        trace(verbose, f'\nMake action: {action}')

        self._last_state = state
        self._last_action = action

        return action

    def _train(self, s, a, r, ns, is_done):
        gamma = self.gamma
        # encode input to 1-hot
        s, ns = self._to_one_hot(s), self._to_one_hot(ns)
        # to torch tensors
        s, r, ns = tuple(map(self._to_tensor, (s, r, ns)))
        a = self._to_tensor(a, torch.int64)
        is_done = self._to_tensor(is_done, torch.bool)

        Q_s = self.qn(s)
        Q_sa = torch.index_select(Q_s, dim=1, index=a)
        Q_ns = self.qn(ns)

        # compute V*(next_states) using predicted next q-values
        V_ns, _ = torch.max(Q_ns, dim=1)
        assert V_ns.dtype == torch.float32

        # TD target
        TD_target = r + gamma * V_ns
        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        TD_target = torch.where(is_done, r, TD_target)

        # MSE
        loss = (Q_sa - TD_target.detach()) ** 2
        loss = torch.mean(loss)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

    @staticmethod
    def _to_tensor(x, dtype=torch.float32):
        # also to batch
        return torch.tensor([x], dtype=dtype)

    def _make_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self._n_actions)

        s = self._to_one_hot(state)
        qvalues = self.qn.get_qvalues([s])
        return np.argmax(qvalues)

    def _to_one_hot(self, state):
        one_hot_state = np.zeros(self._n_states, dtype=np.float)
        one_hot_state[state] = 1.
        return one_hot_state


class DqnAgentNetwork(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape
        self._device = torch.device('cpu')
        assert len(state_shape) == 1

        input_dim, output_dim = state_shape[0], n_actions
        layers = [input_dim, 64, 32, output_dim]
        self.network = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3])
        )

    def forward(self, state):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.network(state)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(qvalues.shape) == 2 \
            and qvalues.shape[0] == state.shape[0] \
            and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        states = torch.tensor(states, device=self._device, dtype=torch.float32)
        qvalues = self.forward(states)
        # or .cpu().numpy()?
        return qvalues.data.detach().numpy()
