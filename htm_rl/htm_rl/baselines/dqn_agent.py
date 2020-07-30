from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.common.utils import trace, timed


class DqnAgent:
    def __init__(self, n_states, n_actions, epsilon, gamma, learning_rate, seed):
        torch.manual_seed(seed)

        self._n_states = n_states
        self._n_actions = n_actions

        self.qn = DqnAgentNetwork((n_states, ), n_actions)
        self._optimizer = torch.optim.Adam(self.qn.parameters(), lr=learning_rate)

        self._epsilon = epsilon
        self._gamma = gamma
        self._train_mode = True

    def reset(self, train_mode=None):
        if train_mode is not None:
            self._train_mode = train_mode
            # sets train or eval (inference) mode
            self.qn.train(train_mode)

    def make_action(self, state):
        if self._train_mode and np.random.rand() < self._epsilon:
            return np.random.choice(self._n_actions)

        s = self._to_one_hot(state)
        qvalues = self.qn.get_qvalues([s])
        return np.argmax(qvalues)

    def train(self, s, a, r, ns, is_done):
        if not self._train_mode:
            return

        gamma = self._gamma
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
        self._optimizer.step()
        self._optimizer.zero_grad()

    @staticmethod
    def _to_tensor(x, dtype=torch.float32):
        # also to batch
        return torch.tensor([x], dtype=dtype)

    def _to_one_hot(self, state):
        one_hot_state = np.zeros(self._n_states, dtype=np.float)
        one_hot_state[state] = 1.
        return one_hot_state


class DqnAgentRunner:
    agent: DqnAgent
    env: Any
    n_episodes: int
    max_steps: int
    verbosity: int
    train_stats: RunStats
    test_stats: RunStats

    def __init__(self, agent, env, n_episodes, max_steps, verbosity):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.train_stats = RunStats(n_episodes)
        self.test_stats = RunStats(n_episodes)

    def run(self):
        trace(self.verbosity, 1, '============> RUN DQN AGENT')
        for _ in trange(self.n_episodes):
            for train_mode, stats in zip([True, False], [self.train_stats, self.test_stats]):
                (steps, reward), elapsed_time = self.run_episode(train_mode=train_mode)
                stats.append_stats(steps, reward, elapsed_time)
                trace(self.verbosity, 2, '')
        trace(self.verbosity, 1, '<============')

    def print_q_values(self):
        self.agent.reset(False)
        with torch.set_grad_enabled(False):
            for s in range(self.agent._n_states):
                a = self.agent.make_action(s)
                print(s, a)

    @timed
    def run_episode(self, train_mode: bool):
        verbosity = self.verbosity

        self.agent.reset(train_mode)
        with torch.set_grad_enabled(train_mode):
            state, step, done = self.env.reset(), 0, False
            trace(verbosity, 2, f'\nState: {state}')

            while step < self.max_steps and not done:
                action = self.agent.make_action(state)
                trace(verbosity, 2, f'\nMake action: {action}')

                next_state, reward, done, info = self.env.step(action)
                self.agent.train(state, action, reward, next_state, done)

                state = next_state
                trace(verbosity, 2, f'\nState: {state}; reward: {reward}')
                step += 1

            return step, reward

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, 'dqn_eps')
        run_results_processor.store_result(self.test_stats, 'dqn_greedy')


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
        :param state: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.network(state)

        if self.training:
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
        return qvalues.data.detach().numpy()
