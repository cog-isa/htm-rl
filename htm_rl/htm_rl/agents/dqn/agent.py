import numpy as np
import torch
from htm.bindings.sdr import SDR

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network_bodies import FCBody
from htm_rl.agents.dqn.network_heads import VanillaNet
from htm_rl.agents.dqn.replay import UniformReplay, PrioritizedTransition, DequeStorage
from htm_rl.agents.dqn.schedule import LinearSchedule
from htm_rl.agents.dqn.utils import tensor, to_np
from htm_rl.common.utils import softmax


class DqnAgent:
    def __init__(self, config):
        self.config = config
        self.replay = DequeStorage(config.replay_buffer_size)
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self._rng = np.random.default_rng(self.config.seed)

        self._state_sdr = SDR(config.state_dim)
        self._total_steps = 0
        self._state = None
        self._action = None

    def act(self):
        prediction = self.network(self._state)
        q_values = to_np(prediction['q'])
        probs = softmax(q_values, self.config.softmax_temp)
        action = self._rng.choice(self.config.action_dim, p=probs)

        self._action = action
        return action

    def observe(self, next_state, reward, is_first):
        next_state = self.to_dense(next_state)
        if self._state is None:
            self._state = next_state
            return

        self.replay.feed({
            's': self._state,
            'a': self._action,
            'r': reward,
            's_next': next_state,
            'done': 1 - is_first,
        })
        self._total_steps += 1

        self.train_step()
        self._state = next_state

    def flush_replay(self):
        self.replay.reset()

    def train_step(self):
        replay_buffer_filled = self.replay.sub_size >= 2 * self.config.batch_size
        train_scheduled = self._total_steps % self.config.train_schedule == 0

        if not (replay_buffer_filled and train_scheduled):
            return

        # uniformly sample a batch of transitions
        i_batch = self._rng.choice(self.replay.sub_size, self.config.batch_size)
        batch = self.replay.extract(
            keys=['s', 'a', 'r', 's_next', 'done'],
            indices=i_batch
        )

        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def compute_loss(self, batch):
        s, a, r, s_n, done = batch.s, batch.a, batch.r, batch.s_next, batch.done
        gamma = self.config.discount

        with torch.no_grad():
            q_next = self.network(s_n)['q'].detach().max(1)[0]
            q_target = r + gamma * q_next * done

        a = a.long()
        q = self.network(s)['q'].gather(1, a.unsqueeze(-1)).squeeze(-1)
        td_error = q - q_target
        return td_error.pow(2).mean() / 2

    def to_dense(self, sparse_sdr):
        self._state_sdr.sparse = sparse_sdr
        return self._state_sdr.dense


def make_agent(_config):
    config = Config()
    config.merge(_config)

    # Sanitize int config values because they might come as floats during sweep hyperparam search!
    if config.replay_buffer_size is not None:
        config.replay_buffer_size = int(config.replay_buffer_size)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.learning_rate)
    config.network_fn = lambda: VanillaNet(
        config.action_dim,
        FCBody(config.state_dim, hidden_units=tuple(config.hidden_units), gates=config.hidden_act_f)
    )
    config.gradient_clip = 5

    agent = DqnAgent(config)
    return agent
