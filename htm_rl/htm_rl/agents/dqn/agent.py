import numpy as np
import torch
from htm.bindings.sdr import SDR

from htm_rl.agents.dqn.config import Config
from htm_rl.agents.dqn.network_bodies import FCBody
from htm_rl.agents.dqn.network_heads import VanillaNet
from htm_rl.agents.dqn.replay import UniformReplay, PrioritizedTransition
from htm_rl.agents.dqn.schedule import LinearSchedule
from htm_rl.agents.dqn.utils import tensor, to_np
from htm_rl.common.utils import softmax


class DqnAgent:
    def __init__(self, config):
        self.config = config
        self.replay = config.replay_fn()
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
            'state': self._state,
            'action': self._action,
            'reward': reward,
            'mask': 1 - is_first
        })
        self._total_steps += 1

        self.train_step()
        self._state = next_state

    def flush_replay(self):
        self.replay = self.config.replay_fn()

    def train_step(self):
        if (
                self.replay.size() < 2 * self.replay.batch_size
                or self._total_steps % self.config.train_schedule != 0
        ):
            return

        transitions = self.replay.sample()

        loss = self.compute_loss(transitions)
        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        states = transitions.state
        next_states = transitions.next_state
        with torch.no_grad():
            q_next = self.network(next_states)['q'].detach().max(1)[0]

        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + self.config.discount * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def to_dense(self, sparse_sdr):
        self._state_sdr.sparse = sparse_sdr
        return self._state_sdr.dense


def make_agent(_config):
    config = Config()
    config.merge(_config)
    config.replay_cls = UniformReplay
    # config.replay_cls = PrioritizedReplay

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.learning_rate)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.history_length = 1

    replay_kwargs = dict(
        memory_size=config.replay_buffer_size,
        batch_size=config.batch_size,
        n_step=1,
        discount=config.discount,
        history_length=config.history_length
    )

    config.replay_fn = lambda: config.replay_cls(**replay_kwargs)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.55, 0.6, 1e5)

    config.gradient_clip = 5
    agent = DqnAgent(config)
    return agent
