import numpy as np
from htm.bindings.sdr import SDR

from htm_rl.agents.dqn.deps.dqn_agent import make_agent
from htm_rl.agents.htm.hierarchy import Hierarchy
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.modules.dreaming.dreamer import Dreamer
from htm_rl.modules.empowerment import Empowerment


class DqnAgentProxy:
    def __init__(self, config):
        obs_size = env.output_sdr_size
        n_actions = env.n_actions
        self.agent = make_agent(state_dim=obs_size, action_dim=n_actions)

        self.state_pattern = SDR(obs_size)

        # proportionality for random generator
        self.alpha = config['alpha']

        self.previous_obs = np.empty(0)
        self.backup = None
        self.real_pos = None

    def reset(self, obs):
        self.agent.reset(obs)



    def make_action(self, state_pattern):
        self.state_pattern.sparse = state_pattern
        action = -1
        return action

    def reinforce(self, reward: float):
        ...

    def reset(self):
        self.state_pattern.sparse = np.empty(0)
        self.previous_obs = np.empty(0)
