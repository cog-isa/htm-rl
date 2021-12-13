from htm_rl.modules.v1 import V1
from .basal_ganglia import BasalGanglia
from .pmc import PMCToM1Basic


class BasicAgent:
    def __init__(self, config):
        self.v1 = V1(**config['v1'])
        self.bg = BasalGanglia(**config['bg'])
        self.pmc = PMCToM1Basic(**config['pmc'])

    def make_action(self, obs):
        stimulus = self.v1.compute(obs)
        response, probs = self.bg.compute(stimulus, learn=True)
        action = self.pmc.compute(response, probs)
        return action

    def reinforce(self, reward):
        self.bg.force_dopamine(reward)

    def reset(self):
        self.bg.reset()
