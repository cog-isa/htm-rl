from htm_rl.modules.v1 import V1
from basal_ganglia import BasalGanglia
from pmc import ThaPMCToM1
import numpy as np


class BasicAgent:
    def __init__(self,
                 camera_resolution,
                 config):
        self.v1 = V1(camera_resolution,
                     config['v1']['complex'],
                     *config['v1']['simple'])
        config['bg']['input_size'] = self.v1.output_sdr_size
        self.bg = BasalGanglia(**config['bg'])
        self.pmc = ThaPMCToM1(**config['pmc'])
        self.probs = None
        self.response = None

    def make_action(self, obs):
        sparse, _ = self.v1.compute(np.array(obs))
        stimulus = np.concatenate(sparse)
        probs = self.bg.compute(stimulus, learn=True)
        action, response = self.pmc.compute(probs)
        self.probs = probs[response]
        self.response = response
        self.bg.update_stimulus(stimulus)
        self.bg.update_response(response)
        return action

    def reinforce(self, reward):
        self.bg.force_dopamine(reward)

    def reset(self):
        self.bg.reset()
