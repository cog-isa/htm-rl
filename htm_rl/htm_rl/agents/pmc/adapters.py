from math import sin, cos, radians
from animalai.envs.actions import AAIActions
import numpy as np


class AAIActionAdapter:
    def __init__(self):
        self.actions = AAIActions().allActions
        self.n_actions = len(self.actions)

    def adapt(self, action):
        """
        :param action: logits
        :return:
        """
        action = self.actions[np.argmax(action)]
        return action.action_tuple


class ArmActionAdapter:
    def __init__(self, limits):
        self.limits = limits

    def adapt(self, action):
        r = (self.limits['r'][0] +
             action[0]*(self.limits['r'][1] -
                        self.limits['r'][0]))
        phi = (self.limits['phi'][0] +
               action[1]*(self.limits['phi'][1] -
                          self.limits['phi'][0]))
        h = (self.limits['h'][0] +
             action[2]*(self.limits['h'][1] -
                        self.limits['h'][0]))

        phi = radians(phi)
        x = r * cos(phi)
        y = r * sin(phi)
        z = h
        return x, y, z
