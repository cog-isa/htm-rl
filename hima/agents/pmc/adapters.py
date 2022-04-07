from math import sin, cos, radians, copysign
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
    def __init__(self, limits: dict, velocity=None, environment=None):
        self.limits = limits
        self.velocity = velocity
        self.environment = environment

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
        if self.velocity is not None:
            c_x, c_y, c_z = self.environment.get_tip_position()
            x = c_x + copysign(min(self.velocity, abs(x - c_x)), x - c_x)
            y = c_y + copysign(min(self.velocity, abs(y - c_y)), y - c_y)
            z = c_z + copysign(min(self.velocity, abs(z - c_z)), z - c_z)
        return x, y, z
