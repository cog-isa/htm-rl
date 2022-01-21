from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import numpy as np


class Arm(RobotComponent):
    def __init__(self,
                 count: int, name: str, num_joints: int, base_name: str = None
                 ):
        """Count is used for when we have multiple copies of arms"""
        joint_names = ['%s_joint%d' % (name, i + 1) for i in range(num_joints)]
        super().__init__(count, name, joint_names, base_name)


class Pulse75(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'pulse75', 6)
        self.set_control_loop_enabled(True)
        self.set_motor_locked_at_zero_velocity(True)
        self.set_joint_intervals([True, True, False, True, True, True],
                                 [[-np.pi, np.pi],
                                  [-np.pi, np.pi],
                                  [-np.pi * 160 / 180, np.pi * 160 / 180],
                                  [-np.pi, np.pi],
                                  [-np.pi, np.pi],
                                  [-np.pi, np.pi]
                                  ])
        self.base = Dummy(f'pulse75_base')
        self.tip = Dummy(f'pulse75_tip')
        self.target = Shape(f'pulse75_target_visible')
