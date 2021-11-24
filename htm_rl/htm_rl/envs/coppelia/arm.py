from pyrep.robots.robot_component import RobotComponent


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
