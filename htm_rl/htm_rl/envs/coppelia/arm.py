from pyrep.robots.arms.arm import Arm


class Pulse75(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'pulse75', 6)
