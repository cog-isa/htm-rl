from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from arm import Pulse75
import numpy as np
import matplotlib.pyplot as plt

SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/main_scene.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 2
EPISODE_LENGTH = 10
EPS = 0.01


class SimEnv:

    def __init__(self, scene_file: str):
        self.pr = PyRep()
        self.pr.launch(scene_file, headless=False)
        self.pr.start()
        self.agent = Pulse75()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.camera = VisionSensor('camera')

    def reset(self) -> np.ndarray:
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        return self.camera.capture_rgb()

    def step(self, action: list[float]) -> np.ndarray:
        self.agent.set_joint_target_positions(action)
        done = False
        while not done:
            self.pr.step()
            current_pos = self.agent.get_joint_positions()
            done = True
            for i, a in enumerate(action):
                if np.abs(a - current_pos[i]) > EPS:
                    done = False
        return self.camera.capture_rgb()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


if __name__ == '__main__':
    env = SimEnv(SCENE_FILE)
    for e in range(EPISODES):
        print('Starting episode %d' % e)
        state = env.reset()
        for i in range(EPISODE_LENGTH):
            print(f'Step {i}')
            action = list(np.random.uniform(-1.0, 1.0, size=(6,)))
            state = env.step(action)
            # save images from camera
            # plt.imshow(state)
            # plt.savefig(join(dirname(abspath(__file__)), f'image_{e}_{i}.png'))

    print('Done!')
    env.shutdown()
