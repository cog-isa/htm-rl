from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.shape import Shape
from arm import Pulse75
import numpy as np
import matplotlib.pyplot as plt

SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/main_scene.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 2
EPISODE_LENGTH = 10
EPS = 0.01


class PulseEnv:
    def __init__(self,
                 scene_file: str,
                 observation: list[str],
                 n_sim_steps_for_action: int,
                 action_cost: float,
                 goal_reward: float,
                 position_threshold: float,
                 change_position: bool,
                 headless=False,
                 seed=None):
        self.pr = PyRep()
        self.pr.launch(scene_file, headless=headless)
        self.pr.start()
        self.agent = Pulse75()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.tip = ForceSensor('pulse75_connection')
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.camera = VisionSensor('camera')
        self.observation = set(observation)
        self.is_first = True
        self.n_sim_steps_for_action = n_sim_steps_for_action
        self.action_cost = action_cost
        self.goal_reward = goal_reward
        self.position_threshold = position_threshold
        self.change_position = change_position
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        # Get a random position within a cuboid and set the target position
        if self.change_position:
            pos = list(self.rng.uniform(POS_MIN, POS_MAX))
            self.target.set_position(pos)

        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.is_first = True
        return self.camera.capture_rgb()

    def act(self, action: list[float]):
        self.is_first = False
        self.agent.set_joint_target_positions(action)
        for step in range(self.n_sim_steps_for_action):
            self.pr.step()

    def observe(self):
        obs = list()
        if 'camera' in self.observation:
            obs.append(self.camera.capture_rgb())
        if 'joint_pos' in self.observation:
            obs.append(self.agent.get_joint_positions())
        if 'joint_vel' in self.observation:
            obs.append(self.agent.get_joint_velocities())
        if 'target_pos' in self.observation:
            obs.append(self.target.get_position())
        if 'target_vel' in self.observation:
            obs.append(self.target.get_velocity())

        reward = self.action_cost
        x, y, z = self.tip.get_position()
        tx, ty, tz = self.target.get_position()
        if ((abs(x - tx) < self.position_threshold) and
                (abs(y - ty) < self.position_threshold) and
                (abs(z - tz) < self.position_threshold)):
            reward += self.goal_reward
            self.reset()

        return obs, reward, self.is_first

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


if __name__ == '__main__':
    env = PulseEnv(SCENE_FILE, ['force'], 2, -0.1, 1)
    for e in range(EPISODES):
        print('Starting episode %d' % e)
        env.reset()
        for i in range(EPISODE_LENGTH):
            print(f'Step {i}')
            action = list(np.random.uniform(-1.0, 1.0, size=(6,)))
            env.act(action)
            state = env.observe()
            print(f'pos {env.tip.get_position()}')
            # save images from camera
            # plt.imshow(state)
            # plt.savefig(join(dirname(abspath(__file__)), f'image_{e}_{i}.png'))
    print('Done!')
    env.shutdown()
