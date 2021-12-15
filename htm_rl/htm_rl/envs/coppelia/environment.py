from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.shape import Shape
from htm_rl.envs.coppelia.arm import Pulse75

from os.path import dirname, join, abspath
from typing import Union
import numpy as np

POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]


class PulseEnv:
    def __init__(self,
                 scene_file: str,
                 joints_to_manage: list[int],
                 observation: list[str],
                 max_steps: int,
                 action_time_step: float,
                 simulation_time_step: float,
                 action_cost: float,
                 goal_reward: float,
                 position_threshold: float,
                 initial_pose: list[tuple[float, float]] = None,
                 initial_target_position: list[int] = None,
                 joints_speed_limit: float = None,
                 camera_resolution: list[int] = None,
                 headless=False,
                 responsive_ui=True,
                 seed=None):
        self.action_time_step = action_time_step
        self.simulation_time_step = simulation_time_step
        self.pr = PyRep()
        self.pr.launch(scene_file, headless=headless, responsive_ui=responsive_ui)
        self.pr.set_simulation_timestep(simulation_time_step)
        self.pr.start()
        self.agent = Pulse75()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent.set_joint_intervals([True]*6, [[-np.pi, np.pi]]*6)
        self.target = Shape('target')
        self.tip = ForceSensor('pulse75_connection')

        if initial_pose is not None:
            self.initial_joint_positions = initial_pose
        else:
            self.initial_joint_positions = self.agent.get_joint_positions()

        if initial_target_position is not None:
            self.initial_target_position = initial_target_position
        else:
            self.initial_target_position = self.target.get_position()

        self.camera = VisionSensor('camera')
        if camera_resolution is not None:
            self.camera.set_resolution(camera_resolution)

        self.observation = set(observation)
        self.is_first = True
        self.should_reset = False
        self.n_sim_steps_for_action = int(action_time_step / simulation_time_step)
        self.action_cost = action_cost
        self.goal_reward = goal_reward
        self.position_threshold = position_threshold
        self.joints_speed_limit = np.pi*joints_speed_limit/180
        self.max_steps = max_steps
        self.n_steps = 0

        joints_mask = np.zeros(self.agent.get_joint_count(), dtype=bool)
        joints_mask[joints_to_manage] = True
        self.joints_to_manage = joints_mask
        self.n_joints = int(sum(joints_mask))

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.pr.stop()
        self.target.set_position(self.initial_target_position)
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.is_first = True
        self.should_reset = False
        self.n_steps = 0
        self.pr.start()

    def act(self, action: Union[list[float], np.ndarray]):
        self.n_steps += 1
        self.is_first = False

        target_positions = np.zeros(self.agent.get_joint_count())
        target_positions[self.joints_to_manage] = np.array(action)

        for i, joint in enumerate(self.agent.joints):
            if self.joints_to_manage[i]:
                joint.set_joint_target_position(target_positions[i])
            else:
                joint.set_joint_target_velocity(0.0)

        for step in range(self.n_sim_steps_for_action):
            self.pr.step()

    def observe(self):
        if self.should_reset:
            self.reset()
        obs = list()
        if 'camera' in self.observation:
            obs.append(self.camera.capture_rgb())
        if 'joint_pos' in self.observation:
            obs.append(self.get_joint_positions())
        if 'joint_vel' in self.observation:
            obs.append(self.get_joint_velocities())
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
            self.should_reset = True
        elif self.n_steps > self.max_steps:
            self.should_reset = True

        return reward, obs, self.is_first

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_joint_positions(self):
        joint_pos = np.array(self.agent.get_joint_positions())
        return joint_pos[self.joints_to_manage]

    def get_joint_velocities(self):
        joint_vel = np.array(self.agent.get_joint_velocities())
        return joint_vel[self.joints_to_manage]

    def get_joints_speed_limit(self):
        return self.joints_speed_limit


if __name__ == '__main__':
    EPISODES = 0
    EPISODE_LENGTH = 10
    EPS = 0.01
    SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/main_scene.ttt')

    env = PulseEnv(SCENE_FILE,
                   joints_to_manage=[3, 4],
                   observation=['joint_vel'],
                   n_sim_steps_for_action=10,
                   action_cost=-0.1,
                   goal_reward=1,
                   position_threshold=EPS,
                   headless=True)
    print('joint intervals', env.agent.get_joint_intervals())
    print('joint velocity limits', env.agent.get_joint_upper_velocity_limits())
    print('joint count', env.agent.get_joint_count())
    print('camera resolution', env.camera.get_resolution())
    print('tip coords', env.tip.get_position())
    print('robot pose', env.agent.get_joint_positions())

    for e in range(EPISODES):
        print('Starting episode %d' % e)
        env.reset()
        for i in range(EPISODE_LENGTH):
            print(f'Step {i}')
            action = list(np.random.uniform(-1.0, 1.0, size=(2,)))
            env.act(action)
            state = env.observe()
            print(f'pos {env.tip.get_position()}')
            print(f'vel {env.agent.get_joint_velocities()}')
            # save images from camera
            # plt.imshow(state)
            # plt.savefig(join(dirname(abspath(__file__)), f'image_{e}_{i}.png'))
    print('Done!')
    env.shutdown()
