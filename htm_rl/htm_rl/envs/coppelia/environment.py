from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape
from htm_rl.envs.coppelia.arm import Pulse75

from os.path import dirname, join, abspath
from typing import Union
import numpy as np

from htm_rl.envs.env import Env


class Environment(Env):
    def __init__(self,
        seed: int,
        scene_file: str,
        headless: bool,
        responsive_ui: bool,
        action_time_step: float,
        joints_to_manage: Union[list[int], str],
        reward_type: str,
        action_type: str,
        observation: list[str],
        action_cost: float,
        goal_reward: float,
        position_threshold: float,
        joints_speed_limit: float,
        max_steps: int,
        initial_pose: list[float] = None,
        initial_target_position: list[float] = None,
        camera_resolution: list[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self._env_info = dict()

        self.pr = PyRep()

        # Base simulation setup
        scene_file = join(dirname(abspath(__file__)), 'scenes', scene_file)
        self.pr.launch(scene_file, headless=headless, responsive_ui=responsive_ui)
        sim_time_step = self.pr.get_simulation_timestep()
        self.n_sim_steps_for_action = int(action_time_step / sim_time_step)
        if self.n_sim_steps_for_action == 0:
            raise ValueError('Action time step is too small')

        self.pr.start()

        # Add robot arm
        self.agent = Pulse75()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent.set_joint_intervals([True] * 6, [[-np.pi, np.pi]] * 6)
        self.joints_speed_limit = np.pi * joints_speed_limit / 180
        if initial_pose is not None:
            self.initial_joint_positions = initial_pose
            self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        else:
            self.initial_joint_positions = self.agent.get_joint_positions()

        if isinstance(joints_to_manage, list):
            if action_type == 'tip':
                raise NotImplementedError
            joints_mask = np.zeros(self.agent.get_joint_count(), dtype=bool)
            joints_mask[joints_to_manage] = True
        elif joints_to_manage == 'all':
            joints_mask = np.ones(self.agent.get_joint_count(), dtype=bool)
        else:
            raise ValueError
        self.joints_to_manage = joints_mask
        self.n_joints = int(sum(joints_mask))

        # Add target
        self.target = Shape('target')
        if initial_target_position is not None:
            self.initial_target_position = initial_target_position
            self.target.set_position(self.initial_target_position)
        else:
            self.initial_target_position = self.target.get_position()

        # Add camera
        self.camera = VisionSensor('camera')
        if camera_resolution is not None:
            self.camera.set_resolution(camera_resolution)

        # RL specific setup
        self.reward_type = reward_type
        self.action_type = action_type
        self.observation = set(observation)

        self.step_reward = 0
        self.action_cost = action_cost
        self.goal_reward = goal_reward
        self.position_threshold = position_threshold
        if self.reward_type == 'gaus_dist':
            r_2 = self.get_sq_dist_tip_target()
            d_2 = self.position_threshold ** 2
            self.step_reward = self.goal_reward * np.exp(-r_2 / d_2)

        self.max_steps = max_steps
        self.episode_step = 0

    def reset(self) -> None:
        self.target.set_position(self.initial_target_position)
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)

        self.episode_step = 0
        self.step_reward = 0
        if self.reward_type == 'gaus_dist':
            r_2 = self.get_sq_dist_tip_target()
            d_2 = self.position_threshold ** 2
            self.step_reward = self.goal_reward * np.exp(-r_2 / d_2)

    def observe(self) -> tuple[float, list, bool]:
        reward = self.step_reward

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

        is_first = self.episode_step == 0
        return reward, obs, is_first

    def act(self, action: Union[list[float], np.ndarray]) -> None:
        if self.is_terminal():
            self.reset()
            return

        if self.action_type == 'joints':
            target_positions = np.zeros(self.agent.get_joint_count())
            target_positions[self.joints_to_manage] = np.array(action)

            for i, joint in enumerate(self.agent.joints):
                if self.joints_to_manage[i]:
                    joint.set_joint_target_position(target_positions[i])
                else:
                    joint.set_joint_target_velocity(0.0)
        elif self.action_type == 'tip':
            raise NotImplementedError
        else:
            raise ValueError("Action type can be: 'joints' or 'tip'")

        for step in range(self.n_sim_steps_for_action):
            self.pr.step()

        self.step_reward = -self.action_cost
        r_2 = self.get_sq_dist_tip_target()
        d_2 = self.position_threshold ** 2
        if self.reward_type == 'sparse':
            if r_2 < d_2:
                self.step_reward += self.goal_reward
        elif self.reward_type == 'gaus_dist':
            self.step_reward += self.goal_reward * np.exp(-r_2/d_2)
        else:
            raise ValueError("Reward type can be: 'sparse or 'gaus_dist'")

        self.episode_step += 1

    @property
    def n_actions(self):
        return self.n_joints

    def get_joint_positions(self):
        joint_pos = np.array(self.agent.get_joint_positions())
        return joint_pos[self.joints_to_manage]

    def get_joint_velocities(self):
        joint_vel = np.array(self.agent.get_joint_velocities())
        return joint_vel[self.joints_to_manage]

    def get_sq_dist_tip_target(self) -> float:
        x, y, z = self.agent.get_tip().get_position()
        tx, ty, tz = self.target.get_position()
        square_dist = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        return square_dist

    def is_terminal(self):
        sq_dist = self.get_sq_dist_tip_target()
        tip_in_loc = sq_dist < self.position_threshold ** 2
        return tip_in_loc or self.episode_step >= self.max_steps

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class PulseEnv:
    def __init__(self,
                 scene_file: str,
                 joints_to_manage: Union[list[int], str],
                 observation: list[str],
                 max_steps: int,
                 action_time_step: float,
                 simulation_time_step: float,
                 action_cost: float,
                 goal_reward: float,
                 position_threshold: float,
                 initial_pose: list[float] = None,
                 initial_target_position: list[float] = None,
                 joints_speed_limit: float = 80,
                 camera_resolution: list[int] = None,
                 headless: bool = False,
                 responsive_ui: bool = True,
                 reward_type: str = 'sparse',
                 action_type: str = 'joints',
                 seed=None):
        self.action_time_step = action_time_step
        self.simulation_time_step = simulation_time_step
        self.pr = PyRep()
        scene_file = join(dirname(abspath(__file__)), 'scenes', scene_file)
        self.pr.launch(scene_file, headless=headless, responsive_ui=responsive_ui)
        self.pr.set_simulation_timestep(simulation_time_step)
        self.pr.start()
        self.agent = Pulse75()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent.set_joint_intervals([True]*6, [[-np.pi, np.pi]]*6)
        self.target = Shape('target')
        self.reward_type = reward_type
        self.action_type = action_type

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
        # should use all joints if you use IK
        # maybe we will remove this in future
        assert (action_type != 'tip') or (joints_to_manage == 'all')

        if isinstance(joints_to_manage, list):
            joints_mask = np.zeros(self.agent.get_joint_count(), dtype=bool)
            joints_mask[joints_to_manage] = True
        elif joints_to_manage == 'all':
            joints_mask = np.ones(self.agent.get_joint_count(), dtype=bool)
        else:
            raise ValueError

        self.joints_to_manage = joints_mask
        self.n_joints = int(sum(joints_mask))

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.target.set_position(self.initial_target_position)
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.is_first = True
        self.should_reset = False
        self.n_steps = 0

    def act(self, action: Union[list[float], np.ndarray]):
        self.n_steps += 1
        self.is_first = False

        if self.action_type == 'joints':
            target_positions = np.zeros(self.agent.get_joint_count())
            target_positions[self.joints_to_manage] = np.array(action)

            for i, joint in enumerate(self.agent.joints):
                if self.joints_to_manage[i]:
                    joint.set_joint_target_position(target_positions[i])
                else:
                    joint.set_joint_target_velocity(0.0)

            for step in range(self.n_sim_steps_for_action):
                self.pr.step()
        elif self.action_type == 'tip':
            try:
                path = self.agent.get_path(
                    position=action, euler=[0, np.pi, 0])
            except ConfigurationPathError:
                for step in range(self.n_sim_steps_for_action):
                    self.pr.step()
                return False

            for step in range(self.n_sim_steps_for_action):
                path.step()
                self.pr.step()
        else:
            raise ValueError

        return True

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

        tip_in_loc = False
        x, y, z = self.agent.get_tip().get_position()
        tx, ty, tz = self.target.get_position()
        if ((abs(x - tx) < self.position_threshold) and
                (abs(y - ty) < self.position_threshold) and
                (abs(z - tz) < self.position_threshold)):
            tip_in_loc = True
            self.should_reset = True
        elif self.n_steps > self.max_steps:
            self.should_reset = True

        reward = -self.action_cost
        if self.reward_type == 'sparse':
            if tip_in_loc:
                reward += self.goal_reward
        elif self.reward_type == 'gaus_dist':
            r_2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
            d_2 = self.position_threshold ** 2
            reward += self.goal_reward * np.exp(-r_2/d_2)

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
    EPISODES = 1
    EPISODE_LENGTH = 1
    EPS = 0.01
    SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/main_scene.ttt')

    env = PulseEnv(SCENE_FILE,
                   joints_to_manage='all',
                   observation=['joint_pos'],
                   max_steps=200,
                   action_time_step=200,
                   simulation_time_step=10,
                   action_cost=-0.1,
                   goal_reward=1,
                   position_threshold=EPS,
                   action_type='tip',
                   initial_pose=[0.0, 0.0, 0.0, 0.0, 1.57, 0.0],  # initial robot joint positions
                   initial_target_position=[0.500, 0.2763, 1.85274],
                   headless=False)

    for e in range(EPISODES):
        print('Starting episode %d' % e)
        env.reset()
        for i in range(EPISODE_LENGTH):
            print(f'Step {i}')
            action = [0.500, 0.2763, 1.85274]
            env.act(action)
            state = env.observe()
            # save images from camera
            # plt.imshow(state)
            # plt.savefig(join(dirname(abspath(__file__)), f'image_{e}_{i}.png'))
    print('Done!')
    env.shutdown()
