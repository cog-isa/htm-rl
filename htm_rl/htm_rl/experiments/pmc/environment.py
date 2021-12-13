import numpy as np
from PIL import Image, ImageDraw


class ReachAndGrasp2D:
    def __init__(self,
                 goal_position,
                 grip_position,
                 grip_radius=0.07,
                 goal_radius=0.05,
                 max_speed=0.1,
                 max_acceleration=0.02,
                 action_cost=-0.01,
                 goal_reward=1,
                 grabbed_reward=0.1,
                 time_constant=1,
                 camera_resolution: tuple[int, int] = (128, 128)):
        self.camera_resolution = camera_resolution
        self.action_cost = action_cost
        self.goal_reward = goal_reward
        self.grabbed_reward = grabbed_reward
        self.max_acceleration = max_acceleration
        self.max_speed = max_speed
        self.time_constant = time_constant
        self.init_grip_position = grip_position
        self.max_grip_radius = grip_radius
        self.init_goal_position = goal_position
        self.goal_radius = goal_radius

        # state variables
        self.goal_position = np.array(self.init_goal_position)
        self.grip_position = np.array(self.init_grip_position)
        self.grip_radius = self.max_grip_radius
        self.distance_to_goal = np.linalg.norm(self.goal_position - self.grip_position)
        self.can_grab = self.check_gripper()
        self.goal_grabbed = False
        self.grip_radius_speed = np.zeros(1)
        self.grip_speed = np.zeros(2)

        self.target_grip_position = None
        self.target_grip_radius = None

    def act(self, action):
        self.target_grip_position = np.array(action[:2])
        self.target_grip_radius = action[2]

    def obs(self):
        return self.reward(), self.render_rgb()

    def reset(self):
        self.goal_position = np.array(self.init_goal_position)
        self.grip_position = np.array(self.init_grip_position)
        self.grip_radius = self.max_grip_radius
        self.distance_to_goal = np.linalg.norm(self.goal_position - self.grip_position)
        self.can_grab = self.check_gripper()
        self.goal_grabbed = False
        self.grip_radius_speed = np.zeros(1)
        self.grip_speed = np.zeros(2)

        self.target_grip_position = None
        self.target_grip_radius = None

    def reward(self):
        reward = -self.action_cost
        if self.can_grab:
            reward += self.goal_reward
        if self.goal_grabbed:
            reward += self.grabbed_reward

        return reward

    def check_gripper(self):
        if self.distance_to_goal < (self.grip_radius - self.goal_radius):
            return True
        else:
            return False

    def simulation_step(self):
        self.goal_grabbed = False

        self.grip_position, self.grip_speed = self.dynamics(
            self.grip_position,
            self.grip_speed,
            self.max_acceleration,
            self.target_grip_position,
            self.max_speed
        )

        # check possibility to grab
        self.distance_to_goal = np.linalg.norm(self.goal_position
                                               - self.grip_position)

        self.can_grab = self.check_gripper()

        self.grip_radius, self.grip_radius_speed = self.dynamics(
            self.grip_radius,
            self.grip_radius_speed,
            self.max_acceleration,
            self.target_grip_radius,
            self.max_speed
        )

        if self.can_grab:
            self.grip_radius = max(self.goal_radius, self.grip_radius)
            if self.grip_radius == self.goal_radius:
                self.grip_speed = np.zeros(1)
                self.goal_grabbed = True
        else:
            self.grip_radius = max(0, self.grip_radius)
            if self.grip_radius == 0:
                self.grip_speed = 0

    def dynamics(self, x, dx, ddx, x_target, max_dx):
        speed_delta = x_target - x
        dx += self.time_constant * ddx * speed_delta / np.linalg.norm(speed_delta)
        norm_speed = np.linalg.norm(dx)
        if norm_speed > max_dx:
            dx = max_dx * dx / norm_speed

        x += self.time_constant * dx
        return x, dx

    def render_rgb(self, show_goal=False):
        image = Image.new('RGB', self.camera_resolution, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        scale_factor = max(self.camera_resolution)

        goal_bbox = np.concatenate(
            [self.goal_position - self.goal_radius,
             self.goal_position + self.goal_radius
             ]).flatten()

        grip_bbox = np.concatenate(
            [self.grip_position - self.grip_radius,
             self.grip_position + self.grip_radius
             ]).flatten()
        if show_goal and (self.target_grip_position is not None):
            target_bbox = np.concatenate(
                [self.target_grip_position - self.target_grip_radius,
                 self.target_grip_position + self.target_grip_radius
                 ]).flatten()
            draw.ellipse(list(target_bbox*scale_factor),
                         outline=(50, 255, 0, 30),
                         width=max(1, int(0.02*scale_factor))
                         )

        draw.ellipse(list(goal_bbox*scale_factor),
                     outline=(0, 0, 0),
                     fill=(0, 255, 255),
                     width=0)
        draw.ellipse(list(grip_bbox*scale_factor),
                     outline=(0, 0, 255),
                     width=max(1, int(0.02*scale_factor)))

        return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = ReachAndGrasp2D(
        goal_position=(0.7, 0.7),
        grip_position=(0.1, 0.1),
    )
    env.act((0.5, 0.5, 0.05))
    plt.imshow(env.render_rgb(True))
    plt.show()
    env.simulation_step()
    plt.imshow(env.render_rgb(True))
    plt.show()
