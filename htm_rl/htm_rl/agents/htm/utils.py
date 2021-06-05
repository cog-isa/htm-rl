import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi


class OptionVis:
    def __init__(self, map_size, size=21, max_options=50, action_displace=None, action_rotation=None):
        if action_displace is None:
            action_displace = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if action_rotation is None:
            action_rotation = [0]*len(action_displace)

        self.map_size = map_size
        self.size = size
        self.options = dict()
        self.initial_pos = (size//2, size//2)
        self.action_displace = np.array(action_displace, dtype=np.int32)
        self.action_rotation = np.array(action_rotation, dtype=np.int32)
        self.max_options = max_options

    def update(self, id_, start_pos, end_pos, actions, predicted_actions):
        id_ = int(id_)
        pos = self.initial_pos
        direction = 0  # North
        policy = np.zeros((self.size, self.size))

        for i, action in enumerate(actions):
            for p_action in predicted_actions[i]:
                p_direction = direction + self.action_rotation[p_action]
                if p_direction < 0:
                    p_direction = 4 - p_direction
                else:
                    p_direction %= 4
                p_displacement = self.transform_displacement(self.action_displace[p_action], p_direction)
                p_pos = (pos[0] + p_displacement[0], pos[1] + p_displacement[1])
                if self.is_in_bounds(p_pos):
                    policy[p_pos] += 1

            direction = direction + self.action_rotation[action]
            if direction < 0:
                direction = 4 - direction
            else:
                direction %= 4
            displacement = self.transform_displacement(self.action_displace[action], direction)
            pos = (pos[0] + displacement[0], pos[1] + displacement[1])

            if not self.is_in_bounds(pos):
                break

        if id_ not in self.options:
            self.options[id_] = dict()
            self.options[id_]['policy'] = np.zeros_like(policy)
            self.options[id_]['init'] = np.zeros(self.map_size)
            self.options[id_]['term'] = np.zeros(self.map_size)
            self.options[id_]['n_uses'] = 0

        self.options[id_]['policy'] += policy
        self.options[id_]['init'][start_pos] += 1
        self.options[id_]['term'][end_pos] += 1
        self.options[id_]['n_uses'] += 1

    def is_in_bounds(self, position):
        return (max(position) < self.size) and (min(position) >= 0)

    def draw_options(self, logger, episode, threshold=0, obstacle_mask=None):
        if len(self.options) == 0:
            return

        max_n_uses = max([value['n_uses'] for value in self.options.values()])
        height = max(self.size + 2, self.map_size[0])
        width = self.size + 2*self.map_size[1] + 3
        for key, value in self.options.items():
            if value['n_uses'] > threshold:
                color_shift = max_n_uses * 0.1
                image = np.zeros((height, width)) - color_shift
                # center marks
                image[0, self.size//2 + 1] = max_n_uses*0.5
                image[-1, self.size//2 + 1] = max_n_uses*0.5
                image[self.size//2 + 1, 0] = max_n_uses*0.5
                image[self.size//2 + 1, self.size+1] = max_n_uses*0.5
                image[1:self.size+1, 1:self.size+1] = value['policy']
                init_map = value['init']
                term_map = value['term']
                if obstacle_mask is not None:
                    init_map[obstacle_mask] = -color_shift
                    term_map[obstacle_mask] = -color_shift
                image[:self.map_size[0], self.size+2:self.size+2+self.map_size[1]] = value['init']
                image[:self.map_size[0], self.size+3+self.map_size[1]:] = value['term']
                plt.imsave(f'/tmp/option_{logger.run.id}_{episode}_{key}.png', image/max_n_uses, vmax=1, cmap='inferno')
                logger.log({f'option {key}': logger.Image(f'/tmp/option_{logger.run.id}_{episode}_{key}.png')}, step=episode)

    def clear_stats(self):
        self.options = dict()

    @staticmethod
    def transform_displacement(disp_xy, direction):
        """
        :param disp_xy:
        :param direction: 0 -- North, 1 -- West, 2 -- South, 3 -- East
        :return: tuple
        """

        def R(k):
            return [[cos(pi * k / 2), -sin(pi * k / 2)],
                    [sin(pi * k / 2), cos(pi * k / 2)]]

        return np.dot(disp_xy, R(direction)).astype('int32')



