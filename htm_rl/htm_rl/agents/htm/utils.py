import numpy as np
import matplotlib.pyplot as plt


class OptionVis:
    def __init__(self, map_size, size=21, max_options=50, action_displace=None):
        if action_displace is None:
            action_displace = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        self.map_size = map_size
        self.size = size
        self.options = dict()
        self.initial_pos = (size//2, size//2)
        self.action_displace = np.array(action_displace, dtype=np.int32)
        self.max_options = max_options

    def update(self, id_, start_pos, end_pos, actions, predicted_actions):
        id_ = int(id_)
        pos = self.initial_pos
        policy = np.zeros((self.size, self.size))

        for i, action in enumerate(actions):
            for p_action in predicted_actions[i]:
                p_pos = (pos[0] + self.action_displace[p_action][0], pos[1] + self.action_displace[p_action][1])
                if self.is_in_bounds(p_pos):
                    policy[p_pos] += 1
            pos = (pos[0] + self.action_displace[action][0], pos[1] + self.action_displace[action][1])

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
        height = max(self.size, self.map_size[0])
        width = self.size + 2*self.map_size[1] + 2
        for key, value in self.options.items():
            if value['n_uses'] > threshold:
                image = np.zeros((height, width)) - 1
                image[:self.size, :self.size] = value['policy']
                init_map = value['init']
                term_map = value['term']
                if obstacle_mask is not None:
                    init_map[obstacle_mask] = -2
                    term_map[obstacle_mask] = -2
                image[:self.map_size[0], self.size+1:self.size+1+self.map_size[1]] = value['init']
                image[:self.map_size[0], self.size+2+self.map_size[1]:] = value['term']
                plt.imsave(f'/tmp/option_{logger.run.id}_{episode}_{key}.png', image/max_n_uses, vmax=1)
                logger.log({f'option {key}': logger.Image(f'/tmp/option_{logger.run.id}_{episode}_{key}.png')}, step=episode)

    def clear_stats(self):
        self.options = dict()



