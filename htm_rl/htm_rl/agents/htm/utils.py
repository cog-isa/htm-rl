import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import cos, sin, pi
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import EntityType
from htm.bindings.sdr import SDR
from htm_rl.agents.htm.basal_ganglia import softmax


class OptionVis:
    def __init__(self, map_size, size=21, max_options=50, action_displace=None, action_rotation=None):
        if action_displace is None:
            action_displace = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if action_rotation is None:
            action_rotation = [0] * len(action_displace)

        self.map_size = map_size
        self.size = size
        self.options = dict()
        self.initial_pos = (size // 2, size // 2)
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
        width = self.size + 2 * self.map_size[1] + 3
        for key, value in self.options.items():
            if value['n_uses'] > threshold:
                color_shift = max_n_uses * 0.1
                image = np.zeros((height, width)) - color_shift
                # center marks
                image[0, self.size // 2 + 1] = max_n_uses * 0.5
                image[-1, self.size // 2 + 1] = max_n_uses * 0.5
                image[self.size // 2 + 1, 0] = max_n_uses * 0.5
                image[self.size // 2 + 1, self.size + 1] = max_n_uses * 0.5
                image[1:self.size + 1, 1:self.size + 1] = value['policy']
                init_map = value['init']
                term_map = value['term']
                if obstacle_mask is not None:
                    init_map[obstacle_mask] = -color_shift
                    term_map[obstacle_mask] = -color_shift
                image[:self.map_size[0], self.size + 2:self.size + 2 + self.map_size[1]] = value['init']
                image[:self.map_size[0], self.size + 3 + self.map_size[1]:] = value['term']
                plt.imsave(f'/tmp/option_{logger.run.id}_{episode}_{key}.png', image / max_n_uses, vmax=1,
                           cmap='inferno')
                logger.log({f'option {key}': logger.Image(f'/tmp/option_{logger.run.id}_{episode}_{key}.png')},
                           step=episode)

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


def compute_q_values(env: Environment, agent, directions: list = None):
    q = dict()
    visual_block = agent.hierarchy.blocks[2]
    output_block = agent.hierarchy.output_block
    options = output_block.sm.get_sparse_patterns()

    state_pattern = SDR(env.output_sdr_size)
    sp_output = SDR(visual_block.tm.basal_columns)

    if directions is None:
        directions = [env.agent.view_direction]
    # пройти по всем состояниям и вычилить для каждого состояния Q
    for i_flat in np.flatnonzero(~env.aggregated_mask[EntityType.Obstacle]):
        env.agent.position = env.agent._unflatten_position(i_flat)
        q[env.agent.position] = dict()
        for direction in directions:
            env.agent.view_direction = direction
            _, observation, _ = env.observe()
            state_pattern.sparse = observation
            visual_block.sp.compute(state_pattern, False, sp_output)
            (option,
             option_value,
             option_values,
             option_index) = output_block.bg.choose(options,
                                                    condition=sp_output,
                                                    option_weights=None,
                                                    return_option_value=True,
                                                    return_values=True,
                                                    return_index=True)
            q[env.agent.position][direction] = option_values[1]
    return q


def draw_values(path: str, env_shape, q, directions: dict = None):
    if directions is None:
        n_directions = 1
    else:
        n_directions = len(directions)

    values = np.zeros((*env_shape, n_directions))

    for pos, q_pos in q.items():
        for direction, q_pos_dir in q_pos.items():
            values[pos[0], pos[1], direction] = np.sum(q_pos_dir * softmax(q_pos_dir))

    rows, cols = env_shape
    if n_directions == 4:
        flat_values = np.zeros((rows*2, cols*2))
        for pos in q.keys():
            flat_values[pos[0]*2, pos[1]*2] = values[pos[0], pos[1], directions['up']]
            flat_values[pos[0]*2, pos[1]*2 + 1] = values[pos[0], pos[1], directions['right']]
            flat_values[pos[0]*2 + 1, pos[1]*2] = values[pos[0], pos[1], directions['left']]
            flat_values[pos[0]*2 + 1, pos[1]*2 + 1] = values[pos[0], pos[1], directions['down']]
    elif n_directions == 1:
        flat_values = values.reshape(env_shape)
    else:
        raise NotImplemented(f'Not implemented for n_directions = {n_directions}')

    plt.figure(figsize=(13, 13))
    ax = sns.heatmap(flat_values, annot=True, fmt=".1g", cbar=False)
    ax.hlines(np.arange(0, flat_values.shape[0], 1+n_directions//4), xmin=0, xmax=flat_values.shape[1])
    ax.vlines(np.arange(0, flat_values.shape[1], 1+n_directions//4), ymin=0, ymax=flat_values.shape[0])
    figure = ax.get_figure()
    figure.savefig(path)


def draw_policy(path: str, env_shape, q, directions: dict = None, actions_map: dict = None):
    if directions is None:
        n_directions = 1
    else:
        n_directions = len(directions)

    probs = np.zeros((*env_shape, n_directions))
    actions = np.zeros((*env_shape, n_directions), dtype='int32')

    for pos, q_pos in q.items():
        for direction, q_pos_dir in q_pos.items():
            probs[pos[0], pos[1], direction] = np.max(softmax(q_pos_dir))
            actions[pos[0], pos[1], direction] = np.argmax(q_pos_dir)

    rows, cols = env_shape
    if n_directions == 4:
        flat_probs = np.zeros((rows*2, cols*2))
        flat_actions = np.zeros((rows*2, cols*2))
        for pos in q.keys():
            flat_probs[pos[0]*2, pos[1]*2] = probs[pos[0], pos[1], directions['up']]
            flat_probs[pos[0]*2, pos[1]*2 + 1] = probs[pos[0], pos[1], directions['right']]
            flat_probs[pos[0]*2 + 1, pos[1]*2] = probs[pos[0], pos[1], directions['left']]
            flat_probs[pos[0]*2 + 1, pos[1]*2 + 1] = probs[pos[0], pos[1], directions['down']]
            
            flat_actions[pos[0]*2, pos[1]*2] = actions[pos[0], pos[1], directions['up']]
            flat_actions[pos[0]*2, pos[1]*2 + 1] = actions[pos[0], pos[1], directions['right']]
            flat_actions[pos[0]*2 + 1, pos[1]*2] = actions[pos[0], pos[1], directions['left']]
            flat_actions[pos[0]*2 + 1, pos[1]*2 + 1] = actions[pos[0], pos[1], directions['down']]
    elif n_directions == 1:
        flat_probs = probs.reshape(env_shape)
        flat_actions = actions.reshape(env_shape)
    else:
        raise NotImplemented(f'Not implemented for n_directions = {n_directions}')
    # draw probabilities
    plt.figure(figsize=(13, 11))
    ax = sns.heatmap(flat_probs, cbar=True, vmin=0, vmax=1)
    ax.hlines(np.arange(0, flat_probs.shape[0], 1+n_directions//4), xmin=0, xmax=flat_probs.shape[1])
    ax.vlines(np.arange(0, flat_probs.shape[1], 1+n_directions//4), ymin=0, ymax=flat_probs.shape[0])
    # draw arrows
    for i in range(flat_actions.shape[0]):
        for j in range(flat_actions.shape[1]):
            action = flat_actions[i, j]
            if n_directions == 1:
                direction = None
            else:
                direction = get_direction_from_flat((i, j))
            row, col, drow, dcol = get_arrow(actions_map[action], direction)
            ax.arrow(j+col, i+row, dcol, drow, length_includes_head=True,
                     head_width=0.2,
                     head_length=0.2)

    figure = ax.get_figure()
    figure.savefig(path)


def get_arrow(action, direction=None):
    if direction is None:
        if action == 'up':
            row = 1
            col = 0.5
            drow = -1
            dcol = 0
        elif action == 'down':
            row = 0
            col = 0.5
            drow = 1
            dcol = 0
        elif action == 'left':
            row = 0.5
            col = 1
            drow = 0
            dcol = -1
        elif action == 'right':
            row = 0.5
            col = 0
            drow = 0
            dcol = 1
    else:
        if direction == 'up':
            if action == 'move':
                row = 1
                col = 0.5
                drow = -1
                dcol = 0
            elif action == 'turn_left':
                row = 0
                col = 1
                drow = 1
                dcol = -1
            elif action == 'turn_right':
                row = 1
                col = 0
                drow = -1
                dcol = 1
        elif direction == 'down':
            if action == 'move':
                row = 0
                col = 0.5
                drow = 1
                dcol = 0
            elif action == 'turn_right':
                row = 0
                col = 1
                drow = 1
                dcol = -1
            elif action == 'turn_left':
                row = 1
                col = 0
                drow = -1
                dcol = 1
        elif direction == 'left':
            if action == 'move':
                row = 0.5
                col = 1
                drow = 0
                dcol = -1
            elif action == 'turn_right':
                row = 1
                col = 1
                drow = -1
                dcol = -1
            elif action == 'turn_left':
                row = 0
                col = 0
                drow = 1
                dcol = 1
        elif direction == 'right':
            if action == 'move':
                row = 0.5
                col = 0
                drow = 0
                dcol = 1
            elif action == 'turn_left':
                row = 1
                col = 1
                drow = -1
                dcol = -1
            elif action == 'turn_right':
                row = 0
                col = 0
                drow = 1
                dcol = 1
    return row, col, drow, dcol


def get_direction_from_flat(pos):
    if (pos[0] % 2) == 0:
        if (pos[1] % 2) == 0:
            return 'up'
        else:
            return 'right'
    elif (pos[1] % 2) == 0:
        return 'left'
    else:
        return 'down'



