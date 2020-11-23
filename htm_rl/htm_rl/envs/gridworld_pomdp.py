import numpy as np
from typing import Tuple

legend = {
    'map':
        {
            0: ' ',
            1: "#",
            2: '*',
            3: '?'
        },
    'agent':
        {
            0: '>',
            1: '^',
            2: '<',
            3: '↓'
        }
}


class GridWorld:
    """
        world_description:
            2d list with numbers that correspond to type of tile:
                0: floor
                1: wall
                2: reward
                3: fog
        world_size:
            tuple: (rows: int, columns: int)
        agent_initial_position:
            dict with grid coordinates:
                {'row': int, 'column': int}
        agent_initial_direction:
            int:
                0: right
                1: up
                2: left
                3: down
        max_sight_range:
            int max number of cells that agent can see on line
        transparent_obstacles:
            bool used by 'window' observation, if true: agent sees map behind obstacle as it is,
            else: agent sees fog behind walls
        observable_vars:
            ordered subset of {'distance', 'surface', 'relative_row',
                                'relative_column', 'relative_direction',
                                'state_id',
                                'flatten_index',
                                'row',
                                'column',
                                'direction',
                                'window'}
            as list, output order will be the same as in observable_vars
            'state_id' correspond to mdp framework, unique id for every state.
        window_coords:
            dict {'top_left': (1, -1), 'bottom_right': (0, 1)}
            window coordinates relative to agent's position
            (0, 0) is the agent's position
                          (1,0)
                            |
                            |
               (0,-1)-----(0,0)-----(0,1)
                            |
                            |
                         (-1,0)
        one_value_state:
            bool convert tuple of state to one value
    """

    def __init__(self, world_description,
                 world_size: tuple,
                 agent_initial_position: dict,
                 agent_initial_direction=0,
                 max_sight_range=None,
                 observable_vars=None,
                 one_value_state=False,
                 transparent_obstacles=True,
                 window_coords=None):

        self.one_value_state = one_value_state
        self.observable_vars = observable_vars
        self.world_size = world_size
        self.world_description = np.ones((world_size[0] + 2, world_size[1] + 2), dtype=np.int)
        self.world_description[1:-1, 1:-1] = np.array(world_description, dtype=np.int)
        self.n_actions = 3
        self.n_states = world_size[0] * world_size[1] * 4

        if max_sight_range is None:
            max_sight_range = max(self.world_description.shape)

        converted_agent_position = {
            'row': agent_initial_position['row'] + 1,
            'column': agent_initial_position['column'] + 1}

        self.agent_position = converted_agent_position
        self.step_number = 0
        self.agent_direction = agent_initial_direction
        self.max_sight_range = max_sight_range
        self.transparent_obstacles = transparent_obstacles
        self.observable_state = None
        self.filtered_observation = None

        if window_coords is None:
            self.window_coords = {
                'top_left': (1, -1),
                'bottom_right': (0, 1)
            }
        else:
            self.window_coords = window_coords

        self.dimensions = {
            'distance': max(self.world_size),
            'surface': 3,
            'relative_row': 2 * (self.world_size[0] + 1),
            'relative_column': 2 * (self.world_size[1] + 1),
            'relative_direction': 4,
            'state_id': self.n_states,
            'flatten_index': self.world_size[0] * self.world_size[1],
            'row': self.world_size[0] + 1,
            'column': self.world_size[1] + 1,
            'direction': 4,
            'window': 4 ** (self.max_sight_range * self.max_sight_range)
        }

        self.n_obs_states = 1
        for key in self.observable_vars:
            self.n_obs_states *= self.dimensions[key]

        self._agent_initial_position = converted_agent_position.copy()
        self._agent_initial_direction = agent_initial_direction
        self.observe()

    def step(self, action: int):
        """
            actions:
                0: rotate clockwise
                1: rotate counterclockwise
                2: step forward
        """
        punish = 0

        if action == 0:
            self.agent_direction -= 1
            if self.agent_direction < 0:
                self.agent_direction += 4
        elif action == 1:
            self.agent_direction = (self.agent_direction + 1) % 4
        elif action == 2:
            if self.agent_direction == 0:
                # right
                new_column = self.agent_position['column'] + 1
                # if not wall then go
                if self.world_description[self.agent_position['row'], new_column] != 1:
                    self.agent_position['column'] = new_column
                else:
                    punish += 100
            elif self.agent_direction == 1:
                # up
                new_row = self.agent_position['row'] - 1
                if self.world_description[new_row, self.agent_position['column']] != 1:
                    self.agent_position['row'] = new_row
                else:
                    punish += 100
            elif self.agent_direction == 2:
                # left
                new_column = self.agent_position['column'] - 1
                if self.world_description[self.agent_position['row'], new_column] != 1:
                    self.agent_position['column'] = new_column
                else:
                    punish += 100
            elif self.agent_direction == 3:
                # down
                new_row = self.agent_position['row'] + 1
                if self.world_description[new_row, self.agent_position['column']] != 1:
                    self.agent_position['row'] = new_row
                else:
                    punish += 100
            else:
                raise ValueError
        else:
            raise ValueError

        self.step_number += 1

        return self.observe(), self.reward() - punish, self.is_terminal(), {'previous_action': action}

    def observe(self):
        # we see only cells that are in front of us and we can't see
        # through walls
        # we have locator that can say what type of surface we see
        # and how far that surface

        if self.agent_direction == 2:
            # for left
            line = self.world_description[self.agent_position['row'], :self.agent_position['column']]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for left
            stop_pos = obstacle_pos.max()

            distance = np.abs(self.agent_position['column'] - stop_pos)

        elif self.agent_direction == 0:
            # for right
            line = self.world_description[self.agent_position['row'], self.agent_position['column'] + 1:]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for right
            stop_pos = obstacle_pos.min()

            distance = stop_pos + 1

        elif self.agent_direction == 1:
            # for up
            line = self.world_description[:self.agent_position['row'], self.agent_position['column']]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for up
            stop_pos = obstacle_pos.max()

            distance = np.abs(self.agent_position['row'] - stop_pos)

        elif self.agent_direction == 3:
            # for down
            line = self.world_description[self.agent_position['row'] + 1:, self.agent_position['column']]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for down
            stop_pos = obstacle_pos.min()

            distance = stop_pos + 1
        else:
            raise ValueError

        if distance > self.max_sight_range:
            surface = 0
            distance = self.max_sight_range
        else:
            surface = line[stop_pos]

        relative_coordinates = (self.agent_position['row'] - self._agent_initial_position['row'],
                                self.agent_position['column'] - self._agent_initial_position['column'])

        relative_direction = self.agent_direction - self._agent_initial_direction
        if relative_direction < 0:
            relative_direction += 4

        flatten_index = self.agent_position['column'] - 1 + (self.agent_position['row'] - 1) * self.world_size[1]

        state_id = (flatten_index
                    + self.agent_direction * self.world_size[0] * self.world_size[1])

        # window

        self.observable_state = {'distance': distance - 1,
                                 'surface': surface,
                                 'relative_row': relative_coordinates[0],
                                 'relative_column': relative_coordinates[1],
                                 'relative_direction': relative_direction,
                                 'state_id': state_id,
                                 'flatten_index': flatten_index,
                                 'row': self.agent_position['row'] - 1,
                                 'column': self.agent_position['column'] - 1,
                                 'direction': self.agent_direction,
                                 'window': self.get_window()
                                 }

        if self.observable_vars is not None:
            filtered_observation = dict()
            for key in self.observable_vars:
                filtered_observation[key] = self.observable_state[key]
            observation = filtered_observation
        else:
            observation = self.observable_state

        self.filtered_observation = observation

        return self.unpack(self.filtered_observation)

    def reward(self):
        if self.observable_state['surface'] == 2 and self.observable_state['distance'] == 0:
            return 1 / (self.observable_state['distance'] + 1e-6)
        else:
            return -1e-2

    def is_terminal(self):
        if self.observable_state['surface'] == 2 and self.observable_state['distance'] == 0:
            return True
        else:
            return False

    def reset(self):
        self.agent_position = self._agent_initial_position.copy()
        self.agent_direction = self._agent_initial_direction
        self.step_number = 0
        self.observe()
        return self.unpack(self.filtered_observation)

    def render(self):
        for i, row in enumerate(self.world_description):
            for j, x in enumerate(row):
                if (i, j) != (self.agent_position['row'], self.agent_position['column']):
                    print(f"{legend['map'][x]}", end='|')
                else:
                    print(f"{legend['agent'][self.agent_direction]}", end='|')
            print()

    def unpack(self, state: dict):
        if len(state.values()) == 1:
            return tuple(state.values())[0]
        elif self.one_value_state:
            n = 1
            s = 0
            for key, value in state.items():
                s += n * value
                n *= self.dimensions[key]
            return s
        else:
            return tuple(state.values())

    def get_window(self):
        top_left = self.window_coords['top_left']
        bottom_right = self.window_coords['bottom_right']
        width = abs(top_left[1] - bottom_right[1])
        height = abs(top_left[0] - bottom_right[0])
        obs = np.zeros((height+1, width+1)) + 3

        if self.agent_direction == 0:
            wd = self.world_description.T[::-1, :]
            agent_row = self.agent_position['column']
            agent_column = self.agent_position['row']
            agent_row = wd.shape[0]-1 - agent_row
        elif self.agent_direction == 2:
            wd = self.world_description.T[:, ::-1]
            agent_row = self.agent_position['column']
            agent_column = wd.shape[1]-1 - self.agent_position['row']
        elif self.agent_direction == 3:
            wd = self.world_description[::-1, ::-1]
            agent_row = wd.shape[0]-1 - self.agent_position['row']
            agent_column = wd.shape[1]-1 - self.agent_position['column']
        else:
            wd = self.world_description
            agent_row = self.agent_position['row']
            agent_column = self.agent_position['column']

        top_left_orig = (agent_row - top_left[0],
                         agent_column + top_left[1])
        bottom_right_orig = (agent_row - bottom_right[0],
                             agent_column + bottom_right[1])

        if top_left_orig[0] < 0:
            top_left_row_obs = -top_left_orig[0]
        else:
            top_left_row_obs = 0
        if top_left_orig[1] < 0:
            top_left_col_obs = -top_left_orig[1]
        else:
            top_left_col_obs = 0

        top_left_clip = (np.clip(top_left_orig[0], 0,
                                 wd.shape[0]-1),
                         np.clip(top_left_orig[1], 0,
                                 wd.shape[1]-1))
        bottom_right_clip = (np.clip(bottom_right_orig[0], 0,
                                     wd.shape[0]-1),
                             np.clip(bottom_right_orig[1], 0,
                                     wd.shape[1]-1))
        orig = wd[top_left_clip[0]: bottom_right_clip[0]+1, top_left_clip[1]: bottom_right_clip[1]+1]

        bottom_right_row_obs = top_left_row_obs + orig.shape[0]
        bottom_right_col_obs = top_left_col_obs + orig.shape[1]

        obs[top_left_row_obs: bottom_right_row_obs, top_left_col_obs: bottom_right_col_obs] = orig
        return obs


class MapGenerator:
    def __init__(self, shape: Tuple[int, int], complexity=0.75, density=0.75, seed: int=0):
        self.shape = shape
        self.complexity = complexity
        self.density = density
        self.seed = seed
        self.random_gen = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        seed = self.random_gen.integers(2 ** 31)
        world = self.generate_maze(seed)
        world = self.place_reward(world, seed)
        return world

    @staticmethod
    def place_reward(maze, seed):
        np.random.default_rng(seed)
        rows, columns = np.nonzero(maze == 0)
        row = np.random.choice(rows, 1)
        column = np.random.choice(columns, 1)
        maze[row, column] = 2
        return maze

    @staticmethod
    def generate_position(maze, seed):
        np.random.default_rng(seed)
        rows, columns = np.nonzero(maze == 0)
        row = np.random.choice(rows, 1)
        column = np.random.choice(columns, 1)
        return row, column

    def generate_maze(self, seed):
        r"""Generate a random maze array.

        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``.

        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        np.random.default_rng(seed)
        shape = (((self.shape[1] + 2) // 2) * 2 + 1, ((self.shape[0] + 2) // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(self.complexity * (5 * (shape[0] + shape[1])))
        density = int(self.density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        maze = np.zeros(shape, dtype=bool)
        # Fill borders
        maze[0, :] = maze[-1, :] = 1
        maze[:, 0] = maze[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = np.random.randint(0, shape[1] // 2 + 1) * 2, np.random.randint(0, shape[0] // 2 + 1) * 2
            maze[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                    if maze[y_, x_] == 0:
                        maze[y_, x_] = 1
                        maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        return maze[1:-1, :][:, 1:-1].astype(int)
