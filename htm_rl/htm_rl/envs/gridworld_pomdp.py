import numpy as np

legend = {
    'map':
        {
            0: ' ',
            1: "#",
            2: '*'
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
        observable_vars:
            ordered subset of {'distance', 'surface', 'relative_row',
                                'relative_column', 'relative_direction',
                                'state_id',
                                'flatten_index',
                                'row',
                                'column',
                                'direction'}
            as list, output order will be the same as in observable_vars
            'state_id' correspond to mdp framework, unique id for every state.
        one_value_state:
            bool convert tuple of state to one value
    """

    def __init__(self, world_description,
                 world_size: tuple,
                 agent_initial_position: dict,
                 agent_initial_direction=0,
                 max_sight_range=None,
                 observable_vars=None,
                 one_value_state=False):

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
        self.observable_state = None
        self.filtered_observation = None

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
            'direction': 4
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

        self.observable_state = {'distance': distance - 1,
                                 'surface': surface,
                                 'relative_row': relative_coordinates[0],
                                 'relative_column': relative_coordinates[1],
                                 'relative_direction': relative_direction,
                                 'state_id': state_id,
                                 'flatten_index': flatten_index,
                                 'row': self.agent_position['row'] - 1,
                                 'column': self.agent_position['column'] - 1,
                                 'direction': self.agent_direction
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
