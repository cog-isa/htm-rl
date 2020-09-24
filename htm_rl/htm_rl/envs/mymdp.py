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

    """

    def __init__(self, world_description,
                 world_size: tuple,
                 agent_initial_position: dict,
                 agent_initial_direction=0,
                 max_sight_range=None):

        self.world_size = world_size
        self.world_description = np.ones((world_size[0]+2, world_size[1]+2))
        self.world_description[1:-1, 1:-1] = np.array(world_description)

        if max_sight_range is None:
            max_sight_range = max(self.world_description.shape)

        converted_agent_position = {
            'row': agent_initial_position['row'] + 1,
            'column': agent_initial_position['column'] + 1}

        self.agent_position = converted_agent_position

        self.agent_direction = agent_initial_direction
        self.max_sight_range = max_sight_range
        self.observable_state = None

        self._agent_initial_position = converted_agent_position.copy()
        self._agent_initial_direction = agent_initial_direction
        self.observation()

    def step(self, action: int):
        """
            actions:
                0: rotate clockwise
                1: rotate counterclockwise
                2: step forward
        """
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
            elif self.agent_direction == 1:
                # up
                new_row = self.agent_position['row'] - 1
                if self.world_description[new_row, self.agent_position['column']] != 1:
                    self.agent_position['row'] = new_row
            elif self.agent_direction == 2:
                # left
                new_column = self.agent_position['column'] - 1
                if self.world_description[self.agent_position['row'], new_column] != 1:
                    self.agent_position['column'] = new_column
            elif self.agent_direction == 3:
                # down
                new_row = self.agent_position['row'] + 1
                if self.world_description[new_row, self.agent_position['column']] != 1:
                    self.agent_position['row'] = new_row
            else:
                raise ValueError
        else:
            raise ValueError

        return self.observation(), self.reward(), self.is_terminal(), {}

    def observation(self):
        # we see only cells that are in front of us and we can't see
        # through walls
        # we have locator that can say what type of surface we see
        # and how far that surface

        if self.agent_direction == 2:
            # for left
            line = self.world_description[self.agent_position['row'], :self.agent_position['column']+1]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for left
            stop_pos = obstacle_pos.max()

            distance = np.abs(self.agent_position['column'] - stop_pos)

        elif self.agent_direction == 0:
            # for right
            line = self.world_description[self.agent_position['row'], self.agent_position['column']:]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for right
            stop_pos = obstacle_pos.min()

            distance = stop_pos

        elif self.agent_direction == 1:
            # for up
            line = self.world_description[:self.agent_position['row']+1, self.agent_position['column']]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for up
            stop_pos = obstacle_pos.max()

            distance = np.abs(self.agent_position['row'] - stop_pos)

        elif self.agent_direction == 3:
            # for down
            line = self.world_description[self.agent_position['row']:, self.agent_position['column']]

            # maze must have wall as border
            obstacle_pos = np.argwhere(line > 0)

            # for down
            stop_pos = obstacle_pos.min()

            distance = stop_pos
        else:
            raise ValueError

        if distance > self.max_sight_range:
            surface = 0
            distance = self.max_sight_range
        else:
            surface = line[stop_pos]

        self.observable_state = {'distance': distance, 'surface': surface}

        return distance, surface

    def reward(self):
        if self.observable_state['surface'] == 2:
            return 1/(self.observable_state['distance'] + 1e-6)
        else:
            return -1e-2

    def is_terminal(self):
        if self.world_description[[self.agent_position['row'], self.agent_position['column']]] == 2:
            return True
        else:
            return False

    def reset(self):
        self.agent_position = self._agent_initial_position.copy()
        self.agent_direction = self._agent_initial_direction
        self.observation()
        return self.observable_state['distance'], self.observable_state['surface']

    def render(self):
        for i, row in enumerate(self.world_description):
            for j, x in enumerate(row):
                if (i, j) != (self.agent_position['row'], self.agent_position['column']):
                    print(f"{legend['map'][x]}", end='|')
                else:
                    print(f"{legend['agent'][self.agent_direction]}", end='|')
            print()
