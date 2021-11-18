import numpy as np


class ImageMovement:
    def __init__(self, window_pos, actions):
        """
        coordinate system
              y
          0----->
          |
        x |
          v

        position = (x, y) or (rows, columns)
        :param window_pos: top left and bottom right corners with respect to
        agent position, for example:
            window 3x3 with agent at the center
            [-1, -1, 1, 1]
        :param actions: action displacements, for example:
            four actions: right, left, down, up
            [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        self.image = np.empty(0)
        self.position = [0, 0]
        self.top_left = [0, 0]
        self.bottom_right = [0, 0]

        self.window_pos = window_pos
        self.actions = actions

    def set_image(self, image):
        self.image = image

    def set_position(self, position):
        self.position = list(position)
        self.top_left = [self.position[0] + self.window_pos[0],
                         self.position[1] + self.window_pos[1]]
        self.bottom_right = [self.position[0] + self.window_pos[2],
                             self.position[1] + self.window_pos[3]]

    def get_possible_actions(self):
        actions = list()
        for action, disp in enumerate(self.actions):
            self.top_left[0] += disp[0]
            self.top_left[1] += disp[1]
            self.bottom_right[0] += disp[0]
            self.bottom_right[1] += disp[1]

            if ((0 < self.top_left[0] < self.image.shape[0]) and
                    (0 < self.top_left[1] < self.image.shape[1]) and
                    (0 < self.bottom_right[0] < self.image.shape[0]) and
                    (0 < self.bottom_right[1] < self.image.shape[1])):
                actions.append(action)
        return actions

    def observe(self):
        return self.image[self.top_left[0]:self.bottom_right[0]+1][:, self.top_left[1]:self.bottom_right[1]+1]

    def act(self, action):
        self.position[0] += self.actions[action][0]
        self.position[1] += self.actions[action][1]

        self.top_left[0] += self.actions[action][0]
        self.top_left[1] += self.actions[action][1]
        self.bottom_right[0] += self.actions[action][0]
        self.bottom_right[1] += self.actions[action][1]
