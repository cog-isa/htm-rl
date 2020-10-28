from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import json
import pyrr
import numpy as np

from utils.picker import Picker
from utils.text import TextDrawer
from utils.window import Window


class Drawer:

    def __init__(self, path, win_params):
        # data
        with open(path, "r") as outfile:
            self.data_base = json.load(outfile)

        self.active_cell = -1
        self.current = 0

        self.cells_in_column = self.data_base['params']['cells_per_column']
        self.num_of_cells = self.data_base['params']['num_cells']

        self.start_point = [0, 0, 0]
        self.size = int(np.sqrt(self.num_of_cells / self.cells_in_column))
        distance = 5.

        target = (distance * self.size / 2., 3 * distance * self.cells_in_column / 2., distance * self.size / 2.)
        self.window = Window(win_params[0], win_params[1], target)
        self.pick = Picker(self.window.width, self.window.height, self.num_of_cells)

        # color settings
        self.background_color = [0.98, 1.0, 0.98]
        self.active_cells_color = [0.9, 0., 0.]
        self.predictive_cells_color = [0., 0.9, 0.]
        self.winner_cells_color = [0.99, 0.06, 0.75]
        self.active_and_predictive_cells_color = [0.9, 0.9, 0.]
        self.active_segments_color = [0., 0., 0.]
        self.picked_color = [0., 0., 0.9]
        self.default_color = [0.8, 0.8, 0.8]
        self.bright_off = 2
        self.bright_less = 0.4

        self.textDrawer = TextDrawer()
        self.textDrawer.load_font('fonts/calibri.ttf', 40 * 64, 1 * 64)

        params = {
            'data_base': self.data_base,
            'cells_in_column': self.cells_in_column,
            'num_of_cells': self.num_of_cells,
            'start_point': self.start_point,
            'size': self.size,
            'background_color': self.background_color,
            'active_cells_color': self.active_cells_color,
            'predictive_cells_color': self.predictive_cells_color,
            'winner_cells_color': self.winner_cells_color,
            'active_and_predictive_cells_color': self.active_and_predictive_cells_color,
            'active_segments_color': self.active_segments_color,
            'picked_color': self.picked_color,
            'default_color': self.default_color,
            'bright_off': self.bright_off,
            'bright_less': self.bright_less,
            'draw_cells': [None],
            'distance': distance,
            'win_width': self.window.width,
            'win_height': self.window.height
        }

        # primitives
        cube_buffer = np.array([-0.5, -0.5, 0.5,
                                0.5, -0.5, 0.5,
                                0.5, 0.5, 0.5,
                                -0.5, 0.5, 0.5,

                                -0.5, -0.5, -0.5,
                                0.5, -0.5, -0.5,
                                0.5, 0.5, -0.5,
                                -0.5, 0.5, -0.5,

                                0.5, -0.5, -0.5,
                                0.5, 0.5, -0.5,
                                0.5, 0.5, 0.5,
                                0.5, -0.5, 0.5,

                                -0.5, 0.5, -0.5,
                                -0.5, -0.5, -0.5,
                                -0.5, -0.5, 0.5,
                                -0.5, 0.5, 0.5,

                                -0.5, -0.5, -0.5,
                                0.5, -0.5, -0.5,
                                0.5, -0.5, 0.5,
                                -0.5, -0.5, 0.5,

                                0.5, 0.5, -0.5,
                                -0.5, 0.5, -0.5,
                                -0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, ], dtype=np.float32)
        cube_indices = np.array([0, 1, 2, 2, 3, 0,
                                      4, 5, 6, 6, 7, 4,
                                      8, 9, 10, 10, 11, 8,
                                      12, 13, 14, 14, 15, 12,
                                      16, 17, 18, 18, 19, 16,
                                      20, 21, 22, 22, 23, 20], dtype=np.uint32)
        sphere_buffer, sphere_indices = self.create_sphere()

        pyramid_buffer = np.array([-0.5, -0.5, -0.5,
                                0.5, -0.5, -0.5,
                                0.5, -0.5, 0.5,
                                -0.5, -0.5, 0.5,

                                0.5, -0.5, -0.5,
                                0.5, -0.5, 0.5,
                                0.0, 0.5, 0.0,

                                -0.5, -0.5, 0.5,
                                -0.5, -0.5, -0.5,
                                0.0, 0.5, 0.0,

                                -0.5, -0.5, -0.5,
                                0.5, -0.5, -0.5,
                                0.0, 0.5, 0.0,

                                0.5, -0.5, 0.5,
                                -0.5, -0.5, 0.5,
                                0.0, 0.5, 0.0], dtype=np.float32)
        pyramid_indices = np.array([0, 1, 2, 2, 3, 0,
                                      4, 5, 6,
                                      7, 8, 9,
                                      10, 11, 12,
                                      13, 14, 15], dtype=np.uint32)

        pyramid2_buffer = np.array([0.5, 0.0, -0.5,
                                   0.5, 0.0, 0.5,
                                   0.0, -0.5, 0.0,

                                   -0.5, 0.0, 0.5,
                                   -0.5, 0.0, -0.5,
                                   0.0, -0.5, 0.0,

                                   -0.5, 0.0, -0.5,
                                   0.5, 0.0, -0.5,
                                   0.0, -0.5, 0.0,

                                   0.5, 0.0, 0.5,
                                   -0.5, 0.0, 0.5,
                                   0.0, -0.5, 0.0,

                                   0.5, 0.0, -0.5,
                                   0.5, 0.0, 0.5,
                                   0.0, 0.5, 0.0,

                                   -0.5, 0.0, 0.5,
                                   -0.5, 0.0, -0.5,
                                   0.0, 0.5, 0.0,

                                   -0.5, 0.0, -0.5,
                                   0.5, 0.0, -0.5,
                                   0.0, 0.5, 0.0,

                                   0.5, 0.0, 0.5,
                                   -0.5, 0.0, 0.5,
                                   0.0, 0.5, 0.0], dtype=np.float32)
        pyramid2_indices = np.array([0, 1, 2,
                                    3, 4, 5,
                                    6, 7, 8,
                                    9, 10, 11,
                                    12, 13, 14,
                                    15, 16, 17,
                                    18, 19, 20,
                                    21, 22, 23], dtype=np.uint32)

        self.cells_active = Cells(params, pyramid_indices, pyramid_buffer)
        self.cells_predictive = Cells(params, pyramid2_indices, pyramid2_buffer)
        self.cells_active_predictive = Cells(params, sphere_indices, sphere_buffer)
        self.cells_not = Cells(params, cube_indices, cube_buffer)
        self.pick_cells = Primitives(params, cube_indices, cube_buffer)
        self.pick_cells.create_instance_cubes_array()

        # self.window.proj_loc = [self.cubes.proj_loc, self.pick_cells.proj_loc]

    @staticmethod
    def create_sphere():
        arcs1 = 40
        arcs2 = 20
        r = 0.5

        buffer = []
        indices = []
        for j in range(arcs2):
            for i in range(arcs1 + 1):
                buffer.append(r * np.sin(np.pi * j / arcs2) * np.cos(2 * np.pi * i / arcs1))
                buffer.append(r * np.sin(np.pi * j / arcs2) * np.sin(2 * np.pi * i / arcs1))
                buffer.append(r * np.cos(np.pi * j / arcs2))
                if j != 0 and i != arcs1:
                    indices.append(j * (arcs1 + 1) + i)
                    indices.append((j + 1) * (arcs1 + 1) + i + 1)
                    indices.append(j * (arcs1 + 1) + i + 1)

                if j != arcs2 - 1 and i != arcs1:
                    indices.append(j * (arcs1 + 1) + i)
                    indices.append((j + 1) * (arcs1 + 1) + i)
                    indices.append((j + 1) * (arcs1 + 1) + i + 1)
        for i in range(arcs1 + 1):
            buffer.append(0)
            buffer.append(0)
            buffer.append(r)

        return np.around(np.array(buffer, dtype=np.float32), decimals=2), np.array(indices, dtype=np.uint32)

    def cells_type(self):
        active = []
        predictive = []
        active_predictive = []
        not_colored = []
        for cell in range(self.num_of_cells):
            if abs(self.data_base[str(self.current)][str(cell)]['St']) == 1:
                active.append(cell)
            elif abs(self.data_base[str(self.current)][str(cell)]['St']) == 2:
                predictive.append(cell)
            elif abs(self.data_base[str(self.current)][str(cell)]['St']) == 3:
                active_predictive.append(cell)
            else:
                not_colored.append(cell)
        return active, predictive, active_predictive, not_colored

    def update(self):
        self.cells_active.draw_cells, self.cells_predictive.draw_cells, self.cells_active_predictive.draw_cells, self.cells_not.draw_cells = self.cells_type()

        self.cells_active.update(self.current,
                          self.window.show_segments,
                          self.window.show_active_segments,
                          self.active_cell)
        self.cells_predictive.update(self.current,
                          self.window.show_segments,
                          self.window.show_active_segments,
                          self.active_cell)
        self.cells_active_predictive.update(self.current,
                                     self.window.show_segments,
                                     self.window.show_active_segments,
                                     self.active_cell)
        self.cells_not.update(self.current,
                                     self.window.show_segments,
                                     self.window.show_active_segments,
                                     self.active_cell)

    def update_color(self):
        self.cells_active.update_instance_cubes_color_array(self.current,
                                                     self.window.show_segments,
                                                     self.window.show_active_segments,
                                                     self.active_cell)
        self.cells_predictive.update_instance_cubes_color_array(self.current,
                                                     self.window.show_segments,
                                                     self.window.show_active_segments,
                                                     self.active_cell)
        self.cells_active_predictive.update_instance_cubes_color_array(self.current,
                                                                self.window.show_segments,
                                                                self.window.show_active_segments,
                                                                self.active_cell)
        self.cells_not.update_instance_cubes_color_array(self.current,
                                                                self.window.show_segments,
                                                                self.window.show_active_segments,
                                                                self.active_cell)

    def move(self, velocity=0.1):
        if self.window.left:
            self.window.cam.process_keyboard("LEFT", velocity)
        if self.window.right:
            self.window.cam.process_keyboard("RIGHT", velocity)
        if self.window.forward:
            self.window.cam.process_keyboard("FORWARD", velocity)
        if self.window.backward:
            self.window.cam.process_keyboard("BACKWARD", velocity)

    def draw_legend(self):
        left = 500
        if self.window.show_segments and not self.window.not_show_segments:
            self.textDrawer.draw_text('Active segments', (self.window.width - left, self.window.width - 30),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.active_segments_color,
                                      backColor=self.active_segments_color)
        else:
            if self.active_cell >= 0:
                self.textDrawer.draw_text('Not connected cells', (self.window.width - left, self.window.width - 380),
                                          (self.window.width, self.window.width), scale=(1.0, 1.),
                                          foreColor=np.array(self.default_color) * self.bright_less,
                                          backColor=np.array(self.default_color) * self.bright_less)
            self.textDrawer.draw_text('Active cells', (self.window.width - left, self.window.width - 30),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.active_cells_color,
                                      backColor=self.active_cells_color)
            self.textDrawer.draw_text('Predictive cells', (self.window.width - left, self.window.width - 100),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.predictive_cells_color,
                                      backColor=self.predictive_cells_color)
            self.textDrawer.draw_text('Winner cells', (self.window.width - left, self.window.width - 170),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.winner_cells_color,
                                      backColor=self.winner_cells_color)
            self.textDrawer.draw_text('Active & Predictive cells', (self.window.width - left, self.window.width - 240),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.active_and_predictive_cells_color,
                                      backColor=self.active_and_predictive_cells_color)
            self.textDrawer.draw_text('Picked out cell', (self.window.width - left, self.window.width - 310),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.picked_color,
                                      backColor=self.picked_color)

    def process(self):
        self.window.poll_events()
        self.move(self.window.velocity)
        view = self.window.cam.get_view_matrix()

        glClearColor(*self.background_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.cells_active.draw(view)
        self.cells_predictive.draw(view)
        self.cells_active_predictive.draw(view)
        self.cells_not.draw(view)


        self.draw_legend()

        if self.window.picker:
            self.pick_cells.draw(view, self.pick.EnableWriting)
            active_ind = self.pick.read(self.window.mouseX, self.window.height - self.window.mouseY)
            # print(active_ind[0], active_ind[0], active_ind[0])
            if active_ind[0] == 255 and active_ind[1] == 255 and active_ind[2] == 255:
                self.active_cell = -1
            else:
                self.active_cell = active_ind[2] + 256 * active_ind[1] + 256 * 256 * active_ind[0]
            # print(self.active_cell)
            self.pick.DisableWriting()
            self.window.picker = False
            self.update_color()

        if self.window.show_segments and self.window.not_show_segments:
            self.window.show_segments = False
            self.update_color()
        if self.window.show_segments and not self.window.not_show_segments:
            if self.window.show_active_segments and self.window.not_show_active_segments:
                self.window.show_active_segments = False
            self.update_color()

        self.window.swap_buffers()


class Primitives:
    def __init__(self, params, indices, buffer):
        self.draw_cells = params['draw_cells']
        self.instance_array_cube = []
        self.instance_array_cube_color = []
        self.distance = params['distance']
        self.num_of_draw_cells = params['num_of_cells']
        self.win_width = params['win_width']
        self.win_height = params['win_height']
        self.start_point = params['start_point']
        self.num_of_cells = params['num_of_cells']
        self.cells_in_column = params['cells_in_column']
        self.size = params['size']

        # shaders
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec3 a_offset;
        layout(location = 2) in vec3 a_color;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;

        out vec3 v_color;

        void main()
        {
            vec3 final_pos = a_position + a_offset;
            gl_Position =  projection * view * model * vec4(final_pos, 1.0f);
            v_color = a_color;
        }
        """
        fragment_src = """
        # version 330

        in vec3 v_color;

        out vec4 out_color;

        void main()
        {
            out_color = vec4(v_color, 0.8);
        }
        """
        self.shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                     compileShader(fragment_src, GL_FRAGMENT_SHADER))
        self.indices = indices

        # VAO, VBO and EBO
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        # cube Buffer
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 3, ctypes.c_void_p(0))

        projection = pyrr.matrix44.create_perspective_projection_matrix(45,
                                                                        self.win_width / self.win_height,
                                                                        0.1,
                                                                        2000)
        cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

        glUseProgram(self.shader)
        model_loc = glGetUniformLocation(self.shader, "model")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, cube_pos)

    def create_instance_cubes_array(self):
        self.instance_cube = glGenBuffers(1)
        glBindVertexArray(self.vao)
        if not (None in self.draw_cells):
            self.num_of_draw_cells = len(self.draw_cells)
        self.instance_array_cube = []
        x, y, z = self.start_point
        for cell in range(self.num_of_cells):
            if cell in self.draw_cells or None in self.draw_cells:
                translation = pyrr.Vector3([0.0, 0.0, 0.0])
                translation.x = self.distance * x
                translation.y = 3 * self.distance * y
                translation.z = self.distance * z
                self.instance_array_cube.append(translation)

            if y < self.cells_in_column - 1:
                y += 1
            else:
                y = 0
                if x < self.size - 1:
                    x += 1
                else:
                    x = 0
                    z += 1

        self.instance_array_cube = np.array(self.instance_array_cube, np.float32).flatten()

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_cube)
        glBufferData(GL_ARRAY_BUFFER, self.instance_array_cube.nbytes, self.instance_array_cube, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)  # 1 means, every instance will have it's own translate

    def draw(self, view, enable):
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        enable()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
        glDrawElementsInstanced(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None, self.num_of_draw_cells)


class Cells(Primitives):
    def __init__(self, params, indices, buffer):
        Primitives.__init__(self, params, indices, buffer)
        self.data_base = params['data_base']
        self.background_color = params['background_color']
        self.default_color = params['default_color']
        self.active_cells_color = params['active_cells_color']
        self.predictive_cells_color = params['predictive_cells_color']
        self.winner_cells_color = params['winner_cells_color']
        self.active_and_predictive_cells_color = params['active_and_predictive_cells_color']
        self.active_segments_color = params['active_segments_color']
        self.picked_color = params['picked_color']
        self.bright_less = params['bright_less']

    def update(self, current, show_segments, show_active_segments, active_cell):
        self.create_instance_cubes_array()
        self.instance_cube_color = glGenBuffers(1)
        self.update_instance_cubes_color_array(current, show_segments, show_active_segments, active_cell)

    def update_instance_cubes_color_array(self, current, show_segments, show_active_segments, active_cell):
        glBindVertexArray(self.vao)
        if active_cell >= 0:
            dict_segments = self.data_base[str(current)][str(active_cell)]['Se']
        self.instance_array_cube_color = []
        for cell in range(self.num_of_cells):
            if cell in self.draw_cells or None in self.draw_cells:
                cell_key = str(cell)
                default_color = pyrr.Vector3(self.default_color)
                if abs(self.data_base[str(current)][cell_key]['St']) == 1: default_color = pyrr.Vector3(
                    self.active_cells_color)
                if abs(self.data_base[str(current)][cell_key]['St']) == 2: default_color = pyrr.Vector3(
                    self.predictive_cells_color)
                if abs(self.data_base[str(current)][cell_key]['St']) == 3: default_color = pyrr.Vector3(
                    self.active_and_predictive_cells_color)
                if self.data_base[str(current)][cell_key]['St'] < 0: default_color = pyrr.Vector3([self.winner_cells_color])
                if active_cell == cell:
                    default_color = pyrr.Vector3(self.picked_color)
                if active_cell >= 0:
                    color = default_color
                    if show_segments:
                        default_color = self.background_color
                    else:
                        default_color = default_color * self.bright_less
                    if len(dict_segments.keys()) != 0:
                        counter = 0
                        base0 = 50
                        for segment in dict_segments.keys():
                            state = dict_segments[segment]['St']
                            connected_cells = dict_segments[segment]['Ce']
                            if cell in connected_cells:
                                if show_segments:
                                    base = base0 + (counter // 6) * 40
                                    if counter % 3 == 0: default_color = pyrr.Vector3([0., 0., base])
                                    if counter % 3 == 1: default_color = pyrr.Vector3([0., base, 0.])
                                    if counter % 3 == 2: default_color = pyrr.Vector3([base, 0., 0.])
                                    if counter % 6 == 3: default_color = pyrr.Vector3([0., base, base])
                                    if counter % 6 == 4: default_color = pyrr.Vector3([base, base, 0.])
                                    if counter % 6 == 5: default_color = pyrr.Vector3([base, 0., base])
                                    if show_active_segments and state:
                                        default_color = pyrr.Vector3(self.active_segments_color)
                                else:
                                    default_color = color
                            counter = counter + 1

                self.instance_array_cube_color.append(default_color)

        self.instance_array_cube_color = np.array(self.instance_array_cube_color, np.float32).flatten()


        glBindBuffer(GL_ARRAY_BUFFER, self.instance_cube_color)
        glBufferData(GL_ARRAY_BUFFER, self.instance_array_cube_color.nbytes, self.instance_array_cube_color,
                         GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)

    def draw(self, view):
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_cube_color)
        glBufferData(GL_ARRAY_BUFFER, self.instance_array_cube_color.nbytes, self.instance_array_cube_color,
                             GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawElementsInstanced(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None, self.num_of_draw_cells)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)