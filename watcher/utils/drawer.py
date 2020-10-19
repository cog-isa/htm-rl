from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import json
import pyrr
import numpy as np

from utils.picker import Picker
from utils.text import TextDrawer


class Drawer:

    def __init__(self, path, window):
        # data
        with open(path, "r") as outfile:
            self.data_base = json.load(outfile)
        self.window = window
        self.current = 0
        self.active_cell = -1

        self.instance_array_cube = []
        self.instance_array_cube_color = []
        self.offset = 0
        self.distance = 4

        self.cells_in_column = self.data_base['params']['cells_per_column']
        self.num_of_cells = self.data_base['params']['num_cells']

        self.start_point = [0, 0, 0]
        self.size = int(np.sqrt(self.num_of_cells / self.cells_in_column))
        self.pick = Picker(self.window.width, self.window.height, self.num_of_cells)
        self.textDrawer = TextDrawer()
        self.textDrawer.load_font('fonts/calibri.ttf', 40 * 64, 1 * 64)

        # color settings
        self.background_color = [0.98, 1.0, 0.98]
        self.active_cells_color = [0.9, 0., 0.]
        self.predictive_cells_color = [0., 0.9, 0.]
        self.active_and_predictive_cells_color = [0.9, 0.9, 0.]
        self.active_segments_color = [0., 0., 0.]
        self.picked_color = [0., 0., 0.9]
        self.default_color = [0.8, 0.8, 0.8]
        self.bright_off = 2
        self.bright_less = 0.

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
        self.cube_indices = np.array([0, 1, 2, 2, 3, 0,
                                      4, 5, 6, 6, 7, 4,
                                      8, 9, 10, 10, 11, 8,
                                      12, 13, 14, 14, 15, 12,
                                      16, 17, 18, 18, 19, 16,
                                      20, 21, 22, 22, 23, 20], dtype=np.uint32)

        # VAO, VBO and EBO
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        # cube Buffer
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, cube_buffer.nbytes, cube_buffer, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.cube_indices.nbytes, self.cube_indices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube_buffer.itemsize * 3, ctypes.c_void_p(0))

        self.instance_cube = glGenBuffers(1)
        self.create_instance_cubes_array()
        self.instance_cube_color = glGenBuffers(1)

        glClearColor(*self.background_color, 1)
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, self.window.width / self.window.height, 0.1,
                                                                        2000)
        cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

        glUseProgram(self.shader)
        model_loc = glGetUniformLocation(self.shader, "model")
        self.window.cam.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(self.window.cam.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, cube_pos)

    def create_instance_cubes_array(self):
        self.instance_array_cube = []
        x, y, z = self.start_point
        for cell in range(self.num_of_cells):
            translation = pyrr.Vector3([0.0, 0.0, 0.0])
            translation.x = self.distance * x + self.offset
            translation.y = self.distance * y + self.offset
            translation.z = self.distance * z + self.offset
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

    def update_instance_cubes_color_array(self):
        if self.active_cell >= 0:
            dict_segments = self.data_base[str(self.current)][str(self.active_cell)]['Se']

        self.instance_array_cube_color = []
        for cell in range(self.num_of_cells):
            cell_key = str(cell)
            default_color = pyrr.Vector3(self.default_color)
            if self.data_base[str(self.current)][cell_key]['St'] == 1: default_color = pyrr.Vector3(
                self.active_cells_color)
            if self.data_base[str(self.current)][cell_key]['St'] == 2: default_color = pyrr.Vector3(
                self.predictive_cells_color)
            if self.data_base[str(self.current)][cell_key]['St'] == 3: default_color = pyrr.Vector3(
                self.active_and_predictive_cells_color)
            if self.active_cell == cell:
                default_color = pyrr.Vector3(self.picked_color)
            if self.active_cell >= 0:
                color = default_color
                if self.window.show_segments:
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
                            if self.window.show_segments:
                                base = base0 + (counter // 6) * 40
                                if counter % 3 == 0: default_color = pyrr.Vector3([0., 0., base])
                                if counter % 3 == 1: default_color = pyrr.Vector3([0., base, 0.])
                                if counter % 3 == 2: default_color = pyrr.Vector3([base, 0., 0.])
                                if counter % 6 == 3: default_color = pyrr.Vector3([0., base, base])
                                if counter % 6 == 4: default_color = pyrr.Vector3([base, base, 0.])
                                if counter % 6 == 5: default_color = pyrr.Vector3([base, 0., base])
                                if self.window.show_active_segments and state:
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
                self.textDrawer.draw_text('Not connected cells', (self.window.width - left, self.window.width - 310),
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
            self.textDrawer.draw_text('Active & Predictive cells', (self.window.width - left, self.window.width - 170),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.active_and_predictive_cells_color,
                                      backColor=self.active_and_predictive_cells_color)
            self.textDrawer.draw_text('Picked out cell', (self.window.width - left, self.window.width - 240),
                                      (self.window.width, self.window.width), scale=(1.0, 1.),
                                      foreColor=self.picked_color,
                                      backColor=self.picked_color)

    def process(self):
        self.window.poll_events()
        self.move(self.window.velocity)
        view = self.window.cam.get_view_matrix()

        glClearColor(*self.background_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
        glDrawElementsInstanced(GL_TRIANGLES, len(self.cube_indices), GL_UNSIGNED_INT, None, self.num_of_cells)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        self.draw_legend()

        if self.window.picker:
            glUseProgram(self.shader)
            glBindVertexArray(self.vao)
            self.pick.EnableWriting()
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
            glDrawElementsInstanced(GL_TRIANGLES, len(self.cube_indices), GL_UNSIGNED_INT, None, self.num_of_cells)
            active_ind = self.pick.read(self.window.mouseX, self.window.height - self.window.mouseY)
            # print(active_ind[0], active_ind[0], active_ind[0])
            if active_ind[0] == 255 and active_ind[1] == 255 and active_ind[2] == 255:
                self.active_cell = -1
            else:
                self.active_cell = active_ind[2] + 256 * active_ind[1] + 256 * 256 * active_ind[0]
            # print(self.active_cell)
            self.pick.DisableWriting()
            self.window.picker = False
            self.update_instance_cubes_color_array()

        if self.window.show_segments and self.window.not_show_segments:
            self.window.show_segments = False
            self.update_instance_cubes_color_array()
        if self.window.show_segments and not self.window.not_show_segments:
            if self.window.show_active_segments and self.window.not_show_active_segments:
                self.window.show_active_segments = False
            self.update_instance_cubes_color_array()

        self.window.swap_buffers()
