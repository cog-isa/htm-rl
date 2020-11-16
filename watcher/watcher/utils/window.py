import glfw
import pyrr
from OpenGL.GL import glViewport, glUniformMatrix4fv, GL_FALSE
from utils.camera import Camera

class Window():

    def __init__(self, width, height, cam_target):
        self.width = width
        self.height = height
        self.cam = Camera(cam_target)

        # control variables
        self.left = False
        self.right = False
        self.forward = False
        self.backward = False
        self.next = False
        self.previous = False
        self.show_segments = False
        self.not_show_segments = True
        self.show_active_segments = False
        self.not_show_active_segments = True
        self.follow_cursor = False

        self.picker = False
        self.picker_right = False
        self.lastX = self.width / 2
        self.lastY = self.height / 2
        self.mouseX = 0
        self.mouseY = 0
        self.velocity = 0.1

        self.proj_loc = None

        # initializing glfw library
        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        # creating the window
        self.window = glfw.create_window(self.width, self.height, "Watcher", None, None)

        # check if window was created
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # set window's position
        glfw.set_window_pos(self.window, 0, 0)

        # set the callback functions
        glfw.set_key_callback(self.window, self.key_input_clb)
        glfw.set_cursor_pos_callback(self.window, self.mouse_look_clb)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_window_size_callback(self.window, self.window_resize_clb)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)

        glfw.make_context_current(self.window)

    def key_input_clb(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.forward = True
        elif key == glfw.KEY_W and action == glfw.RELEASE:
            self.forward = False
        if key == glfw.KEY_S and action == glfw.PRESS:
            self.backward = True
        elif key == glfw.KEY_S and action == glfw.RELEASE:
            self.backward = False
        if key == glfw.KEY_A and action == glfw.PRESS:
            self.left = True
        elif key == glfw.KEY_A and action == glfw.RELEASE:
            self.left = False
        if key == glfw.KEY_D and action == glfw.PRESS:
            self.right = True
        elif key == glfw.KEY_D and action == glfw.RELEASE:
            self.right = False
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.next = True
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.previous = True
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_segments = True
            self.not_show_segments = False
        elif key == glfw.KEY_Q and action == glfw.RELEASE:
            self.not_show_segments = True
        if key == glfw.KEY_E and action == glfw.PRESS:
            self.show_active_segments = True
            self.not_show_active_segments = False
        elif key == glfw.KEY_E and action == glfw.RELEASE:
            self.not_show_active_segments = True
        if key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
            self.follow_cursor = True
        elif key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
            self.follow_cursor = False

    def mouse_look_clb(self, window, xpos, ypos):
        self.mouseX = xpos
        self.mouseY = ypos

        if self.follow_cursor:
            xoffset = xpos - self.lastX
            yoffset = self.lastY - ypos
        else:
            xoffset = 0
            yoffset = 0

        self.lastX = xpos
        self.lastY = ypos

        self.cam.process_mouse_movement(xoffset, yoffset)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.picker = True
        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            self.picker_right = True
        if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            self.follow_cursor = True
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
            self.follow_cursor = False

    def mouse_scroll_callback(self, window, xoffset, yoffset):
        self.velocity += 0.05 * yoffset/abs(yoffset)
        if self.velocity < 0.01:
            self.velocity = 0.01

    def window_resize_clb(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)

    def terminate(self):
        glfw.terminate()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def poll_events(self):
        glfw.poll_events()

    def swap_buffers(self):
        glfw.swap_buffers(self.window)

