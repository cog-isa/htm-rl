from OpenGL.GL import *
import pyrr
import numpy as np

class Picker():

    def __init__(self, width, height, num_of_cells):
        self.FBO = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.FBO)

        pick_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, pick_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pick_texture, 0)

        depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)

        glReadBuffer(GL_NONE)
        glDrawBuffer(GL_COLOR_ATTACHMENT0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.instance_pick_color = glGenBuffers(1)
        self.instance_array_pick_color = []
        x, y, z = 0, 0, 0
        for cell in range(num_of_cells):
            z = cell % 256
            y = (cell // 256) % 256
            x = ((cell // 256) // 256) % 256
            default_color = pyrr.Vector3([0., 0., 0.])
            default_color.x = x / 255.
            default_color.y = y / 255.
            default_color.z = z / 255.
            self.instance_array_pick_color.append(default_color)

        self.instance_array_pick_color = np.array(self.instance_array_pick_color, np.float32).flatten()

    def EnableWriting(self):
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.FBO)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_pick_color)
        glBufferData(GL_ARRAY_BUFFER, self.instance_array_pick_color.nbytes, self.instance_array_pick_color,
                     GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)

    def DisableWriting(self):
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    def read(self, x, y):
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.FBO)
        glReadBuffer(GL_COLOR_ATTACHMENT0)

        data = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)

        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        return data