import freetype as ft
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import numpy as np
import operator

_textVertexShaderSource = r'''
#version 330 core
layout (location = 0) in vec3 aVec;
layout (location = 1) in vec2 aTexPos;
out vec2 bTexPos;

uniform mat4 projection;

void main(){
    gl_Position = projection * vec4(aVec, 1.0);
    bTexPos = aTexPos;
}  
'''

_textFragmentShaderSource = r'''
#version 330 core
in vec2 bTexPos;
out vec4 color;

uniform vec3 textColor;
uniform sampler2D textureSample;
uniform vec2 textureSize;

void main(){
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(textureSample, bTexPos).r);
    color = vec4(textColor, 1.0) * sampled;
}  
'''


def _get_rendering_buffer(xpos, ypos, w, h, zfix=0.0):
    return np.asarray([
        xpos, ypos - h, zfix, 0.0, 1.0,
        xpos, ypos, zfix, 0.0, 0.0,
              xpos + w, ypos, zfix, 1.0, 0.0,
        xpos, ypos - h, zfix, 0.0, 1.0,
              xpos + w, ypos, zfix, 1.0, 0.0,
              xpos + w, ypos - h, zfix, 1.0, 1.0
    ], np.float32)

def _create_text_texture(bitmapArray):
    height, width = bitmapArray.shape

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)


    # pass texture data

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width,
        height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        bitmapArray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )

    # set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D, 0)

    return texture

def get_numpy_float32_array_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, ft.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, ft.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')

class TextDrawer:

    def __init__(self):
        self.face = None
        self.stroker = None
        self.foreTextures = dict()
        self.backTextures = dict()

        # compile rendering program
        self.shader = compileProgram(compileShader(_textVertexShaderSource, GL_VERTEX_SHADER),
                                     compileShader(_textFragmentShaderSource, GL_FRAGMENT_SHADER))

        # make projection uniform
        self.textureSize_loc = glGetUniformLocation(self.shader, "textureSize")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.textColor_loc = glGetUniformLocation(self.shader, "textColor")

        # create rendering buffer
        self.vbo = glGenBuffers(1)

        # initialize VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, _get_rendering_buffer(0, 0, 0, 0).nbytes, _get_rendering_buffer(0, 0, 0, 0), GL_STATIC_DRAW)


        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float),
                              ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self.zNear = -1.0
        self.zFar = 1.0

    def delete(self):
        self.foreTextures.clear()
        self.backTextures.clear()
        self.face = None
        self.stroker = None
        self.renderProgram.delete()
        self.projectionUniform = None
        self.textColorUniform = None
        self.textureSizeUniform = None
        self.vbo.delete()
        glDeleteVertexArrays(1, [self.vao])

    def load_font(self, fontFilename, fontSize, outlineSize=2 * 64):
        self.foreTextures.clear()
        self.backTextures.clear()

        self.face = ft.Face(fontFilename)
        self.face.set_char_size(fontSize)

        self.outlineSize = outlineSize
        self.stroker = ft.Stroker()
        self.stroker.set(outlineSize, ft.FT_STROKER_LINECAPS['FT_STROKER_LINECAP_ROUND'],
                         ft.FT_STROKER_LINEJOINS['FT_STROKER_LINEJOIN_ROUND'], 0)

        # load all ASCII characters
        for i in range(128):
            self.load_character(chr(i))

    def load_character(self, character):

        if character not in self.foreTextures:
            # load background glyph
            # the render option will lead to an outline glyph (not rendered)
            self.face.load_char(character, ft.FT_LOAD_FLAGS['FT_LOAD_DEFAULT'])
            backGlyph = ft.FT_Glyph()
            ft.FT_Get_Glyph(self.face.glyph._FT_GlyphSlot, ft.byref(backGlyph))
            backGlyph = ft.Glyph(backGlyph)
            # add border to the glyph
            error = ft.FT_Glyph_StrokeBorder(ft.byref(backGlyph._FT_Glyph), self.stroker._FT_Stroker, False, False)
            if error:
                raise ft.FT_Exception(error)

            # the render option will lead to a rendered glyph
            backBitmapGlyph = backGlyph.to_bitmap(ft.FT_RENDER_MODES['FT_RENDER_MODE_NORMAL'], 0)

            backBitmap = backBitmapGlyph.bitmap
            backHeight, backWidth = backBitmap.rows, backBitmap.width
            backBitmap = np.array(backBitmap.buffer, dtype=np.uint8).reshape((backHeight, backWidth))

            backTexture = _create_text_texture(backBitmap)

            backSlot = CharacterSlot(backTexture, backBitmapGlyph)
            self.backTextures[character] = backSlot

            # load foreground glyph
            self.face.load_char(character, ft.FT_LOAD_FLAGS['FT_LOAD_RENDER'])
            foreBitmap = self.face.glyph.bitmap
            foreHeight, foreWidth = foreBitmap.rows, foreBitmap.width
            foreBitmap = np.array(foreBitmap.buffer, dtype=np.uint8).reshape((foreHeight, foreWidth))

            foreTexture = _create_text_texture(foreBitmap)

            foreSlot = CharacterSlot(foreTexture, self.face.glyph)
            self.foreTextures[character] = foreSlot

    def get_character(self, ch):
        if ch not in self.foreTextures:
            self.load_character(ch)
        return (self.foreTextures[ch], self.backTextures[ch])

    def _draw_text(self, text, textPos, windowSize, scale, linespread, foreColor, backColor):
        if len(text) == 0:
            return

        blendEnabled = glIsEnabled(GL_BLEND)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)

        foreColor = np.asarray(foreColor, np.float32)
        backColor = np.asarray(backColor, np.float32)

        projectionMat = pyrr.matrix44.create_orthogonal_projection_matrix(0, windowSize[0], 0, windowSize[1], self.zNear, self.zFar)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projectionMat)

        lineY = textPos[1]
        nowX = textPos[0]

        # split text into lines
        lines = text.split('\n')

        for line in lines:

            if len(line) > 0:
                # analyze this line
                bearings = []
                for ch in line:
                    _, backSlot = self.get_character(ch)
                    bearings.append(backSlot.bearing)

                minBearings_X = min(zip(*bearings), key=operator.itemgetter(0))[0] * scale[0]
                maxBearings_Y = max(zip(*bearings), key=operator.itemgetter(1))[1] * scale[1]
                nowX = nowX - minBearings_X

                for ch in line:
                    foreSlot, backSlot = self.get_character(ch)

                    # draw the background
                    xpos = nowX + backSlot.bearing[0] * scale[0]
                    yLowerd = (maxBearings_Y - backSlot.bearing[1] * scale[1])
                    ypos = lineY - yLowerd

                    w = backSlot.textureSize[0] * scale[0]
                    h = backSlot.textureSize[1] * scale[1]


                    glUniform2fv(self.textureSize_loc, 1, np.array((w, h), np.float32))
                    glBindTexture(GL_TEXTURE_2D, backSlot.texture)
                    glUniform3fv(self.textColor_loc, 1, backColor)

                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                    glBufferData(GL_ARRAY_BUFFER, _get_rendering_buffer(xpos, ypos, w, h, 0.99).nbytes,
                                 _get_rendering_buffer(xpos, ypos, w, h, 0.99), GL_STATIC_DRAW)

                    glDrawArrays(GL_TRIANGLES, 0, 6)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    # draw the foreground
                    xpos = nowX + foreSlot.bearing[0] * scale[0]
                    yLowerd = (maxBearings_Y - foreSlot.bearing[1] * scale[1])
                    ypos = lineY - yLowerd

                    w = foreSlot.textureSize[0] * scale[0]
                    h = foreSlot.textureSize[1] * scale[1]

                    glBindTexture(GL_TEXTURE_2D, foreSlot.texture)
                    glUniform3fv(self.textColor_loc, 1, foreColor)
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                    glBufferData(GL_ARRAY_BUFFER, _get_rendering_buffer(xpos, ypos, w, h, 0.99).nbytes,
                                 _get_rendering_buffer(xpos, ypos, w, h, 0.99), GL_STATIC_DRAW)

                    glDrawArrays(GL_TRIANGLES, 0, 6)
                    glBindTexture(GL_TEXTURE_2D, 0)

                    # the advance is number of 1/64 pixels
                    nowX += ((foreSlot.advance + 2.0 * self.outlineSize) / 64.0) * scale[0]

            yOffset = self.get_character('X')[1].textureSize[1] * scale[1] * linespread
            lineY -= yOffset

        glBindVertexArray(0)

        if not blendEnabled:
            glDisable(GL_BLEND)


    def draw_text(self, text, textPos, windowSize, foreColor=(1.0, 1.0, 1.0), backColor=(0.0, 0.0, 0.0), scale=(1.0, 1.0),
                  linespread=1.5):
        return self._draw_text(text, textPos, windowSize, scale, linespread, foreColor, backColor)

