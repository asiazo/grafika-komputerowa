import glfw
from OpenGL.GL import *
import pybullet as p
from ctypes import c_void_p

# Inicjalizacja PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Inicjalizacja GLFW
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# Tworzenie okna GLFW
window = glfw.create_window(800, 600, "Physics Simulation", None, None)

# Funkcja wywoływana podczas zmiany rozmiaru okna
def window_resize(window, width, height):
    glViewport(0, 0, width, height)

glfw.set_window_size_callback(window, window_resize)

# Ustalanie koloru tła
glClearColor(0.2, 0.3, 0.3, 1.0)

# Tworzenie tekstury
texture_width, texture_height = 800, 600
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Shader do renderowania tekstury na płaszczyźnie
vertex_shader = """
# version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader = """
# version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, TexCoord);
}
"""

# Kompilacja shadera
shader_program = glCreateProgram()
vertex_shader_object = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader_object, vertex_shader)
glCompileShader(vertex_shader_object)
if not glGetShaderiv(vertex_shader_object, GL_COMPILE_STATUS):
    raise Exception("Vertex shader compilation failed: " + glGetShaderInfoLog(vertex_shader_object))

fragment_shader_object = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader_object, fragment_shader)
glCompileShader(fragment_shader_object)
if not glGetShaderiv(fragment_shader_object, GL_COMPILE_STATUS):
    raise Exception("Fragment shader compilation failed: " + glGetShaderInfoLog(fragment_shader_object))

glAttachShader(shader_program, vertex_shader_object)
glAttachShader(shader_program, fragment_shader_object)
glLinkProgram(shader_program)
if not glGetProgramiv(shader_program, GL_LINK_STATUS):
    raise Exception("Shader program linking failed: " + glGetProgramInfoLog(shader_program))

glDeleteShader(vertex_shader_object)
glDeleteShader(fragment_shader_object)

# Wierzchołki i indeksy płaszczyzny
plane_vertices = [
    -1.0, -1.0, 0.0, 0.0, 0.0,
     1.0, -1.0, 0.0, 1.0, 0.0,
    -1.0,  1.0, 0.0, 0.0, 1.0,
     1.0,  1.0, 0.0, 1.0, 1.0,
]

plane_indices = [
    0, 1, 2,
    1, 2, 3
]

plane_vertices = GLfloat * len(plane_vertices)(*plane_vertices)
plane_indices = GLuint * len(plane_indices)(*plane_indices)

# VAO i VBO dla płaszczyzny
plane_vao = glGenVertexArrays(1)
glBindVertexArray(plane_vao)

plane_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, plane_vbo)
glBufferData(GL_ARRAY_BUFFER, sizeof(plane_vertices), plane_vertices, GL_STATIC_DRAW)

plane_ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(plane_indices), plane_indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), None)
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), c_void_p(3 * sizeof(GLfloat)))

# Główna pętla aplikacji
while not glfw.window_should_close(window):
    # Symulacja kroku czasowego w PyBullet
    p.stepSimulation()

    # Renderowanie do tekstury
    glBindFramebuffer(GL_FRAMEBUFFER, 0)  # Przełączenie na domyślny bufor ramki
    glUseProgram(shader_program)
    glBindTexture(GL_TEXTURE_2D, texture)  # Ustawić jako cel renderowania

    glClear(GL_COLOR_BUFFER_BIT)
    glBindVertexArray(plane_vao)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # Wyświetlanie na ekranie
    glfw.swap_buffers(window)
    glfw.poll_events()

# Zwolnienie zasobów
glDeleteTextures(1, [texture])
glDeleteVertexArrays(1, [plane_vao])
glDeleteBuffers(1, [plane_vbo, plane_ebo])
glfw.terminate()