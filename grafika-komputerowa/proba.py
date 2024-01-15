import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import math

# Inicjalizacja GLFW
if not glfw.init():
    raise Exception("GLFW initialization failed")

# Tworzenie okna
window = glfw.create_window(800, 600, "Prosta Animacja w OpenGL", None, None)
if not window:
    glfw.terminate()
    raise Exception("Tworzenie okna GLFW nie powiodło się")

# Ustawienie okna jako bieżącego kontekstu
glfw.make_context_current(window)

width, height = 800, 600
aspect_ratio = width / height

projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, aspect_ratio, 0.1, 100.0)
view = pyrr.matrix44.create_look_at(pyrr.Vector3([3, 3, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

# Wierzchołki trójkąta (x, y, r, g, b)
vertices = np.array([
    -0.5, -0.5, 1.0, 0.0, 0.0,
    0.5, -0.5, 0.0, 1.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 1.0
], dtype=np.float32)

# VAO i VBO
vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Konfiguracja atrybutów wierzchołków
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(8))

# Vertex shader
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec3 a_color;
out vec3 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 0.0, 1.0);
    color = a_color;
}
"""

# Fragment shader
fragment_shader_source = """
#version 330 core
in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0); // Kolor trójkąta (pomarańczowy)
}
"""

# Kompilacja vertex shadera
vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)

# Kompilacja fragment shadera
fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

# Utworzenie programu shaderowego i załączenie shaderów
shader_program = compileProgram(vertex_shader, fragment_shader)

# Ustalanie lokalizacji zmiennych uniform
model_loc = glGetUniformLocation(shader_program, "model")
view_loc = glGetUniformLocation(shader_program, "view")
projection_loc = glGetUniformLocation(shader_program, "projection")

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800/600, 0.1, 100)
view = pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,3]), pyrr.Vector3([0,0,0]), pyrr.Vector3([0,1,0]))

# Główna pętla aplikacji
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Wyczyszczenie bufora koloru
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    # Aktualizacja macierzy modelu (obrót trójkąta)
    angle = glfw.get_time() * 2.0
    model = np.identity(4, dtype=np.float32)
    model = np.dot(model, np.array([[math.cos(angle), -math.sin(angle), 0.0, 0.0],
                                    [math.sin(angle), math.cos(angle), 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))

    
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)  
    glUseProgram(shader_program)
    # Przekazanie macierzy do shadera
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T)

    # Renderowanie trójkąta
    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)

    glfw.swap_buffers(window)

# Zakończenie aplikacji
glfw.terminate()
