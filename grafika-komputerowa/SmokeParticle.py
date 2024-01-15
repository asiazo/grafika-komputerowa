import numpy as np
import random
import ctypes
from OpenGL.GL import *
import pyrr
import glfw
from TextureLoader import load_texture

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW initialization failed")
# Create a window
window = glfw.create_window(1280, 720, "Smoke Particles Example", None, None)

# Check if the window was created successfully
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

# Set the window's position
glfw.set_window_pos(window, 400, 200)

# Make the window's context current
glfw.make_context_current(window)


class SmokeParticles:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particle_positions = np.array([self.random_position() for _ in range(num_particles)], dtype=np.float32)

        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.particle_positions.nbytes, self.particle_positions, GL_STATIC_DRAW)

        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def random_position(self):
        return [random.uniform(-5, 5), random.uniform(0, 10), random.uniform(-5, 5)]
    
    def draw(self, shader_program, model_loc, view_matrix, projection_matrix, smoke_texture):
        glUseProgram(shader_program)
        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, smoke_texture)

        for position in self.particle_positions:
            model_matrix = pyrr.matrix44.create_from_translation(pyrr.Vector3(position))
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
            glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, projection_matrix)

            glDrawArrays(GL_TRIANGLES, 0, 6)  # Assuming each particle is a quad (2 triangles)

        glBindVertexArray(0)
        glUseProgram(0)

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program(vertex_src, fragment_src):
    program = glCreateProgram()
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program
    
vertex_src = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_src = """
#version 330 core
out vec4 FragColor;

uniform sampler2D smokeTexture;

void main()
{
    FragColor = texture(smokeTexture, gl_PointCoord); // Use gl_PointCoord for point sprites
}
"""

shader_program = create_shader_program(vertex_src, fragment_src)

# Load the smoke texture
textures = glGenTextures(1)
smoke_texture = load_texture("textures/smoke.jpg")

# Create an instance of the SmokeParticles class
smoke_particles = SmokeParticles()

# Create view and projection matrices
view_matrix = pyrr.Matrix44.look_at(pyrr.Vector3([0, 0, 5]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
projection_matrix = pyrr.Matrix44.perspective_projection(45.0, 1280 / 720, 0.1, 100.0)

# Get the location of the "model" uniform in the shader
model_loc = glGetUniformLocation(shader_program, "model")

# Main application loop
while not glfw.window_should_close(window):
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Handle user input or other game logic here

    # Render the smoke particles
    smoke_particles.draw(shader_program, model_loc, view_matrix, projection_matrix, smoke_texture)

    # Swap the front and back buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()

# Terminate GLFW when the window is closed
glfw.terminate()
