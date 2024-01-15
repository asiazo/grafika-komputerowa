import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import numpy as np
from TextureLoader import load_texture
from ObjLoader import ObjLoader
from camera import Camera
#from SmokeParticle import SmokeParticles


vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 v_texture;
out vec3 FragPos;
out vec3 Normal;

void main()
{
    v_texture = a_texture;
    FragPos = vec3(view * model * vec4(a_position, 1.0));
    Normal = mat3(transpose(inverse(view * model))) * a_normal;

    gl_Position = projection * view * model * vec4(a_position, 1.0);
}
"""

fragment_src = """
# version 330
in vec2 v_texture;
in vec3 FragPos;
in vec3 Normal;

out vec4 out_color;

uniform sampler2D s_texture;
uniform vec3 lightPos; // Position of the light source
uniform vec3 lightColor; // Color of the light
uniform vec3 viewPos; // Position of the camera

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * texture(s_texture, v_texture).rgb;
    out_color = vec4(result, 1.0);
}
"""


# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

# load here the 3d meshes
chibi_indices, chibi_buffer = ObjLoader.load_model("meshes/chibi.obj")

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

light_pos = pyrr.Vector3([1.2, 1.0, 2.0])  # Adjust the position as needed
light_color = pyrr.Vector3([1.0, 1.0, 1.0])  # White light

# Get uniform locations for the light
light_pos_loc = glGetUniformLocation(shader, "lightPos")
light_color_loc = glGetUniformLocation(shader, "lightColor")
view_pos_loc = glGetUniformLocation(shader, "viewPos")

# Set the uniform values
glUseProgram(shader)
glUniform3fv(light_pos_loc, 1, light_pos)
glUniform3fv(light_color_loc, 1, light_color)
glUniform3fv(view_pos_loc, 1, pyrr.Vector3([0, 0, 8]))

# VAO and VBO
VAO = glGenVertexArrays(2)
VBO = glGenBuffers(2)
# EBO = glGenBuffers(1)

# Chibi VAO
glBindVertexArray(VAO[0])
# Chibi Vertex Buffer Object
glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, chibi_buffer.nbytes, chibi_buffer, GL_STATIC_DRAW)

# chibi vertices
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, chibi_buffer.itemsize * 8, ctypes.c_void_p(0))
# chibi textures
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, chibi_buffer.itemsize * 8, ctypes.c_void_p(12))
# chibi normals
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, chibi_buffer.itemsize * 8, ctypes.c_void_p(20))
glEnableVertexAttribArray(2)

textures = glGenTextures(2)
load_texture("meshes/chibi.png", textures[0])

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280 / 720, 0.1, 100)
chibi_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -5, -10]))

# eye, target, up
view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 8]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)


# Define the floor vertices, texture coordinates, and normals
floor_vertices = [  # x, y, z, s, t, nx, ny, nz
    -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -10.0, 10.0, 0.0, 0.0, 1.0, 0.0,
    -10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 1.0, 0.0,
    10.0, 0.0, 10.0, 10.0, 10.0, 0.0, 1.0, 0.0
]

# Convert the floor_vertices to a numpy array
floor_vertices = np.array(floor_vertices, dtype=np.float32)
# Generowanie identyfikatora tekstury
ground_texture = glGenTextures(1)
load_texture("textures/73.jpg", ground_texture)


# Floor VAO
glBindVertexArray(VAO[1])
# Floor Vertex Buffer Object
glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, floor_vertices.nbytes, floor_vertices, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
glEnableVertexAttribArray(2)


first_mouse = True
lastX, lastY = 0, 0

last_frame = glfw.get_time()
# Define the camera speed
camera_speed = 5.0 

# Utwórz instancję kamery
camera = Camera()

#smoke_particles = SmokeParticles(camera)

# Function to be called for every mouse event
def mouse_callback(window, xpos, ypos):
    # Assuming you're storing the last mouse position globally
    global lastX, lastY, first_mouse

    if first_mouse:
        lastX, lastY = xpos, ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos  # Reversed since y-coordinates range from bottom to top
    lastX, lastY = xpos, ypos

    # Call the camera method to process this mouse movement
    camera.process_mouse_movement(xoffset, yoffset)

glfw.set_cursor_pos_callback(window, mouse_callback)
# Function to process keyboard input in the main loop
def process_input(window, delta_time):
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera.process_keyboard("FORWARD", camera_speed * delta_time)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera.process_keyboard("BACKWARD", camera_speed * delta_time)
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera.process_keyboard("LEFT", camera_speed * delta_time)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera.process_keyboard("RIGHT", camera_speed * delta_time)

scale_factor = 0.25 
scale_matrix = pyrr.matrix44.create_from_scale(pyrr.Vector3([scale_factor, scale_factor, scale_factor]))

chibi_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))  # Postać na podłożu


# Definicje różnych widoków kamery jako macierze widoku
view1 = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 8]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
view2 = pyrr.matrix44.create_look_at(pyrr.Vector3([5, 5, 5]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
view3 = pyrr.matrix44.create_look_at(pyrr.Vector3([-5, -5, -5]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

camera_views = [view1, view2, view3]
current_view_index = 0  # Aktualnie wybrany widok


# Funkcja zmiany widoku kamery
def change_camera_view():
    global current_view_index
    current_view_index = (current_view_index + 1) % len(camera_views)
    
# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    current_frame = glfw.get_time()
    delta_time = current_frame - last_frame
    last_frame = current_frame

    # Input
    process_input(window, delta_time)

    # Obsługa zmiany widoku kamery po naciśnięciu klawisza V
    if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS:
        change_camera_view()
    
    # Ustaw aktywny widok kamery
    view = camera.get_view_matrix()
    #view = camera_views[current_view_index]
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniform3fv(light_pos_loc, 1, light_pos)

    rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())
    model = pyrr.matrix44.multiply(scale_matrix, rot_y)

    # Set the view and projection matrix uniforms if they change
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # Draw the floor
    glBindVertexArray(VAO[1])
    glBindTexture(GL_TEXTURE_2D, ground_texture) 
    floor_model = pyrr.matrix44.create_identity(dtype=np.float32)  # Identity matrix for the floor model
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, floor_model)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)


    # draw the chibi character
    glBindVertexArray(VAO[0])
    glBindTexture(GL_TEXTURE_2D, textures[0])
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(chibi_indices))

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()