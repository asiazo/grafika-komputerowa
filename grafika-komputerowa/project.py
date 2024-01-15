import glfw
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
from ObjLoader import ObjLoader
import dearpygui.dearpygui as dpg
import random  

dpg.create_context()

# Funkcje wywoływane przez przyciski
def save_callback(sender, app_data):
    print("Ustawienia zostały zapisane!")

def load_callback(sender, app_data):
    print("Ustawienia zostały wczytane!")

with dpg.theme() as red_button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [150, 0, 0, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [200, 0, 0, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [250, 0, 0, 255])

with dpg.theme() as green_button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 150, 0, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [0, 200, 0, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 250, 0, 255])

with dpg.window(label="Ustawienia", width=600, height=300):
    save_button = dpg.add_button(label="Zapisz ustawienia", callback=save_callback)
    load_button = dpg.add_button(label="Wczytaj ustawienia", callback=load_callback)
    
    dpg.bind_item_theme(save_button, red_button_theme)
    dpg.bind_item_theme(load_button, green_button_theme)

dpg.create_viewport(title='Game', width=600, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
# Parametry światła
light_pos_camera = pyrr.Vector3([0.0, 0.0, 0.0])  # Inicjalizacja pozycji światła z pozycji kamery
light_pos = pyrr.Vector3([2.0, 4.0, 2.0])  # Pozycja światła
light_ambient = pyrr.Vector3([0.2, 0.2, 0.2])  # Światło otoczenia
light_diffuse = pyrr.Vector3([0.5, 0.5, 0.5])  # Światło rozproszone
light_specular = pyrr.Vector3([1.0, 1.0, 1.0])  # Światło lustrzane

light_pos1 = pyrr.Vector3([2.0, 4.0, 2.0])  # Pozycja światła 1
light_ambient1 = pyrr.Vector3([0.2, 0.2, 0.2])  # Światło otoczenia 1
light_diffuse1 = pyrr.Vector3([0.5, 0.5, 0.5])  # Światło rozproszone 1
light_specular1 = pyrr.Vector3([1.0, 1.0, 1.0])  # Światło lustrzane 1

light_pos2 = pyrr.Vector3([-2.0, 4.0, -2.0])  # Pozycja światła 2
light_ambient2 = pyrr.Vector3([0.2, 0.2, 0.2])  # Światło otoczenia 2
light_diffuse2 = pyrr.Vector3([0.5, 0.5, 0.5])  # Światło rozproszone 2
light_specular2 = pyrr.Vector3([1.0, 1.0, 1.0])  # Światło lustrzane 2
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 v_texture;
out vec3 Normal;
out vec3 FragPos;

void main()
{
    v_texture = a_texture;
    Normal = a_normal;
    FragPos = vec3(model * vec4(a_position, 1.0));

    gl_Position = projection * view * model * vec4(a_position, 1.0);
}
"""
fragment_src = """
# version 330

in vec2 v_texture;
in vec3 Normal;  // Normalny
in vec3 FragPos; // Pozycja fragmentu
uniform sampler2D s_texture;

out vec4 out_color;

uniform vec3 lightPos;  // Pozycja światła

uniform vec3 viewPos;   // Pozycja kamery
uniform vec3 lightColor;

uniform vec3 light_pos_camera;  // Pozycja kamery

void main()
{
    / Oświetlenie otoczenia
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * ambientLight;

    scss
    Copy code
    // Oświetlenie rozproszone
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diffuseLight * diff * lightColor;

    // Oświetlenie lustrzane
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * specularLight * spec * lightColor;

    vec3 textureColor = texture(s_texture, v_texture).rgb;
    vec3 result = (ambient + diffuse + specular) * textureColor;
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

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

# Zarejestruj callback
glfw.set_key_callback(window, key_callback)

# Wierzchołki płaszczyzny (np. kwadrat w poziomie x-z)
plane_vertices = [
    -10.0, 0.0, -10.0, 0.0, 0.0,  # Lewy dolny róg
    -10.0, 0.0,  10.0, 0.0, 1.0,  # Lewy górny róg
     10.0, 0.0, -10.0, 1.0, 0.0,  # Prawy dolny róg
     10.0, 0.0,  10.0, 1.0, 1.0   # Prawy górny róg
]

plane_indices = [
    0, 1, 2,  # Pierwszy trójkąt
    1, 2, 3   # Drugi trójkąt
]
# Wierzchołki tła (prostokąta)
background_vertices = [
    -10.0, -10.0, 0.0, 0.0, 0.0,
    -10.0,  10.0, 0.0, 0.0, 1.0,
     10.0, -10.0, 0.0, 1.0, 0.0,
     10.0,  10.0, 0.0, 1.0, 1.0
]

background_indices = [
    0, 1, 2,
    1, 2, 3
]

# Konwersja do formatu, który rozumie OpenGL
plane_vertices = np.array(plane_vertices, dtype=np.float32)
plane_indices = np.array(plane_indices, dtype=np.uint32)

# VAO i VBO dla płaszczyzny
plane_vao = glGenVertexArrays(1)
glBindVertexArray(plane_vao)

# VBO
plane_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, plane_vbo)
glBufferData(GL_ARRAY_BUFFER, plane_vertices.nbytes, plane_vertices, GL_STATIC_DRAW)

# EBO
plane_ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, plane_indices.nbytes, plane_indices, GL_STATIC_DRAW)

# Konfiguracja atrybutów wierzchołków
# Pozycje
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
# Tekstury
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

# Generowanie identyfikatora tekstury
ground_texture = glGenTextures(1)


# Ładowanie tekstury
load_texture("textures/73.jpg", ground_texture)

def update_chibi_scale(time):
    scale_factor = math.sin(time) * 0.1 + 1.0  # Delikatne pulsowanie skali
    return pyrr.Matrix44.from_scale(pyrr.Vector3([scale_factor, scale_factor, scale_factor]))


# Camera attributes
cam_pos = pyrr.Vector3([0, 0, 8])
cam_yaw = -90.0  # Horizontal angle
cam_pitch = 0.0  # Vertical angle
last_x, last_y = 800 / 2, 600 / 2  # Last mouse position
first_mouse = True

# load here the 3d meshes
chibi_indices, chibi_buffer = ObjLoader.load_model("meshes/chibi.obj")

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

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
chibi_rotation_angle = 0.000

def update_camera_position(time):
    global cam_pos, view

    cam_pos.x = 10 * math.sin(time)
    cam_pos.z = 10 * math.cos(time)

    # Aktualizacja macierzy widoku
    front = pyrr.Vector3([
        math.cos(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch)),
        math.sin(math.radians(cam_pitch)),
        math.sin(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch))
    ])
    view = pyrr.matrix44.create_look_at(cam_pos, cam_pos + front, pyrr.Vector3([0, 1, 0]))
    # Aktualizacja pozycji światła w oparciu o pozycję kamery
    light_pos_camera.x = cam_pos.x
    light_pos_camera.y = cam_pos.y
    light_pos_camera.z = cam_pos.z



def mouse_look_callback(window, xpos, ypos):
    global cam_yaw, cam_pitch, last_x, last_y, first_mouse

    if first_mouse:  # this will only be true in the first iteration
        last_x, last_y = xpos, ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos  # Reversed since y-coordinates range from bottom to top
    last_x = xpos
    last_y = ypos

    sensitivity = 0.1  # Change this value to your liking
    xoffset *= sensitivity
    yoffset *= sensitivity

    cam_yaw += xoffset
    cam_pitch += yoffset

    if cam_pitch > 89.0:
        cam_pitch = 89.0
    if cam_pitch < -89.0:
        cam_pitch = -89.0

    front = pyrr.Vector3([
        math.cos(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch)),
        math.sin(math.radians(cam_pitch)),
        math.sin(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch))
    ])
    global view
    view = pyrr.matrix44.create_look_at(cam_pos, cam_pos + front, pyrr.Vector3([0, 1, 0]))


textures = glGenTextures(2)
load_texture("meshes/chibi.png", textures[0])


glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
# Ustawienie pozycji światła i koloru
light_color_loc = glGetUniformLocation(shader, "lightColor")
light_pos_loc = glGetUniformLocation(shader, "lightPos")
view_pos_loc = glGetUniformLocation(shader, "viewPos")

glUniform3f(light_color_loc, light_diffuse.x, light_diffuse.y, light_diffuse.z)
glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z)
glUniform3f(view_pos_loc, cam_pos.x, cam_pos.y, cam_pos.z)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280 / 720, 0.1, 100)
chibi_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -5, -10]))

# eye, target, up
view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 8]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

glfw.set_cursor_pos_callback(window, mouse_look_callback)
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# Konfiguracja atrybutów wierzchołków
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))


# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Update the chibi model's rotation angle
    chibi_rotation_angle += 0.001  # You can adjust the rotation speed

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())
    model = pyrr.matrix44.multiply(rot_y, chibi_pos)

    # Apply rotation animation
    rotation_matrix = pyrr.Matrix44.from_y_rotation(chibi_rotation_angle)
    model = pyrr.matrix44.multiply(rotation_matrix, model)

    #light_color_loc = glGetUniformLocation(shader, "lightColor1")
    #light_pos_loc = glGetUniformLocation(shader, "lightPos1")
    #glUniform3f(light_color_loc, light_diffuse1.x, light_diffuse1.y, light_diffuse1.z)
    #glUniform3f(light_pos_loc, light_pos1.x, light_pos1.y, light_pos1.z)

    #light_color_loc = glGetUniformLocation(shader, "lightColor2")
    #light_pos_loc = glGetUniformLocation(shader, "lightPos2")
    #glUniform3f(light_color_loc, light_diffuse2.x, light_diffuse2.y, light_diffuse2.z)
    #glUniform3f(light_pos_loc, light_pos2.x, light_pos2.y, light_pos2.z)

    #glBindVertexArray(plane_vao)
    #glBindTexture(GL_TEXTURE_2D, ground_texture)  
    #glDrawElements(GL_TRIANGLES, len(plane_indices), GL_UNSIGNED_INT, None)

    glBindVertexArray(VAO[0])
    glBindTexture(GL_TEXTURE_2D, textures[0])
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(chibi_indices))

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()