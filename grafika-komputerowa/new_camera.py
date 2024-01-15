import pyrr
import glfw
import math
from OpenGL.GL import *

class Camera:
    def __init__(self):
        self.camera_pos = pyrr.Vector3([0, 0, 3])
        self.camera_front = pyrr.Vector3([0, 0, -1])
        self.camera_up = pyrr.Vector3([0, 1, 0])
        self.camera_right = pyrr.Vector3([1, 0, 0])
        self.yaw = -90
        self.pitch = 0
        self.lastX = 800 / 2
        self.lastY = 600 / 2
        self.fov = 45

    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

# Mouse callback function
def mouse_callback(window, xpos, ypos, camera, first_mouse):
    sensitivity = 0.1
    
    if first_mouse:
        camera.lastX = xpos
        camera.lastY = ypos
        first_mouse = False

    xoffset = xpos - camera.lastX
    yoffset = camera.lastY - ypos  # Reversed since y-coordinates go from bottom to top
    camera.lastX = xpos
    camera.lastY = ypos

    xoffset *= sensitivity
    yoffset *= sensitivity

    camera.yaw += xoffset
    camera.pitch += yoffset

    # Update camera_front vector
    front = pyrr.Vector3()
    front.x = math.cos(math.radians(camera.yaw)) * math.cos(math.radians(camera.pitch))
    front.y = math.sin(math.radians(camera.pitch))
    front.z = math.sin(math.radians(camera.yaw)) * math.cos(math.radians(camera.pitch))
    camera.camera_front = pyrr.vector3.normalize(front)

# Keyboard input handling
def process_input(window):
    global camera
    camera_speed = 0.05
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera.camera_pos += camera_speed * camera.camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera.camera_pos -= camera_speed * camera.camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera.camera_pos -= pyrr.vector3.normalize(pyrr.vector3.cross(camera.camera_front, camera.camera_up)) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera.camera_pos += pyrr.vector3.normalize(pyrr.vector3.cross(camera.camera_front, camera.camera_up)) * camera_speed

# Initialization
camera = Camera()
first_mouse = True

# Set the mouse callback
glfw.set_cursor_pos_callback(window, mouse_callback)

# Capture the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# Main loop
while not glfw.window_should_close(window):
    process_input(window)
    # Clear the buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Update the view matrix with the camera's new position and orientation
    view = camera.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # Rendering commands...

    # Swap buffers and poll IO events
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

