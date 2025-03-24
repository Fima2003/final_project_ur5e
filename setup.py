import mujoco
import numpy as np
import mujoco_viewer
import dm_control


def setup():
    XML_PATH = "ur5e_scene.xml"
    
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    # Set the camera parameters before rendering:
    viewer.cam.azimuth = 0        # horizontal angle, in degrees
    viewer.cam.elevation = -20     # vertical angle, in degrees
    viewer.cam.distance = 3.0       # zoom, distance from the center
    viewer.cam.lookat[:] = [0.3, 0.0, 0.4]  # the point the camera looks at

    with open(XML_PATH, 'r') as f:
        xml_string = f.read()
    physics = dm_control.mujoco.Physics.from_xml_string(xml_string)

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id].copy()
        data.ctrl[:] = model.key_ctrl[key_id].copy()

    # Make sure the state is updated:
    mujoco.mj_forward(model, data)

    return model, data, viewer, physics
