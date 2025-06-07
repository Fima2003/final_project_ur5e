from loop_rate_limiters import RateLimiter
from mink import SO3
import mujoco
import mujoco.viewer
from UR5e import UR5e
from pathlib import Path
import numpy as np


_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"
_MINK_XML = _HERE / "mink_scene.xml"

model = mujoco.MjModel.from_xml_path(_XML.as_posix())
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

frequency = 200.0

shark_body_id = model.body("shark_body").id
shark_body_position = data.xpos[shark_body_id]

rot = data.xmat.reshape(model.nbody, 3, 3)[shark_body_id]
quat = data.xquat.reshape(model.nbody, 4)[shark_body_id]

print(f"Initial position={shark_body_position} and rotation={quat}")

robot = UR5e(_MINK_XML)


with mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=True,
) as viewer:

    # SETUP THE CAMERA VIEW
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -15
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
    
    # RATE LIMITER
    rate = RateLimiter(frequency=frequency, warn=False)
    
    # SKIP FIRST 100 STEPS FOR FISH TO FALL
    for i in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # OBTAIN FISH POSITION AND ROTATION
    final_shark_body_position = data.xpos[shark_body_id]
    final_rot = data.xmat.reshape(model.nbody, 3, 3)[shark_body_id]

    # SET ROBOT'S DESTINATION AND MOVE IT IN BG SIMULATOR
    robot.set_destination(np.array(final_shark_body_position + [-0, 0, 0.6]),
                        SO3.from_matrix(np.array(
                            [
                                [0, 1, 0],
                                [0, 0, -1],
                                [1, 0, 0]
                            ]
                        ))
                    )
    move_control = robot.go(frequency=frequency)
    
    for i in range(len(move_control)):
        data.ctrl = move_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()
        
    for i in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # GRIPPER IN ACTION
    grip_control = robot.grip()
    for i in range(len(grip_control)):
        data.ctrl = grip_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # CONTINUE MOVEMENT OF ROBOT
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()
