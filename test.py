from loop_rate_limiters import RateLimiter
from mink import SO3
import mujoco
import mujoco.viewer
from UR5e import UR5e
from pathlib import Path
import numpy as np
from ur_analytic_ik import ur5e


_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"
MINK_XML = _HERE / "mink_scene.xml"
FREQUENCY = 200.0

# Load actual scene
model = mujoco.MjModel.from_xml_path(_XML.as_posix())
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

# Initialize the UR5e robot with the Mink scene(simplified version without the shark)
robot = UR5e(MINK_XML)


# Fetch shark id and initial position
shark_body_id = model.body("shark_body").id
shark_body_position = data.xpos[shark_body_id]

# Fetch open box id and initial position
box_body_id = model.body("open_box_body").id
open_box_position = data.xpos[box_body_id]

print(f"Shark's initial position={shark_body_position}")
print(f"Open box position={open_box_position}")


with mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=True,
) as viewer:

    ###########################
    ### SOME INITIALIZATION ###
    ###########################
    # SETUP THE CAMERA VIEW
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -15
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

    # RATE LIMITER
    rate = RateLimiter(frequency=FREQUENCY, warn=False)

    ###########################
    ###      SIMULATION     ###
    ###########################
    # SKIP FIRST 100 STEPS FOR FISH TO FALL
    for i in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # OBTAIN FISH POSITION AND ROTATION
    final_shark_body_position = data.xpos[shark_body_id]

    # SET ROBOT'S DESTINATION AS SHARK'S POSITION
    robot.set_destination(
        np.array(final_shark_body_position + [0.05, -0.3, 0]),
        SO3.from_matrix(
            np.array(
                [
                    [0, -1, 0],
                    [0, 0, 1],
                    [-1, 0, 0]
                ]
            )
        )
    )
    move_control = robot.go(frequency=FREQUENCY)

    # MOVE ROBOT TO THE SHARK
    for i in range(len(move_control)):
        data.ctrl = move_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    stay_control = robot.stay(400)
    for i in range(len(stay_control)):
        data.ctrl = stay_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # GRIPPER IN ACTION
    grip_control = robot.grip()
    for i in range(150):
        data.ctrl = grip_control[0]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    stay_control = robot.stay(dur=400)
    print(stay_control)
    for i in range(len(stay_control)):
        data.ctrl = stay_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # MOVE ROBOT WITH SHARK TO THE OPEN BOX
    robot.set_destination(
        np.array(open_box_position + [0.0, -0.3, 0.3]),
        SO3.from_matrix(
            np.array(
                [
                    [0, -1, 0],
                    [0, 0, 1],
                    [-1, 0, 0]
                ]
            )
        )
    )
    move_control = robot.go(frequency=FREQUENCY)
    for i in range(len(move_control)):
        data.ctrl = move_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

    # CONTINUE MOVEMENT OF ROBOT
    stay_control = robot.stay(4000)
    for i in range(len(stay_control)):
        data.ctrl = stay_control[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()
