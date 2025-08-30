from loop_rate_limiters import RateLimiter
from mink import SO3
import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np

from Arm import Arm


def run_the_mf(robot: Arm):

    _HERE = Path(__file__).parent
    _XML = _HERE / "scene.xml"
    FREQUENCY = 200.0

    # Load actual scene
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # Fetch shark id and initial position
    shark_body_id = model.body("shark_body").id
    shark_body_position = data.xpos[shark_body_id]

    # Fetch open box id and initial position
    box_body_id = model.body("open_box_body").id
    open_box_position = data.xpos[box_body_id]

    print(f"Shark's initial position={shark_body_position}")
    print(f"Open box position={open_box_position}")

    def implement_control(control):
        """Function to implement control commands."""
        for i in range(len(control)):
            data.ctrl = control[i]
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()

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
        viewer.cam.distance = 2.5
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
        final_shark_body_position = data.xpos[shark_body_id].copy()

        # Step 1. MOVE ROBOT TO INITIAL COMFORTABLE FOR PICKING POSITION
        configuration = np.array([0, 1.15, 0.23, 0, 0, -np.pi/2])
        configuration_control = robot.simulate_configuration(
            configuration, run_for=10, stay_for=400)
        implement_control(configuration_control)

        # Step 2. SET ROBOT'S DESTINATION ABOVE SHARK'S POSITION
        robot.set_destination(
            np.array(final_shark_body_position + [-0.02, 0.3, 0.39]),
            SO3.from_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, -1],
                        [-1, 0, 0]
                    ]
                )
            )
        )
        move_control = robot.go(frequency=FREQUENCY, run_for=300, stay_for=300)
        implement_control(move_control)

        # STEP 3. SET ROBOT'S DESTINATION BELOW SHARK'S POSITION
        robot.set_destination(
            np.array(final_shark_body_position + [-0.02, 0.3, 0.1]),
            SO3.from_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, -1],
                        [-1, 0, 0]
                    ]
                )
            )
        )
        move_control = robot.go(frequency=FREQUENCY, run_for=100, stay_for=300)
        implement_control(move_control)

        # STEP 4. GRIP THE FISH
        grip_control = robot.grip(run_for=300, stay_for=300)
        implement_control(grip_control)
        print("Gripper closed, fish is caught.")

        # STEP 5. SET ROBOT'S DESTINATION AS SLIGHTLY ABOVE SHARK'S POSITION
        robot.set_destination(
            np.array(final_shark_body_position + [0.0, 0.35, 0.4]),
            SO3.from_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, -1],
                        [-1, 0, 0]
                    ]
                )
            )
        )
        move_control = robot.go(frequency=FREQUENCY, run_for=100, stay_for=300)
        implement_control(move_control)

        # STEP 6. MOVE ROBOT WITH SHARK TO THE COMFORTABLE POSITION
        configuration = np.array([
            0, 1.15, 0.23, 0,
            move_control[100][4],
            move_control[100][5]
        ])
        configuration_control = robot.simulate_configuration(
            configuration, run_for=10, stay_for=400)
        implement_control(configuration_control)

        # STEP 7. MOVE ROBOT WITH SHARK TO THE OPEN BOX
        robot.set_destination(
            np.array(open_box_position + [0, -0.26, 0.6]),
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
        move_control = robot.go(frequency=FREQUENCY, run_for=300, stay_for=300)
        implement_control(move_control)
        
        # STEP 8. LOWER ROBOT WITH SHARK TO THE OPEN BOX
        robot.set_destination(
            np.array(open_box_position + [0, -0.3, 0.6]),
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
        move_control = robot.go(frequency=FREQUENCY, run_for=300, stay_for=300)
        implement_control(move_control)
        
        # STEP 8. LOWER ROBOT WITH SHARK TO THE OPEN BOX
        robot.set_destination(
            np.array(open_box_position + [0, -0.3, 0.4]),
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
        move_control = robot.go(frequency=FREQUENCY, run_for=300, stay_for=300)
        implement_control(move_control)
        
        # STEP 9. UNGRIP CONTROL
        ungrip_control = robot.ungrip(run_for=600, stay_for=3000)
        implement_control(ungrip_control)
