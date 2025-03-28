from ik_settings import *
from utils import move_arm_to_position, squeeze_object
import matplotlib.pyplot as plt
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib import TeleopMocap
from mink.lie.so3 import SO3
import matplotlib
matplotlib.use("Agg")


_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    # Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between (wrist3, floor) and (wrist3, wall).
    wrist_3_geoms = mink.get_body_geom_ids(
        model, model.body("wrist_3_link").id)
    collision_pairs = [
        (wrist_3_geoms, ["floor", "wall1", "wall2", "wall3"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
    ]

    max_velocities = {
        "shoulder_pan_joint": np.pi,
        "shoulder_lift_joint": np.pi,
        "elbow_joint": np.pi,
        "wrist_1_joint": np.pi,
        "wrist_2_joint": np.pi,
        "wrist_3_joint": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    # Initialize key_callback function.
    key_callback = TeleopMocap(data)

    gripped = False  # Initialize gripped state
    gripper_command = 0.0

    # Initialize recording lists
    recorded_times = []
    recorded_qpos = []
    recorded_qvel = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # Set up the camera
        viewer.cam.azimuth = -90  # Set azimuth angle
        viewer.cam.elevation = -15  # Set elevation angle
        viewer.cam.distance = 2.0  # Set distance from the scene
        # Set the point the camera looks at
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=100.0, warn=False)

        # Define the poses for the arm
        orientation = SO3.from_rpy_radians(0, np.pi, np.pi/2)
        soft_cube_body_id = model.body("soft_cube").id
        soft_cube_position = data.xpos[soft_cube_body_id]
        # Get the dimensions of the soft_cube
        soft_cube_geom_id = model.body("soft_cube").id
        soft_cube_dimensions = model.geom_size[soft_cube_geom_id]
        print(f"Soft Cube dimensions: {soft_cube_dimensions}")
        print(f"Soft Cube position: {soft_cube_position}")

        grab_position = np.array(
            [soft_cube_position[0], soft_cube_position[1], soft_cube_position[2] - soft_cube_dimensions[2] + 0.2])
        print(f"Grab position: {grab_position}")

        target_position = grab_position + np.array([0.0, 0.0, 0.2])
        print(f"Target position: {grab_position}")

        # Use the position of "soft_cube" as the target position
        target_pose = mink.SE3.from_rotation_and_translation(
            orientation,
            grab_position
        )
        new_target_pose = mink.SE3.from_rotation_and_translation(
            orientation,
            target_position
        )
        final_target_pose = mink.SE3.from_rotation_and_translation(
            orientation,
            target_position + np.array([-0.1, 0.4, 0.0])
        )

        # Move the arm to the target pose
        if move_arm_to_position(
            viewer, model, data, configuration,
            tasks, limits, solver,
            rate, end_effector_task, target_pose,
            pos_threshold, ori_threshold, max_iters,
            recorded_times, recorded_qpos, recorded_qvel   # pass recording lists
        ):
            gripper_distance_threshold = 1e-3
            squeeze_object(viewer, model, data, configuration,
                           rate, gripper_distance_threshold,
                           recorded_times, recorded_qpos, recorded_qvel)  # pass recording lists
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        move_arm_to_position(
            viewer, model, data, configuration,
            tasks, limits, solver,
            rate, end_effector_task, new_target_pose,
            pos_threshold, ori_threshold, max_iters,
            recorded_times, recorded_qpos, recorded_qvel   # pass recording lists
        )
        
        print("Finished moving.")

        while viewer.is_running():
            viewer.sync()
            rate.sleep()

        # Convert recording lists to numpy arrays for plotting if desired
        import numpy as np
        recorded_times = np.array(recorded_times)
        recorded_qpos = np.array(recorded_qpos)
        recorded_qvel = np.array(recorded_qvel)

        # Fetch joint names from the max_velocities dictionary if available
        n_joints = 6
        if n_joints == len(max_velocities):
            joint_names = list(max_velocities.keys())
        else:
            joint_names = [f'Joint {j}' for j in range(n_joints)]

        # Plot joint positions over time
        plt.figure()
        for j in range(n_joints):
            plt.plot(recorded_times, recorded_qpos[:, j], label=joint_names[j])
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position")
        plt.title("Joint Positions Over Time")
        plt.legend()
        plt.savefig("graphs/joint_positions.png")
        plt.close()

        # Plot joint velocities over time
        plt.figure()
        for j in range(n_joints):
            plt.plot(recorded_times, recorded_qvel[:, j], label=joint_names[j])
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Velocity")
        plt.title("Joint Velocities Over Time")
        plt.legend()
        plt.savefig("graphs/joint_velocities.png")
        plt.close()
