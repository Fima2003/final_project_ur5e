import mujoco
import numpy as np
import mink


def setup(path):
    model = mujoco.MjModel.from_xml_path(path.as_posix())
    data = mujoco.MjData(model)

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
        (wrist_3_geoms, ["floor", "table", 'wall4', 'wall2', 'wall3']),
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
    return model, data, configuration, tasks, limits, end_effector_task


def move_arm_to_position(
    viewer, model, data, configuration: mink.Configuration,
    tasks, limits, solver,
    rate, end_effector_task: mink.FrameTask, target_pose,
    pos_threshold, ori_threshold, max_iters,
    recorded_times, recorded_qpos, recorded_qvel,   # <--- added parameters
):
    """Moves the arm to the specified target pose using IK."""
    while viewer.is_running():
        # Set the end-effector target to the desired pose
        end_effector_task.set_target(target_pose)

        # Perform IK iterations
        for i in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, rate.dt,
                                solver, damping=1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            err = end_effector_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                break

        # Apply control command for the arm (gripper remains unchanged here)
        arm_ctrl = configuration.q[:6]
        data.ctrl[:6] = arm_ctrl

        mujoco.mj_step(model, data)

        # Synchronize configuration with the latest simulation state
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Recompute error after update
        err = end_effector_task.compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        # Record data at this time step
        recorded_times.append(data.time)
        recorded_qpos.append(data.qpos.copy())
        recorded_qvel.append(data.qvel.copy())

        viewer.sync()
        rate.sleep()

        if pos_achieved and ori_achieved:
            print("Target pose achieved.")
            return True
    return False


def squeeze_object(viewer, model, data, configuration, rate, d_gripper_box_threshold,
                   recorded_times, recorded_qpos, recorded_qvel):  # <--- added parameters
    """Squeezes the gripper until the force threshold is reached, then lifts the object up."""
    i = 0
    while viewer.is_running():

        pos_gripper = data.sensordata[:3]
        pos_box = data.sensordata[3:]
        d_gripper_box = np.linalg.norm(pos_gripper - pos_box)
        if d_gripper_box <= d_gripper_box_threshold:
            i += 1
        else:
            i = 0
        if i > 200:
            print("Distance threshold reached (", d_gripper_box_threshold, "), stopping to squeeze")
            return True

        # Command gripper closing while squeezing
        arm_ctrl = configuration.q[:6]
        gripper_command = 255.0
        data.ctrl = np.concatenate((arm_ctrl, np.array([gripper_command])))
        mujoco.mj_step(model, data)

        # Record data at this time step
        recorded_times.append(data.time)
        recorded_qpos.append(data.qpos.copy())
        recorded_qvel.append(data.qvel.copy())

        viewer.sync()
        rate.sleep()
