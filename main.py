import math
from setup import setup
import numpy as np
import matplotlib.pyplot as plt

import mujoco

from utils import get_soft_cube_position
import mink
from mink.contrib import TeleopMocap
from loop_rate_limiters import RateLimiter


def dh_transform(a, alpha, d, theta):
    """
    Computes the standard Denavit–Hartenberg transformation matrix.
    a: link length
    alpha: link twist (radians)
    d: link offset
    theta: joint angle (radians)
    """
    st = math.sin(theta)
    ct = math.cos(theta)
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0,    sa,      ca,     d],
        [0,     0,       0,     1]
    ], dtype=float)


def ur5e_ik(T_des):
    """
    Computes all possible closed-form inverse kinematics solutions for the UR5e.

    Input:
      T_des: a 4x4 homogeneous transform representing the desired end-effector pose 
             in the robot's base frame.

    Returns:
      A list of candidate solutions. Each solution is a 6-element numpy array representing 
      joint angles [t1, t2, t3, t4, t5, t6] in radians.
      If the target is unreachable, no solution is returned for that branch.
    """
    # UR5e DH parameters (in meters)
    d1 = 0.1625
    a2 = -0.425    # note: negative as per your model convention
    a3 = -0.3922
    d6 = 0.0996

    # Extract desired end-effector position and rotation from T_des
    px, py, pz = T_des[0, 3], T_des[1, 3], T_des[2, 3]
    R_des = T_des[:3, :3]
    # The end-effector's local z-axis (third column of the rotation matrix)
    ez = R_des[:, 2]

    # Compute wrist center position by subtracting d6 along ez
    wx = px - d6 * ez[0]
    wy = py - d6 * ez[1]
    wz = pz - d6 * ez[2]

    solutions = []

    # Solve for theta1 (base joint) candidates:
    t1_a = math.atan2(wy, wx)
    t1_b = t1_a + math.pi
    t1_candidates = [t1_a, t1_b]

    for t1 in t1_candidates:
        # Compute r and s: distance components from base to wrist center
        r = math.sqrt(wx**2 + wy**2)
        s = wz - d1

        # Compute D using the law of cosines for the triangle formed by links a2 and a3:
        D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)

        # If D is outside the range [-1, 1], the target is unreachable for this candidate.
        if D < -1.0 or D > 1.0:
            continue

        # Two possible solutions for theta3 (elbow configuration)
        phi3 = math.acos(D)
        for t3 in [phi3, -phi3]:
            # Compute theta2 based on geometry:
            k1 = a2 + a3 * math.cos(t3)
            k2 = a3 * math.sin(t3)
            t2 = math.atan2(s, r) - math.atan2(k2, k1)

            # Forward kinematics from base to joint 3:
            T01 = dh_transform(0, math.pi/2, d1, t1)
            T12 = dh_transform(a2, 0, 0, t2)
            T23 = dh_transform(a3, 0, 0, t3)
            T_base_j3 = T01 @ T12 @ T23

            # Compute transformation from joint 3 to end-effector:
            T_j3_ee = np.linalg.inv(T_base_j3) @ T_des
            R_wrist = T_j3_ee[:3, :3]

            # Solve for wrist joint (t4, t5, t6) angles:
            cos_t5 = R_wrist[2, 2]
            # For safety, if cos_t5 is not within [-1, 1], skip this solution
            if cos_t5 < -1.0 or cos_t5 > 1.0:
                continue

            t5 = math.acos(cos_t5)
            if abs(math.sin(t5)) < 1e-6:
                t4 = 0.0
                t6 = math.atan2(-R_wrist[1, 0], R_wrist[0, 0])
            else:
                t4 = math.atan2(R_wrist[1, 2], R_wrist[0, 2])
                t6 = math.atan2(R_wrist[2, 1], -R_wrist[2, 0])

            # Construct the candidate solution vector and normalize angles to [-pi, pi]
            sol = np.array([t1, t2, t3, t4, t5, t6], dtype=float)
            sol = (sol + math.pi) % (2 * math.pi) - math.pi
            solutions.append(sol)

    return solutions


def move_to_position(model, data, viewer, position, steps):
    """
    Moves the robot's end-effector to the given position using IK.
    Since the robot is fixed, its base frame is the same as the world frame.
    We apply a rotation offset to align the gripper with the object.
    After interpolation, extra simulation steps are run to let the motion finish.
    """
    # Desired end-effector transform in world coordinates:
    T_des = np.eye(4)
    T_des[:3, 3] = position

    # Apply a rotation offset: adjust by -pi/2 about the z-axis
    R_offset = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    T_des[:3, :3] = R_offset
    print(f"Desired T (world and robot base coincide):\n{T_des}")

    # Compute IK directly using T_des:
    solutions = ur5e_ik(T_des)
    print(f"I found {len(solutions)} solutions to the given IK problem")
    if not solutions:
        print("No IK solution found for the target position:", position)
        return

    # Since the robot's arm joints are stored in data.qpos[0:6] (fixed base),
    # select the best solution based on the current arm configuration:
    current_arm = data.qpos[0:6].copy()
    best_sol = min(
        solutions, key=lambda sol: np.linalg.norm(sol - current_arm))

    # Interpolate from current configuration to best_sol:
    for i in range(steps + 1):
        alpha = i / steps  # alpha goes from 0 (current) to 1 (target)
        interp_conf = (1 - alpha) * current_arm + alpha * best_sol
        data.qpos[0:6] = interp_conf

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.render()

    # Extra simulation steps to ensure the robot fully settles in the target position:
    extra_steps = 500
    for _ in range(extra_steps):
        mujoco.mj_step(model, data)
        viewer.render()


def main():
    model, data, mink_config = setup()
    configuration = mink_config["configuration"]
    limits = mink_config["limits"]
    tasks = mink_config["tasks"]
    end_effector_task = mink_config["end_effector_task"]

    mid = model.body("soft_cube").mocapid[0]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Initialize key_callback function.
    key_callback = TeleopMocap(data)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(
            model, data, "soft_cube", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "soft_cube")
            end_effector_task.set_target(T_wt)

            # Continuously check for autonomous key movement.
            key_callback.auto_key_move()

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()

    # Create a target position 0.1 m above the cube.
    # target_pos = cube_pos.copy()
    # target_pos[2] += 0.1

    # # Move the robot's gripper to the target position.
    # move_to_position(model, data, viewer, target_pos, 500)
    # # Close the viewer when done.
    # viewer.close()


if __name__ == "__main__":
    main()
