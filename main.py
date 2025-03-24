from ik import qpos_from_site_pose
from setup import setup
import numpy as np
import mujoco
from utils import get_soft_cube_position


def move_to_position(model, data, viewer, position, rotation, physics, steps, extra_steps=1000):
    """
    Moves the robot's end-effector to the given position using IK.
    Since the robot is fixed, its base frame is the same as the world frame.
    We apply a rotation offset to align the gripper with the object.
    After interpolation, extra simulation steps are run to let the motion finish.
    """
    current_arm = data.qpos[0:6].copy()
    best_sol = qpos_from_site_pose(
        physics=physics,
        site_name='attachment_site',
        target_pos=position,
        target_quat=[0, -np.pi/4, 0, 0],
        joint_names=None,
        tol=1e-6,
        rot_weight=1.0,
        max_steps=100
    ).qpos[0:6]

    for i in range(steps + 1):
        alpha = i / steps  # alpha goes from 0 (current) to 1 (target)
        interp_conf = (1 - alpha) * current_arm + alpha * best_sol
        data.qpos[0:6] = interp_conf

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.render()

    # Extra simulation steps to ensure the robot fully settles in the target position:
    for _ in range(extra_steps):
        viewer.render()


def squeeze(model, data, viewer, percentage=None, extra_steps=1000):
    """
    Controls the gripper. If `percentage` is between 0 and 1, sets gripper position directly.
    If `percentage` is None, closes until it 'feels' resistance (based on low joint velocity).
    """
    # Freeze current robot arm pose
    arm_qpos = data.qpos[:6].copy()

    if percentage is not None:
        percentage = np.clip(percentage, 0.0, 1.0)
        data.ctrl[6] = int(255 * percentage)
        print(f"Squeezing gripper to {percentage * 100:.1f}%")

        data.qpos[0:6] = arm_qpos
        mujoco.mj_forward(model, data)

        for _ in range(extra_steps):
            data.qpos[0:6] = arm_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.render()
    else:
        print("Auto-squeezing until contact detected...")
        data.qpos[0:6] = arm_qpos
        mujoco.mj_forward(model, data)

        left_pad_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_pad1")
        right_pad_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_pad1")
        cube_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "soft_cube")

        for ctrl_val in np.linspace(0, 255, num=60):  # Finer steps
            data.ctrl[6] = int(ctrl_val)
            data.qpos[0:6] = arm_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.render()

            left_pad_pos = data.site_xpos[left_pad_id]
            right_pad_pos = data.site_xpos[right_pad_id]
            cube_pos = data.xpos[cube_body_id]

            left_dist = np.linalg.norm(left_pad_pos - cube_pos)
            right_dist = np.linalg.norm(right_pad_pos - cube_pos)
            print(left_dist, right_dist)

            if min(left_dist, right_dist) < 0.003:
                print(
                    f"Contact detected at ctrl={int(ctrl_val)} with distance {min(left_dist, right_dist):.4f}")
                break

        for _ in range(extra_steps):
            data.qpos[0:6] = arm_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.render()


def main():
    model, data, viewer, physics = setup()

    cube_pos = get_soft_cube_position(model, data)
    print(f"Cube position: {cube_pos}")

    # Create a target position 0.13 m above the cube.
    target_pos = cube_pos.copy()
    target_pos[2] += 0.15
    target_quat = [2, -np.pi/4, np.pi/3, 0]
    move_to_position(model, data, viewer, target_pos,
                     target_quat, physics, 100, extra_steps=0)
    squeeze(model, data, viewer)

    # Keep the viewer open for a while (or until closed manually).
    viewer.close()


if __name__ == "__main__":
    main()
