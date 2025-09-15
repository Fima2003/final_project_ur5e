import os
import numpy as np
import quaternion as qt
from action import Action, ActionType
from path import get_actions
from pathlib import Path
import mujoco
import mujoco.viewer
from mink import SO3
from ur5e import UR5e
from loop_rate_limiters import RateLimiter
from plot_data import PlotData
import os
import numpy.linalg as LA
from utils import combine_and_plot
    
check_error = lambda current, goal, pose_error: (LA.norm(abs(current) - abs(goal)) > pose_error)
  
def simulate_mujoco(
    data: mujoco.MjData, # type: ignore
    model: mujoco.MjModel, # type: ignore
    twin: UR5e,
    position: np.ndarray,
    orientation: np.ndarray,
    gripper: bool,
    viewer: mujoco.viewer.Handle,
    tool_body_id: int,
    rate: RateLimiter,
    pose_error: float = 0.026,
    vel_error: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Collect samples into lists first (faster and avoids np.append misuse)
    tcp_pose_list: list[np.ndarray] = []      # each elem: [x,y,z, rx,ry,rz]
    tcp_speed_list: list[np.ndarray] = []     # each elem: 6D spatial velocity
    joint_position_list: list[np.ndarray] = []# each elem: 6 joint positions
    joint_speed_list: list[np.ndarray] = []   # each elem: 6 joint velocities
    times_list: list[float] = []              # each elem: time in seconds
    
    if not isinstance(orientation, qt.quaternion):
        orientation = qt.from_rotation_vector(orientation)
    print(f"\nMoving to the new pose: {position}, {orientation}")
    twin.set_destination(
        position,
        SO3.from_matrix(qt.as_rotation_matrix(orientation))
    )
    full_ctrl = twin.go(frequency=1/rate.dt)
    print(f"Control to be applied: {full_ctrl}")
    data.ctrl[:] = full_ctrl.copy()
    #! Don't forget to uncomment this
    # if gripper:
    #     data.ctrl[-2] = 1
    #     data.ctrl[-1] = 0.22
    # else:
    #     data.ctrl[-2] = 0
    #     data.ctrl[-1] = -1
    i=0
    update_i = True
    print("Starting position:", np.array(data.xpos[tool_body_id]))
    print("Initial position error:", LA.norm(np.array(data.xpos[tool_body_id]) - np.array(position)))
    print("Starting orientation:", np.array(data.xquat[tool_body_id]))
    print("Initial orientation error:", LA.norm(np.array(data.xquat[tool_body_id]) - qt.as_float_array(orientation)))
    while (
        check_error(np.array(data.xpos[tool_body_id]), np.array(position), pose_error)
        or check_error(np.array(data.xquat[tool_body_id]), qt.as_float_array(orientation), pose_error)
        or LA.norm(np.array(data.qvel)) > vel_error
    ):
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()
        # TCP pose: 3D position + 3D rotation vector
        tool_pos = np.array(data.xpos[tool_body_id]).copy()
        tool_rotvec = qt.as_rotation_vector(
            qt.from_float_array(np.array(data.xquat[tool_body_id]).copy())
        )
        tcp_pose_list.append(np.concatenate([tool_pos, tool_rotvec]))
        velocity_result = np.zeros(6)
        mujoco.mj_objectVelocity( # type: ignore
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY, # type: ignore
            tool_body_id, 
            velocity_result,
            0
        )
        joint_position_list.append(np.array(data.qpos[:6]).copy())
        tcp_speed_list.append(np.array(velocity_result).copy())
        joint_speed_list.append(np.array(data.qvel[:6]).copy())
        times_list.append(i * rate.dt)
        
        if update_i:
            i+=1
        if i > 1200:
            print("Taking too long to reach the target pose, viewing current error:")
            
            print(f"Current position: {np.array(data.xpos[tool_body_id])}, target position: {position}")
            print(f"Position error: {LA.norm(np.array(data.xpos[tool_body_id]) - np.array(position))}: {check_error(np.array(data.xpos[tool_body_id]), np.array(position), pose_error)}")
            
            print(f"Current orientation: {np.array(data.xquat[tool_body_id])}, target orientation: {qt.as_float_array(orientation)}")
            print(f"Orientation error: {LA.norm(abs(np.array(data.xquat[tool_body_id])) - abs(qt.as_float_array(orientation)))}, {check_error(np.array(data.xquat[tool_body_id]), qt.as_float_array(orientation), pose_error)}")
            print(f"Velocity error: {LA.norm(np.array(data.qvel))}, {LA.norm(np.array(data.qvel)) > vel_error}")
            update_i = False
            i-=100
    robot_init_joints = []
    for i in range(len(full_ctrl[:6])):
        ctrl = full_ctrl[i]
        if i == 1 or i == 3:
            ctrl -= np.pi/2
        robot_init_joints.append(ctrl)
    # Convert collected lists to numpy arrays with consistent shapes
    joint_position_arr = np.array(joint_position_list) if joint_position_list else np.empty((0,6))
    joint_speed_arr = np.array(joint_speed_list) if joint_speed_list else np.empty((0,6))
    tcp_pose_arr = np.array(tcp_pose_list) if tcp_pose_list else np.empty((0,6))
    tcp_speed_arr = np.array(tcp_speed_list) if tcp_speed_list else np.empty((0,6))
    times_arr = np.array(times_list) if times_list else np.empty((0,))

    return np.array(robot_init_joints), joint_position_arr, joint_speed_arr, tcp_pose_arr, tcp_speed_arr, times_arr

def mujoco_grip(
    twin: UR5e,
    data: mujoco.MjData, # type: ignore
    model: mujoco.MjModel, # type: ignore
    viewer: mujoco.viewer.Handle,
    rate: RateLimiter
):
    twin.grip()
    for _ in range(500):
        actual_ctrl = data.ctrl.copy()
        actual_ctrl[-1] = -1
        data.ctrl = actual_ctrl
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()
        
    for _ in range(500):
        actual_ctrl = data.ctrl.copy()
        actual_ctrl[-2] = 1
        actual_ctrl[-1] = -1
        data.ctrl = actual_ctrl
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()
    
    for _ in range(500):
        actual_ctrl = data.ctrl.copy()
        actual_ctrl[-1] = 0.22
        data.ctrl = actual_ctrl
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()

def mujoco_release(
    twin: UR5e,
    data: mujoco.MjData, # type: ignore
    model: mujoco.MjModel, # type: ignore
    viewer: mujoco.viewer.Handle,
    rate: RateLimiter
):
    twin.release()
    for _ in range(500):
        actual_ctrl = data.ctrl.copy()
        actual_ctrl[-2] = 0
        actual_ctrl[-1] = 0.22
        
        data.ctrl = actual_ctrl
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()
    
    for _ in range(500):
        actual_ctrl = data.ctrl.copy()
        actual_ctrl[-1] = -1
        data.ctrl = actual_ctrl
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()

def act(
    action: Action,
    data: mujoco.MjData, # type: ignore
    model: mujoco.MjModel, # type: ignore
    viewer: mujoco.viewer.Handle,
    tool_body_id: int,
    gripper: bool,
    rate: RateLimiter
) -> tuple[bool, PlotData | None]:
    type, target, twin, idx = action.type, action.target, action.twin, action.idx
    
    if type == ActionType.MOVE:
        if not isinstance(target, np.ndarray):
            raise ValueError("Planner must be provided for MOVE action")
        pos, ori = target[:3], target[3:]
        
        _, joint_position, joint_velocity, tcp_pose, tcp_velocity, times = simulate_mujoco(
            data,
            model,
            twin,
            pos,
            ori,
            gripper,
            viewer,
            tool_body_id,
            rate
        )
        return gripper, PlotData(
            joint_position=joint_position,
            joint_speed=joint_velocity,
            tcp_pose=tcp_pose,
            tcp_speed=tcp_velocity,
            times=times,
            act_idx=idx
        )
    elif type == ActionType.GRIP:
        print("\nGripping the fish")
        mujoco_grip(
            twin,
            data,
            model,
            viewer,
            rate
        )
        return True, None
    elif type == ActionType.RELEASE:
        print("\nReleasing the fish")
        mujoco_release(
            twin,
            data,
            model,
            viewer,
            rate
        )
        return False, None
    print(f"Action {type} completed.")
    return False, None
        
if __name__ == "__main__":
    _HERE = Path(os.getcwd())
    _XML = _HERE / "scene.xml"
    FREQUENCY = 200.0
    twin = UR5e()
    rate = RateLimiter(FREQUENCY, warn=False)
    gripper = False

    model = mujoco.MjModel.from_xml_path(_XML.as_posix()) # type: ignore
    data = mujoco.MjData(model) # type: ignore
    mujoco.mj_step(model, data) # type: ignore

    shark_body_id = model.body("shark_body").id
    shark_body_position = data.xpos[shark_body_id]

    # Fetch open box idtop, tool id and base id
    box_body_id = model.body("open_box_body").id
    open_box_position = data.xpos[box_body_id]
    tool_body_id = model.body("attachment").id

    print(f"Open box position={open_box_position}")

    viewer = mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
    )
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15
    viewer.cam.distance = 2.5
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

    for i in range(100):
        mujoco.mj_step(model, data) # type: ignore
        viewer.sync()
        rate.sleep()
    print(f"Shark's initial position={shark_body_position}")

    actions, initial_pose_pre = get_actions(shark_body_position, open_box_position, twin, poly_order=5, time_for_trajectory=10.0)
    
    _, joint_position, joint_velocity, tcp_pose, tcp_velocity, times = simulate_mujoco(
        data,
        model,
        twin,
        initial_pose_pre[:3],
        initial_pose_pre[3:],
        gripper,
        viewer,
        tool_body_id,
        rate
    )
    plots = [PlotData(
                joint_position=joint_position,
                joint_speed=joint_velocity,
                tcp_pose=tcp_pose,
                tcp_speed=tcp_velocity,
                times=times,
                act_idx=0
            )]
    for i in range(len(actions)):
        print(i)
        if gripper is None:
            raise ValueError("Gripper must be closed to start the actions")
        gripper, plot_data = act(actions[i], data, model, viewer, tool_body_id, gripper, rate)
        if plot_data is not None:
            plots.append(plot_data)
    
    print("All actions completed.")
    viewer.close()

    combine_and_plot(plots, Path(os.getcwd()) / 'graphs' / 'mujoco')
    
    