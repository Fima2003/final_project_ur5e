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
import matplotlib
# Use a non-interactive backend to avoid macOS NSWindow errors when not on main thread
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    
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

def combine_and_plot(plots_list: list[PlotData], graphs_dir: Path):
    if not plots_list:
        print("No plot data to visualize.")
        return

    # Ensure output directory exists
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Sort by action index to maintain correct order
    plots_sorted = sorted(plots_list, key=lambda p: p.act_idx)

    all_times: list[float] = []
    all_joint_pos: list[np.ndarray] = []
    all_joint_speed: list[np.ndarray] = []
    all_tcp_pose: list[np.ndarray] = []
    all_tcp_speed: list[np.ndarray] = []
    # Keep boundaries between actions: (time, act_idx_of_previous_segment)
    boundaries: list[tuple[float, int]] = []

    t_offset = 0.0
    for p in plots_sorted:
        if p.times is None or p.times.size == 0:
            continue
        times = p.times.ravel()
        # Compute dt from this segment if possible
        dt = (times[1] - times[0]) if times.size > 1 else 0.0
        seg_times = (times + t_offset).tolist()

        # Accumulate
        all_times.extend(seg_times)
        if p.joint_position is not None and p.joint_position.size > 0:
            all_joint_pos.extend([row for row in p.joint_position])
        if p.joint_speed is not None and p.joint_speed.size > 0:
            all_joint_speed.extend([row for row in p.joint_speed])
        if p.tcp_pose is not None and p.tcp_pose.size > 0:
            all_tcp_pose.extend([row for row in p.tcp_pose])
        if p.tcp_speed is not None and p.tcp_speed.size > 0:
            all_tcp_speed.extend([row for row in p.tcp_speed])

        # Mark a boundary at the end of this segment (label with this segment's act_idx)
        seg_end = seg_times[-1] if len(seg_times) > 0 else t_offset
        boundaries.append((seg_end, getattr(p, 'act_idx', 0)))

        # Advance offset to make time continuous across actions
        t_offset = seg_end + (dt if dt > 0 else 0)

    # Convert to arrays
    T = np.array(all_times)
    JP = np.array(all_joint_pos) if all_joint_pos else np.empty((0,6))
    JV = np.array(all_joint_speed) if all_joint_speed else np.empty((0,6))
    TP = np.array(all_tcp_pose) if all_tcp_pose else np.empty((0,6))
    TV = np.array(all_tcp_speed) if all_tcp_speed else np.empty((0,6))

    # Guard: if no data, skip plotting
    if T.size == 0:
        print("No time samples collected; skipping plots.")
        return

    # Helper to draw vertical boundaries on one or more axes
    def draw_boundaries(axes):
        ax_list = axes if isinstance(axes, (list, np.ndarray, tuple)) else [axes]
        # skip the final boundary (no action after it)
        bnds = boundaries[:-1] if len(boundaries) > 1 else []
        for ax in ax_list:
            if not bnds:
                continue
            ymin, ymax = ax.get_ylim()
            for x, act_id in bnds:
                ax.axvline(x=x, color='k', linestyle=':', linewidth=1, alpha=0.5)
                # place label at top with slight padding
                ax.text(x, ymax, f'act {act_id}', rotation=90, va='top', ha='right',
                        fontsize=8, alpha=0.7,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0))

    # TCP Pose (position and orientation vector)
    if TP.size > 0:
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            axs[0].plot(T, TP[:, i], label=labels[i])
        axs[0].set_ylabel('Position [m]')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='best')

        labels_r = ['rx', 'ry', 'rz']
        for i in range(3):
            axs[1].plot(T, TP[:, 3 + i], label=labels_r[i])
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Rotation vector [rad]')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='best')
        # add boundaries on both subplots
        draw_boundaries([axs[0], axs[1]])
        fig.tight_layout()
        fig.savefig((graphs_dir / 'tcp_pose.png').as_posix(), dpi=150)
        plt.close(fig)

    # TCP Speed (spatial velocity)
    if TV.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        spd_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        for i in range(min(6, TV.shape[1])):
            ax.plot(T, TV[:, i], label=spd_labels[i] if i < len(spd_labels) else f'c{i}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('TCP speed [m/s, rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'tcp_speed.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Positions
    if JP.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(min(6, JP.shape[1])):
            ax.plot(T, JP[:, i], label=f'J{i+1}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint position [rad]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'joint_positions.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Speeds
    if JV.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(min(6, JV.shape[1])):
            ax.plot(T, JV[:, i], label=f'J{i+1}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint speed [rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'joint_speeds.png').as_posix(), dpi=150)
        plt.close(fig)

    print("Saved plots to:")
    print(f" - {graphs_dir / 'tcp_pose.png'}")
    print(f" - {graphs_dir / 'tcp_speed.png'}")
    print(f" - {graphs_dir / 'joint_positions.png'}")
    print(f" - {graphs_dir / 'joint_speeds.png'}")
        
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
    
    