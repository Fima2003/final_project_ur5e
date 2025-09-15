import os
import numpy as np
import time
from action import Action, ActionType
from mj import mujoco_grip, mujoco_release, simulate_mujoco
import URBasic
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from pathlib import Path
import mujoco
import mujoco.viewer
from ur5e import UR5e
from loop_rate_limiters import RateLimiter
from plot_data import PlotData
from path import get_actions
from real_robot import robot_grip, robot_release, simulate_robot_joints
import matplotlib

from utils import combine_plots_for_comparison, plot_comparison
# Use non-interactive backend (macOS main-thread windows issue)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# This allows stopping the program with Ctrl+C (KeyboardInterrupt). This line has to be at the top of the file(!!):
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

mujoco_plots = []
robot_plots = []

#######################################
########### SET UP MUJOCO #############
#######################################

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

open_box_body_id = model.body("open_box_body").id
open_box_position = data.xpos[open_box_body_id]
tool_body_id = model.body("attachment").id

print(f"Open box position={open_box_position}")

actions, initial_pose_pre = get_actions(shark_body_position, open_box_position, twin)

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

#######################################
########## SET UP REAL ROBOT ##########
#######################################

# Initialization of the RTDE and the robot model:
host = '192.168.1.100'                               # UR5 robot i.p. address
robotModel = URBasic.RobotModel()
robot = URBasic.UrScriptExt(host=host, robotModel=robotModel)
robot.reset_error()

# TCP and Payload initialization: (TCP = tool center point)
robot.set_payload_mass(m=1.000)                      # Fill in the payload mass.
robot.set_payload_cog(CoG=(-0.003, -0.006, 0.046))   # Fill in the center of gravity of the payload.
robot.set_tcp(pose=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0])   # Do not uncomment this unless the tutor told you to change TCP coordinates.

# Initialize Gripper
time.sleep(2) # Wait for 0.6 sec so the robot would be stable before taking measurements.
rob = urx.Robot(host)
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
robotiqgrip.gripper_action(0)
robotiqgrip.open_gripper()
time.sleep(2)

# Controller Setup
# Set PD parameters:
alpha = 1 #0.4
Kp_pos = alpha*5000*np.ones(3)              # Fill in the Kp for the xyz location error.
Kd_pos = np.sqrt(alpha)*100*np.ones(3)      # Fill in the Kd for the Vxyz linear velocity error.
Kp_ori = alpha*100*np.ones(3)               # Fill in the Kp for the Rx,Ry,Rz orientation error.
Kd_ori = np.sqrt(alpha)*10*np.ones(3)       # Fill in the Kd for the omega_xyz angular velocity error.

print(f'Kp_pos = {Kp_pos}\nKd_pos = {Kd_pos}\nKp_ori = {Kp_ori}\nKd_ori = {Kd_ori}')

# Move to the initial pose:
robot_joints, joint_position, joint_velocity, tcp_pose, tcp_velocity, times = simulate_mujoco(
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
mujoco_plots.append(PlotData(
    joint_position=joint_position,
    joint_speed=joint_velocity,
    tcp_pose=tcp_pose,
    tcp_speed=tcp_velocity,
    times=times,
    act_idx=0
))
print(f"Moving the robot to the initial joints: {robot_joints}")

plots = simulate_robot_joints(robot, joint_positions=robot_joints, a=0.3, v=0.1, wait=True)
# Assume always returns plots
assert plots is not None, "simulate_robot_joints returned None unexpectedly"
joint_position, joint_velocity, tcp_pose, tcp_velocity, times = plots
robot_plots.append(PlotData(
    joint_position=joint_position,
    joint_speed=joint_velocity,
    tcp_pose=tcp_pose,
    tcp_speed=tcp_velocity,
    times=times,
    act_idx=0
))
time.sleep(0.6)

def act(
    action: Action,
    data: mujoco.MjData, # type: ignore
    model: mujoco.MjModel, # type: ignore
    viewer: mujoco.viewer.Handle,
    tool_body_id: int,
    robot: URBasic.UrScriptExt,
    gripper: bool,
    rate: RateLimiter
):
    type, target, twin, idx = action.type, action.target, action.twin, action.idx
    print(f"\n--- Executing action {idx}: {type} ---")
    
    if type == ActionType.MOVE:
        if target == None:
            raise ValueError("Planner must be provided for MOVE action")
        pos, ori = target[:3], target[3:]
        
        robot_joints, joint_position, joint_velocity, tcp_pose, tcp_velocity, times = simulate_mujoco(
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
        mujoco_plots.append(PlotData(
            joint_position=joint_position,
            joint_speed=joint_velocity,
            tcp_pose=tcp_pose,
            tcp_speed=tcp_velocity,
            times=times,
            act_idx=action.idx
        ))
        plots = simulate_robot_joints(
            robot=robot,
            joint_positions=robot_joints,
            a=0.2,
            v=0.2,
            wait=True
        )
        # Assume always returns plots
        assert plots is not None, "simulate_robot_joints returned None unexpectedly"
        joint_position, joint_velocity, tcp_pose, tcp_velocity, times = plots
        robot_plots.append(PlotData(
            joint_position=joint_position,
            joint_speed=joint_velocity,
            tcp_pose=tcp_pose,
            tcp_speed=tcp_velocity,
            times=times,
            act_idx=action.idx
        ))
        return gripper
    elif type == ActionType.GRIP:
        print("\nGripping the fish")
        mujoco_grip(
            twin,
            data,
            model,
            viewer,
            rate
        )
        robot_grip(robotiqgrip)
        return True
    elif type == ActionType.RELEASE:
        print("\nReleasing the fish")
        mujoco_release(
            twin,
            data,
            model,
            viewer,
            rate
        )
        robot_release(robotiqgrip)
        return False
    print(f"Action {type} completed.")
    return True

for i in range(len(actions)):
    gripper = act(actions[i], data, model, viewer, tool_body_id, robot, gripper, rate)

viewer.close()

combined = combine_plots_for_comparison(robot_plots, mujoco_plots)
plot_comparison(combined, Path(os.getcwd()) / 'graphs')