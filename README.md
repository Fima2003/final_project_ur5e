# UR5e: Simulation + Real Robot (minimal)

## Overview
- Runs a MuJoCo simulation and the real UR5e (Robotiq gripper) from `main.py`.
- Logs and plots over time: TCP pose, TCP speed, joint positions, and joint speeds.
- Plots are continuous across actions with vertical dotted lines labeled by the preceding action index.

## Prerequisites
- Python 3.10+ (tested on macOS and Windows)
- MuJoCo and Mink (installed via requirements below)
- Optional: UR5e reachable on your network. Default IP is `192.168.1.100` (can be changed in `main.py`).

## Install dependencies
- Use: `python -m pip install -r requirements.txt`

Notes:
- This installs `mujoco==3.x` and `mink`. On macOS, prefer `mjpython` so MuJoCo runtime paths are set correctly and viewer can be used.

## Scene selection
- `scene.xml` is the base scene. Inside it, choose which robot model to include:
	- `robot.xml` (default)
	- `robot vertical.xml` (gripper mounted vertically)
- Edit the `<include file="..." />` line in `scene.xml` to switch. The only difference is how the gripper is attached to the end-effector.

## Run
- macOS: `mjpython main.py`
- Windows: `python main.py`

What happens:
- The script initializes MuJoCo, plans motions, and (by default) connects to the real robot at `192.168.1.100`.
- Figures are saved to `graphs/` (no GUI windows):
	- `tcp_pose.png`
	- `tcp_speed.png`
	- `joint_positions.png`
	- `joint_speeds.png`

## Notes
- In `URBasic/urScript.py`, functions `waitRobotIdleOrStopFlag` and `movej` were slightly altered to enable graph plotting.
- On macOS we use a non-interactive Matplotlib backend; plots are written to disk.

## Safety
- Running `main.py` will move the real robot. Ensure the workspace is clear, the robot is in a safe state, and an E‑Stop is within reach.
