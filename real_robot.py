import time
import URBasic
import numpy as np
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

def robot_grip(robotiquegrip: Robotiq_Two_Finger_Gripper):
    robotiquegrip.gripper_action(255)
    time.sleep(1.5)
    robotiquegrip.close_gripper()
    time.sleep(1.5)
    return True

def robot_release(robotiquegrip: Robotiq_Two_Finger_Gripper):
    robotiquegrip.gripper_action(0)
    time.sleep(1.5)
    robotiquegrip.open_gripper()
    time.sleep(1.5)
    return True

def simulate_robot_joints(
    robot: URBasic.UrScriptExt,
    joint_positions: np.ndarray,
    a=0.2,
    v=0.2,
    wait=True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    plots = robot.movej(joint_positions, a=a, v=v, wait=wait)
    return plots