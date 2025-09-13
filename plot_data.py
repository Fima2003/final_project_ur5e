import numpy as np

class PlotData:
    joint_position: np.ndarray
    joint_speed: np.ndarray
    tcp_pose: np.ndarray
    tcp_speed: np.ndarray
    times: np.ndarray
    act_idx: int = 0

    def __init__(
        self,
        joint_position: np.ndarray,
        joint_speed: np.ndarray,
        tcp_pose: np.ndarray,
        tcp_speed: np.ndarray,
        times: np.ndarray,
        act_idx: int = 0
    ):
        self.joint_position = joint_position
        self.joint_speed = joint_speed
        self.tcp_pose = tcp_pose
        self.tcp_speed = tcp_speed
        self.times = times
        self.act_idx = act_idx