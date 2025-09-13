from enum import Enum

import numpy as np

from ur5e import UR5e


class ActionType(Enum):
    MOVE = 1
    GRIP = 2
    RELEASE = 3


class Action:
    type: ActionType
    target: np.ndarray | None
    twin: UR5e
    idx: int = 0
    
    def __init__(self, type: ActionType, twin: UR5e, target: np.ndarray | None = None, idx: int = 0):
        self.type = type
        self.target = target
        self.twin = twin
        self.idx = idx
