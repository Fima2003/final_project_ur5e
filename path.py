from typing import List, Tuple
import numpy as np
from action import Action, ActionType

def get_actions(shark_body_position, open_box_position, twin) -> Tuple[List[Action], np.ndarray]:
    initial_pose_pre = np.array([
        0.0, -0.6, 0.25,
        0.21590105, 3.15800794, 0.09401606
    ])

    # Pose 2. Slightly above the fish
    above_fish_pose = np.array([
        *(shark_body_position + np.array([0, -0.3, 0.25])),
        0.21590105, 3.15800794, 0.09401606
    ])
    # # Pose 3. Below the fish
    below_fish_pose = np.array([
        *(shark_body_position + np.array([0, -0.3, 0.2])), 
        0.21590105, 3.15800794, 0.09401606
    ])
    # # Pose 4. Grip the fish
    # Pose 5. Lift up the fish
    lift_fish_pose = np.array([
        *(shark_body_position + np.array([0, -0.3, 0.2])),
        0.21590105, 3.15800794, 0.09401606
    ])
    # # Pose 6. Move to initial position
    initial_pose_mid = np.array([
        0.0, -0.6, 0.2,
        0.21590105, 3.15800794, 0.09401606
    ])
    actions = [
        Action(
            ActionType.MOVE,
            twin,
            target=above_fish_pose.copy(),
            idx=1
        ),
        Action(
            ActionType.MOVE,
            twin,
            target=below_fish_pose.copy(),
            idx=2
        ),
        # Action(ActionType.GRIP, twin=twin),
        Action(
            ActionType.MOVE, 
            twin,
            target=lift_fish_pose.copy(),
            idx=3
        ),
        Action(
            ActionType.MOVE, 
            twin,
            target=initial_pose_mid.copy(),
            idx=4
        ),
        Action(
            ActionType.RELEASE,
            twin
        )
    ]
    return actions, initial_pose_pre