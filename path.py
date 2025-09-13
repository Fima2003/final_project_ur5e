from typing import List, Tuple
import numpy as np
from action import Action, ActionType

def get_actions(shark_body_position, open_box_position, twin, poly_order, time_for_trajectory) -> Tuple[List[Action], np.ndarray]:
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
    
    # # Pose 7. Move robot with shark to the open box
    # above_box_pose = np.array([
    #     *(open_box_position + [0, -0.3, 0.2]), 
    #     0.21590105, 3.15800794, 0.09401606
    # ])
    # Pose 8. Drop the fish into the box
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
        # Action(
        #     ActionType.MOVE, 
        #     twin,
        #     PathPlanner(
        #         [dict(initial_pose=initial_pose_mid.copy(), target_pose=above_box_pose.copy(), time_for_trajectory=10, poly_order=5)],
        #     )
        # ),
        # Action(
        #     ActionType.RELEASE,
        #     twin
        # )
    ]
    return actions, initial_pose_pre