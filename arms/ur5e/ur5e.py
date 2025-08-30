import mink
import numpy as np

from Arm import Arm


class UR5e(Arm):
    def __init__(self):
        super().__init__(actuators=6)

    def initialize(self):
        self.end_effector_task: mink.FrameTask = mink.FrameTask(
            'attachment_site', 'site', position_cost=1.0, orientation_cost=0.5, lm_damping=1e-6)
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [
            self.end_effector_task,
            self.posture_task
        ]

        self.max_velocities = {
            "shoulder_pan_joint": np.pi/4,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        wrist_3_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("wrist_3_link").id
        )
        wrist_2_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("wrist_2_link").id
        )
        wrist_1_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("wrist_1_link").id
        )
        forearm_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("forearm_link").id
        )
        upper_arm_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("upper_arm_link").id
        )
        shoulder_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("shoulder_link").id
        )
        right_subassy_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("right_subassy").id
        )
        left_subassy_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("left_subassy").id
        )
        initial_collision_pairs = [
            (wrist_3_geoms, collisions := [
                "floor",
                # "table",  "wall1", "wall2", "wall3"
            ]),
            (wrist_2_geoms, collisions),
            (wrist_1_geoms, collisions),
            (forearm_geoms, collisions),
            (upper_arm_geoms, collisions),
            (shoulder_geoms, collisions),
            (right_subassy_geoms, collisions),
            (left_subassy_geoms, collisions),
        ]
        extra_collision_pairs = [
        ]
        self.limits = [
            mink.ConfigurationLimit(
                model=self.config.model),
            mink.CollisionAvoidanceLimit(
                model=self.config.model,
                geom_pairs=initial_collision_pairs,
                minimum_distance_from_collisions=0.5
            ),
            mink.VelocityLimit(self.model, self.max_velocities)
        ]

        self.initial_position = self.data.qpos[:self.model.nu].copy()
