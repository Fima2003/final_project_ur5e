import mink
import numpy as np

from arms.Arm import Arm


class FR3(Arm):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.end_effector_task: mink.FrameTask = mink.FrameTask(
            'attachment_site', 'site', position_cost=1.0, orientation_cost=0.5, lm_damping=1e-6)
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [
            self.end_effector_task,
            self.posture_task
        ]

        self.max_velocities = {
            "fr_joint1": 5/6 * np.pi,
            "fr_joint2": 5/6 * np.pi,
            "fr_joint3": 5/6 * np.pi,
            "fr_joint4": 5/6 * np.pi,
            "fr_joint5": 301/180*np.pi,
            "fr_joint6": 301/180*np.pi,
            "fr_joint7": 301/180*np.pi,
        }
        
        wrist_3_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("fr_joint7").id
        )
        collision_pairs = [
            (wrist_3_geoms, [
                "floor",
                "table",  # "wall1", "wall2", "wall3"
            ]),
        ]
        self.limits = [
            mink.ConfigurationLimit(
                model=self.config.model),
            mink.CollisionAvoidanceLimit(
                model=self.config.model,
                geom_pairs=collision_pairs,
                minimum_distance_from_collisions=0.5
            ),
            mink.VelocityLimit(self.model, self.max_velocities)
        ]

        self.initial_position = self.data.qpos[:self.model.nu].copy()
