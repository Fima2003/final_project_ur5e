from loop_rate_limiters import RateLimiter
import mujoco
import mink
import numpy as np


class UR5e:
    def __init__(self, scene_xml: str = './mink_scene.xml'):
        self.model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())

        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)

        self.initialize()

    def initialize(self):
        self.end_effector_task: mink.FrameTask = mink.FrameTask(
            'attachment_site', 'site', position_cost=1.0, orientation_cost=0.5, lm_damping=1e-6)
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [
            self.end_effector_task,
            self.posture_task
        ]

        self.max_velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }
        wrist_3_geoms = mink.get_body_geom_ids(
            self.model, self.model.body("wrist_3_link").id)
        collision_pairs = [
            (wrist_3_geoms, [
                "floor", 
                # "table", "wall1", "wall2", "wall3"
            ]),
        ]
        self.limits = [
            mink.ConfigurationLimit(
                model=self.config.model),
            mink.CollisionAvoidanceLimit(
                model=self.config.model,
                geom_pairs=collision_pairs,
                minimum_distance_from_collisions=0.1
            ),
            mink.VelocityLimit(self.model, self.max_velocities)
        ]

    def set_destination(self, position, rotation: mink.SO3 = mink.SO3.from_rpy_radians(0, np.pi, np.pi/2)):
        target = mink.SE3.from_rotation_and_translation(
            rotation,
            position
        )
        self.end_effector_task.set_target(target)

    def go(
        self,
        run_for=150,
        frequency=200.0,
        solver="daqp",
        max_iters=20,
        pos_threshold=1e-4,
        ori_threshold=1e-4
    ):
        final_control = []
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

        self.posture_task.set_target_from_configuration(self.config)
        rate = RateLimiter(frequency, warn=False)

        for i in range(run_for):
            for j in range(max_iters):
                vel = mink.solve_ik(self.config, self.tasks,
                                    rate.dt, solver, limits=self.limits)
                self.config.integrate_inplace(vel, rate.dt)
                err = self.end_effector_task.compute_error(self.config)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            final_control.append(self.config.q[:self.model.nu])
            self.data.ctrl = self.config.q[:self.model.nu]
            mujoco.mj_forward(self.model, self.data)

        return final_control

    def stay(self, dur=150):
        final_control = []
        mujoco.mj_forward(self.model, self.data)
        print(f"Current configuration: {self.config.q[:self.model.nu]}")

        for _ in range(dur):
            final_control.append(self.config.q[:self.model.nu])
            self.data.ctrl = self.config.q[:self.model.nu]
            mujoco.mj_step(self.model, self.data)
        
        print(f"Final configuration: {self.config.q[:self.model.nu]}")

        return final_control

    def grip(self):
        final_control = []
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        arm_control = self.config.q[:6]
        hook_command = 0.2
        gripper_command = 0.29

        new_control = np.concatenate((arm_control, np.array(
            [gripper_command, hook_command])))
        final_control.append(new_control)

        return final_control

    def ungrip(self):
        # Ungrip the object
        raise NotImplementedError("Ungrip action is not implemented")
