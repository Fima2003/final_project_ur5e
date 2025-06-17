from loop_rate_limiters import RateLimiter
import mujoco
import mink
import numpy as np

from utils import get_mink_xml


class Arm:
    def __init__(self):
        scene_xml = get_mink_xml()
        self.model = mujoco.MjModel.from_xml_string(scene_xml)

        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        self.additional_actuators = 1  # 1 if no hook, 2 if hook is present

        self.initialize()

    def initialize(self):
        raise Exception("This method should be overridden in subclasses")

    def set_destination(self, position, rotation: mink.SO3 = mink.SO3.from_rpy_radians(0, np.pi, np.pi/2)):
        target = mink.SE3.from_rotation_and_translation(
            rotation,
            position
        )
        self.end_effector_task.set_target(target)

    def go(
        self,
        run_for=150,
        stay_for=150,
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
        final_control = np.concatenate(
            (final_control, self._stay(run_for=stay_for)))
        return final_control

    def _stay(self, run_for=150):
        final_control = []
        mujoco.mj_forward(self.model, self.data)

        # Preserve the current gripper and hook state
        nu = self.model.nu
        gripper_hook_state = self.data.qpos[nu -
                                            self.additional_actuators:nu].copy()
        arm_state = self.config.q[:nu-self.additional_actuators].copy()
        full_state = np.concatenate((arm_state, gripper_hook_state))

        for _ in range(run_for):
            final_control.append(full_state.copy())
            self.data.ctrl = full_state.copy()
            mujoco.mj_step(self.model, self.data)

        return final_control

    def grip(self, run_for=150, stay_for=100):
        final_control = []
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        arm_control = self.config.q[:6]
        hook_command = 0.2
        gripper_command = 1

        new_control = np.concatenate((arm_control, np.array([
            gripper_command, hook_command] if self.additional_actuators == 2 else [gripper_command])))
        final_control.append(new_control)

        # Update the robot's state with the new control
        self.data.qpos[:self.model.nu] = new_control
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

        final_control.extend(self._stay(run_for=stay_for))

        return final_control

    def ungrip(self):
        raise NotImplementedError("Ungrip action is not implemented")

    def reset(self):
        """
        Reset the robot's arm joints to the initial position, keeping the gripper and hook state unchanged.
        Returns:
            list: A list containing a single array of the full control state (for UR5e replica replay).
        """
        nu = self.model.nu
        self.data.qpos[:nu-self.additional_actuators] = self.initial_position[:nu -
                                                                              self.additional_actuators]
        print(self.data.qpos)
        final_control = self.data.qpos.copy()
        print("Resetting to initial position:", final_control)
        self.config.update(final_control)
        mujoco.mj_forward(self.model, self.data)
        # Return as array of arrays for replica replay
        return [self.config.q[:nu].copy() for i in range(400)]
