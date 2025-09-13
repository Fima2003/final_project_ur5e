from typing import Sequence
from loop_rate_limiters import RateLimiter
import mujoco
import mink
import numpy as np

from mink_utils import get_mink_xml


class Arm:
    end_effector_task: mink.FrameTask
    limits: Sequence[mink.Limit]
    tasks: Sequence[mink.Task]
    # posture_task: mink.PostureTask
    
    def __init__(self, scene_path='./scene.xml', actuators=6, additional_actuators=1):
        scene_xml = get_mink_xml(scene_path=scene_path)
        self.model = mujoco.MjModel.from_xml_string(scene_xml) # type: ignore

        self.data = mujoco.MjData(self.model) # type: ignore
        self.config = mink.Configuration(self.model)
        self.actuators = actuators
        self.additional_actuators = additional_actuators  # 1 if no hook, 2 if hook is present

        self.initialize()

    def initialize(self):
        raise Exception("This method should be overridden in subclasses")
    
    def set_control(self, control):
        self.data.ctrl = control.copy()
        for i in range(1000):
            mujoco.mj_step(self.model, self.data) # type: ignore
        self.config.update(self.data.qpos)
        

    def set_destination(self, position, rotation: mink.SO3 = mink.SO3.from_rpy_radians(0, np.pi, np.pi/2)):
        target = mink.SE3.from_rotation_and_translation(
            rotation,
            position
        )
        self.end_effector_task.set_target(target)

    def go(
        self,
        frequency=200.0,
        solver="daqp",
        max_iters=10000,
        pos_threshold=0.025,
        ori_threshold=0.025
    ):
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data) # type: ignore

        # self.posture_task.set_target_from_configuration(self.config)
        rate = RateLimiter(frequency, warn=False)

        for _ in range(max_iters):
            vel = mink.solve_ik(self.config, self.tasks,
                                rate.dt, solver, limits=self.limits)
            self.config.integrate_inplace(vel, rate.dt)
            err = self.end_effector_task.compute_error(self.config)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                print(np.linalg.norm(err[:3]))
                print("Achieved target earlier")
                break

        
        self.data.ctrl = self.config.q[:self.model.nu]
        mujoco.mj_forward(self.model, self.data) # type: ignore
        return self.config.q[:self.model.nu]

    def _stay(self, run_for=150):
        final_control = []
        mujoco.mj_forward(self.model, self.data) # type: ignore

        # Preserve the current gripper and hook state
        nu = self.model.nu
        gripper_hook_state = self.data.qpos[nu -
                                            self.additional_actuators:nu].copy()
        arm_state = self.config.q[:nu-self.additional_actuators].copy()
        full_state = np.concatenate((arm_state, gripper_hook_state))

        for _ in range(run_for):
            final_control.append(full_state.copy())
            self.data.ctrl = full_state.copy()
            mujoco.mj_step(self.model, self.data) # type: ignore

        return final_control

    def grip(self, run_for=150, stay_for=100):
        final_control = []
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data) # type: ignore
        arm_control = self.config.q[:self.actuators]
        hook_command = 0.22
        gripper_command = 1

        new_control = np.concatenate((arm_control, np.array([
            gripper_command, hook_command] if self.additional_actuators == 2 else [gripper_command])))
        final_control.append(new_control)

        # Update the robot's state with the new control
        self.data.qpos[:self.model.nu] = new_control
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data) # type: ignore

        final_control.extend(self._stay(run_for=stay_for))

        return final_control

    def release(self, run_for=150, stay_for=100):
        final_control = []
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data) # type: ignore
        arm_control = self.config.q[:self.actuators]
        hook_command = 0.2
        gripper_command = 0

        new_control = np.concatenate((arm_control[:len(arm_control)-1], np.array([-np.pi/2-np.pi/4]), np.array([
            gripper_command, hook_command] if self.additional_actuators == 2 else [gripper_command])))
        final_control.append(new_control)

        # Update the robot's state with the new control
        self.data.qpos[:self.model.nu] = new_control
        self.config.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data) # type: ignore

        final_control.extend(self._stay(run_for=stay_for))

        return final_control