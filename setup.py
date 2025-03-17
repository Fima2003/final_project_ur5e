import mujoco
import mink
import numpy as np


def setup():
    XML_PATH = "ur5e_scene.xml"
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id].copy()
        data.ctrl[:] = model.key_ctrl[key_id].copy()

    # Make sure the state is updated:
    mujoco.mj_forward(model, data)

    return model, data, mink_setup(model, data)


def mink_setup(model, data):
    configuration = mink.Configuration(model)
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between (wrist3, floor) and (wrist3, wall).
    wrist_3_geoms = mink.get_body_geom_ids(
        model, model.body("wrist_3_link").id)
    collision_pairs = [
        (wrist_3_geoms, ["floor", "base_box"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
    ]

    max_velocities = {
        "shoulder_pan_joint": np.pi,
        "shoulder_lift_joint": np.pi,
        "elbow_joint": np.pi,
        "wrist_1_joint": np.pi,
        "wrist_2_joint": np.pi,
        "wrist_3_joint": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)
    mink_config = {
        "configuration": configuration,
        "limits": limits,
        "tasks": tasks,
        "end_effector_task": end_effector_task,
    }

    return mink_config


if __name__ == "__main__":
    setup()
    