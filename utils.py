import mujoco
import trimesh


def create_soft_box():
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    cube.export('soft_cube.stl')
    print("Soft cube mesh saved.")


def get_soft_cube_position(model, data):
    cube_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "soft_cube")
    if cube_body_id < 0:
        raise ValueError("Soft cube body not found in the model.")

    # Use the helper function to get the cube's position.
    cube_pos = data.xpos[cube_body_id].copy()
    print("Soft cube position:", cube_pos)
    return cube_pos
