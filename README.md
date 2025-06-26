# Universal Robots UR5e Simulation

This project simulates a Universal Robots UR5e robotic arm using MuJoCo. The simulation incorporates inverse kinematics (IK), collision avoidance, and tactile feedback for tasks such as grasping a soft object with a specialized gripper. The project leverages the [mujoco](https://mujoco.org) physics engine, [mink](https://github.com/stanfordnmmp/mink), and other libraries such as NumPy and Matplotlib.

## Features

- **Inverse Kinematics (IK):** Compute arm motions to reach a target pose.

- **Collision Avoidance:** Prevent the arm from colliding with the floor, walls, or other objects.

- **Soft Object Interaction:** Squeeze a soft cube to grasp it.

- **Data Recording & Visualization:** Record joint positions and velocities and plot their evolution over time.

## Installation

1. Ensure you have Python (>=3.8) installed.

2. Install the project dependencies:

```sh

pip install -r requirements.txt

```

3. Set up MuJoCo and the necessary licenses as described on the [MuJoCo website](https://mujoco.org).

## Usage

There are 3 parts to every simulation:

1. **Environment:** The simulation environment is defined using XML files, including objects, walls, and floor. It should have `<environment>` tag, under which all of the boundaries should be placed. This is done for Simplifying the Inverse Kinematics. The rest should be placed outside of `<environment>` tag. For example, refer to the [lab.xml](environments/lab/lab.xml) file for a sample environment configuration. The Environment should be placed under `environments/`

2. **Arm:** The UR5e robotic arm model is used for manipulation tasks using XML. You can copy-paste arms from [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main). The basic requirements are: the main tag of `<worldbody>` should have `class="base"`, and the eef should be `<site pos="0 0 0.045" name="attachment_site" />`. For example, refer to [Franka Research 3 robotic arm](arms/fr3/fr3.xml)

3. **Gripper:** A gripper designed for object interaction and grasping using XML. For example, refer to [Robotiq 2F85](grippers/2f85/2f85.xml)

4. **Creating a scene**: To run a simulation, you first need to create a scene. It is done automatically using [create_scene.py](create_scene.py). An example of usage:
    ```
    python create_scene.py <scene_name> <environment_name> <arm_name> <gripper_name>
    ```
    all of the names are self-explanatory. Running without any parameters will create scene called lab-ur5e with ur5e as robotic arm, 2f85 with custom gripper as a gripper, and a lab environment.
5. **Execution of Scene**:
Executing a scene is very straightforward:
    ```
    python execute_scene.py <scene_name>
    ```
    Executables folder can be modified to your own risk.

#### Good Luck and בהצלחה!


## License

This project is provided as-is for research and prototyping purposes.

## Acknowledgments

- [MuJoCo](https://mujoco.org/)

- [Mink](https://github.com/stanfordnmmp/mink)

- NumPy, Matplotlib, and other community libraries.
