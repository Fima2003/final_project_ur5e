
# Universal Robots UR5e Simulation

  

This project simulates a Universal Robots UR5e robotic arm using MuJoCo. The simulation incorporates inverse kinematics (IK), collision avoidance, and tactile feedback for tasks such as grasping a soft object with a specialized gripper. The project leverages the [mujoco](https://mujoco.org) physics engine, [mink](https://github.com/stanfordnmmp/mink), and other libraries such as NumPy and Matplotlib.

  

## Features

  

-  **Inverse Kinematics (IK):** Compute arm motions to reach a target pose.

-  **Collision Avoidance:** Prevent the arm from colliding with the floor, walls, or other objects.

-  **Soft Object Interaction:** Squeeze a soft cube to grasp it.

-  **Data Recording & Visualization:** Record joint positions and velocities and plot their evolution over time.

-  **Custom Environment:** Defined via XML files ([scene.xml](scene.xml) and [ur5e.xml](ur5e.xml)) with a set of 3D assets.

  

## File Structure

  

-  **main.py** – Entry point for the simulation. Sets up objects, configures the camera, and drives the control loop. See [main.py](main.py).

-  **utils.py** – Contains helper functions for IK setup and movement. See [utils.py](utils.py).

-  **ik_settings.py** – Holds settings and parameters for inverse kinematics.

-  **scene.xml** – Defines the environment (floor, walls, soft object) used in the simulation.

-  **ur5e.xml** – Contains robot model definitions including links, joints, and visual meshes.

-  **assets/** – Contains mesh files (e.g. OBJ, STL) for the robot and environment.

-  **graphs/** – Directory for generated plots.

-  **requirements.txt** – Lists the dependencies required to run the simulation.

  

## Installation

  

1. Ensure you have Python (>=3.8) installed.

2. Install the project dependencies:

```sh

pip install -r requirements.txt

```

3. Set up MuJoCo and the necessary licenses as described on the [MuJoCo website](https://mujoco.org).

  

## Usage

  

Run the simulation by executing:

  

```sh

python  main.py

```

  

The simulation window will launch using MuJoCo’s viewer while data is recorded and saved for later visualization.

  

## Customization

  

- Modify the IK settings in [`ik_settings.py`](ik_settings.py).

- Adjust the simulation environment in [`scene.xml`](scene.xml).

- Update visualization options or plotting routines in [`main.py`](main.py).

  

## License

  

This project is provided as-is for research and prototyping purposes.

  

## Acknowledgments

  

-  [MuJoCo](https://mujoco.org/)

-  [Mink](https://github.com/stanfordnmmp/mink)

- NumPy, Matplotlib, and other community libraries.