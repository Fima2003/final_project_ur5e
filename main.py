import os
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt  # Import matplotlib for plotting

import mujoco
import mujoco_viewer

##############################################################################
# 1. UR5e DH Parameters & Closed-Form IK
##############################################################################

# Verify these for UR5e (in meters):
d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

def dh_transform(a, alpha, d, theta):
    """
    Standard Denavit-Hartenberg matrix.
    a: link length
    alpha: link twist
    d: link offset
    theta: joint angle
    """
    st, ct = math.sin(theta), math.cos(theta)
    sa, ca = math.sin(alpha), math.cos(alpha)
    return np.array([
        [ ct,         -st*ca,       st*sa,   a*ct ],
        [ st,          ct*ca,      -ct*sa,   a*st ],
        [  0,             sa,          ca,      d ],
        [  0,              0,           0,      1 ]
    ], dtype=float)

def partial_fk_joint3(t1, t2, t3):
    """
    Compute the transform from the base to the 'joint 3' axis (before wrist joints).
    Using the first 3 joints of UR5e with the chosen DH convention.
    """
    # T01
    T01 = dh_transform(0,       math.pi/2, d1, t1)
    # T12
    T12 = dh_transform(a2,      0,         0,  t2)
    # T23
    T23 = dh_transform(a3,      0,         0,  t3)
    return T01 @ T12 @ T23

def wrist_angles_from_T(T_j3_ee):
    """
    Extract t4, t5, t6 from the transform of the last 3-joint wrist.
    (Assumes Z-Y-X or a similar convention for the UR wrist.)
    
    Adjust if your D-H frames or wrist axes differ from this assumption!
    """
    R = T_j3_ee[:3, :3]
    # In a typical UR kinematic approach:
    #  t4 rotates about Z
    #  t5 rotates about Y
    #  t6 rotates about X
    # This snippet is illustrative; your actual frames may differ.

    # t5 = arccos(R[2,2])  (watch sign, singularities)
    # For numerical stability:
    cos_t5 = R[2, 2]
    cos_t5 = max(min(cos_t5, 1.0), -1.0)  # clamp
    t5 = math.acos(cos_t5)

    # near singularities:
    if abs(math.sin(t5)) < 1e-6:
        # fall-back approach
        t4 = 0.0
        t6 = math.atan2(-R[1,0], R[0,0])
    else:
        t4 = math.atan2(R[1, 2], R[0, 2])
        t6 = math.atan2(R[2, 1], -R[2, 0])
    
    return (t4, t5, t6)

def ur5e_ik(T_des):
    """
    Compute all possible closed-form IK solutions for the UR5e, given:
      T_des: 4x4 transform for the end-effector in the base frame.
    Returns a list of possible 6-vector solutions [t1, t2, t3, t4, t5, t6].
    """
    px, py, pz = T_des[0,3], T_des[1,3], T_des[2,3]
    R_des = T_des[:3,:3]
    
    # end-effector's local z-axis in world coords
    ez = R_des @ np.array([0, 0, 1])
    
    # wrist center (subtract d6 along the z-axis)
    wx = px - d6 * ez[0]
    wy = py - d6 * ez[1]
    wz = pz - d6 * ez[2]
    
    # We'll gather solutions here:
    solutions = []
    
    # 1) Theta1 candidates
    shoulder_1 = math.atan2(wy, wx)
    t1_candidates = [shoulder_1, shoulder_1 + math.pi]
    
    for t1 in t1_candidates:
        # compute r, s for triangle involving link2, link3
        r = math.sqrt(wx**2 + wy**2)
        # shift by first link offset
        # in some references, r might be: r - a1 if there's an a1 != 0
        # but with standard UR5e, a1=0, so we skip that.
        s = wz - d1
        
        # law of cosines for t3
        # distance^2 = a2^2 + a3^2 + 2*a2*a3*cos(t3)
        # D = (r^2 + s^2 - a2^2 - a3^2)/(2*a2*a3)
        D = (r**2 + s**2 - a2**2 - a3**2) / (2*a2*a3)
        # clamp D in [-1, 1]
        D = max(min(D, 1.0), -1.0)
        
        try:
            phi3 = math.acos(D)
        except ValueError:
            # No real solution if |D|>1
            continue
        
        # t3 can be +phi3 or -phi3
        t3_list = [phi3, -phi3]
        
        for t3 in t3_list:
            # Solve for t2 using geometry:
            #   k1 = a2 + a3*cos(t3)
            #   k2 = a3*sin(t3)
            #   r = sqrt(wx^2 + wy^2), s = wz - d1
            k1 = a2 + a3*math.cos(t3)
            k2 = a3*math.sin(t3)

            # We want:
            #   r = k1*cos(t2) + k2*sin(t2)
            #   s = k1*sin(t2) - k2*cos(t2)
            # => t2 = atan2( s*k1 - r*k2, r*k1 + s*k2 )
            denom = (r*k1 + s*k2)
            numer = (s*k1 - r*k2)
            t2 = math.atan2(numer, denom)
            
            # Now partial forward kinematics to joint3
            T_base_j3 = partial_fk_joint3(t1, t2, t3)
            # T_j3_ee = inv(T_base_j3)*T_des
            T_j3_ee = np.linalg.inv(T_base_j3) @ T_des

            t4, t5, t6 = wrist_angles_from_T(T_j3_ee)

            sol = np.array([t1, t2, t3, t4, t5, t6], dtype=float)
            # Wrap angles to [-pi, pi] if desired
            sol = (sol + math.pi) % (2*math.pi) - math.pi
            solutions.append(sol)
    
    return solutions


##############################################################################
# 2. Combine with the MuJoCo Scene + Viewer
##############################################################################

def main():
    XML_PATH = "ur5e_scene.xml"  # Path to your MuJoCo scene
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Create a viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Set the camera parameters before rendering:
    viewer.cam.azimuth = 0        # horizontal angle, in degrees
    viewer.cam.elevation = -20     # vertical angle, in degrees
    viewer.cam.distance = 3.0       # zoom, distance from the center
    viewer.cam.lookat[:] = [0.3, 0.0, 0.4]  # the point the camera looks at

    # Number of frames to record (simulation steps)
    fps = 30
    DURATION_SECONDS = 20
    frames_to_record = DURATION_SECONDS * fps

    # Initialize lists to store joint positions and velocities
    joint_positions = []  # List of lists: each sublist contains positions of all joints at a step
    joint_velocities = [] # List of lists: each sublist contains velocities of all joints at a step

    # Optional: set to the "home" keyframe
    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id].copy()
        data.ctrl[:] = model.key_ctrl[key_id].copy()

    # Desired end-effector transform T_des (4x4)
    # Example target: x=0.9, y=0.1, z=0.9 with no rotation
    T_des = np.eye(4)
    T_des[0,3] = 0.9
    T_des[1,3] = 0.1
    T_des[2,3] = 0.9

    # 1) Compute closed-form IK
    sol_list = ur5e_ik(T_des)
    print(f"Found {len(sol_list)} possible IK solutions.")

    # 2) Pick one solution (e.g., first valid)
    if len(sol_list) == 0:
        print("No valid IK solutions for the given T_des.")
    else:
        chosen_sol = sol_list[0]  # or apply your joint-limit filtering here
        print("Chosen solution (radians):", chosen_sol)

        # 3) Set joint angles in MuJoCo
        data.qpos[:6] = chosen_sol

        # 4) Recompute forward kinematics in MuJoCo
        mujoco.mj_forward(model, data)

    # 5) Simulation and Data Recording Loop
    print("Starting simulation and data recording...")
    for frame_idx in range(frames_to_record):
        # Step the physics
        mujoco.mj_step(model, data)

        # Render in the viewer
        viewer.render()

        # Record joint positions and velocities
        # Assuming the first 6 joints correspond to the UR5e arm
        current_positions = data.qpos[:6].copy()
        current_velocities = data.qvel[:6].copy()
        joint_positions.append(current_positions)
        joint_velocities.append(current_velocities)

        if frame_idx % 10 == 0:
            print(f"Recorded frame {frame_idx + 1}/{frames_to_record}")

    print("Simulation and data recording complete.")

    # 6. Close the viewer
    viewer.close()

    # 7. Convert recorded data to numpy arrays for plotting
    joint_positions = np.array(joint_positions)      # Shape: (frames, 6)
    joint_velocities = np.array(joint_velocities)    # Shape: (frames, 6)

    # 8. Create time array for plotting
    time_array = np.linspace(0, DURATION_SECONDS, frames_to_record)

    # 9. Plot Joint Positions
    plt.figure(figsize=(12, 8))
    for joint_idx in range(6):
        plt.plot(time_array, joint_positions[:, joint_idx], label=f'Joint {joint_idx + 1} Position')
    plt.title('UR5e Joint Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (radians)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('joint_positions.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

    # 10. Plot Joint Velocities
    plt.figure(figsize=(12, 8))
    for joint_idx in range(6):
        plt.plot(time_array, joint_velocities[:, joint_idx], label=f'Joint {joint_idx + 1} Velocity')
    plt.title('UR5e Joint Velocities Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (radians/sec)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('joint_velocities.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

    print("Plots saved as 'joint_positions.png' and 'joint_velocities.png'.")

if __name__ == "__main__":
    main()