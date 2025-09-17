from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_data import PlotData


def combine_and_plot(plots_list: list[PlotData], graphs_dir: Path):
    if not plots_list:
        print("No plot data to visualize.")
        return

    # Ensure output directory exists
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Sort by action index to maintain correct order
    plots_sorted = sorted(plots_list, key=lambda p: p.act_idx)

    all_times: list[float] = []
    all_joint_pos: list[np.ndarray] = []
    all_joint_speed: list[np.ndarray] = []
    all_tcp_pose: list[np.ndarray] = []
    all_tcp_speed: list[np.ndarray] = []
    # Keep boundaries between actions: (time, act_idx_of_previous_segment)
    boundaries: list[tuple[float, int]] = []

    t_offset = 0.0
    for p in plots_sorted:
        if p.times is None or p.times.size == 0:
            continue
        times = p.times.ravel()
        # Compute dt from this segment if possible
        dt = (times[1] - times[0]) if times.size > 1 else 0.0
        seg_times = (times + t_offset).tolist()

        # Accumulate
        all_times.extend(seg_times)
        if p.joint_position is not None and p.joint_position.size > 0:
            all_joint_pos.extend([row for row in p.joint_position])
        if p.joint_speed is not None and p.joint_speed.size > 0:
            all_joint_speed.extend([row for row in p.joint_speed])
        if p.tcp_pose is not None and p.tcp_pose.size > 0:
            all_tcp_pose.extend([row for row in p.tcp_pose])
        if p.tcp_speed is not None and p.tcp_speed.size > 0:
            all_tcp_speed.extend([row for row in p.tcp_speed])

        # Mark a boundary at the end of this segment (label with this segment's act_idx)
        seg_end = seg_times[-1] if len(seg_times) > 0 else t_offset
        boundaries.append((seg_end, getattr(p, 'act_idx', 0)))

        # Advance offset to make time continuous across actions
        t_offset = seg_end + (dt if dt > 0 else 0)

    # Convert to arrays
    T = np.array(all_times)
    JP = np.array(all_joint_pos) if all_joint_pos else np.empty((0,6))
    JV = np.array(all_joint_speed) if all_joint_speed else np.empty((0,6))
    TP = np.array(all_tcp_pose) if all_tcp_pose else np.empty((0,6))
    TV = np.array(all_tcp_speed) if all_tcp_speed else np.empty((0,6))

    # Guard: if no data, skip plotting
    if T.size == 0:
        print("No time samples collected; skipping plots.")
        return

    # Helper to draw vertical boundaries on one or more axes
    def draw_boundaries(axes):
        ax_list = axes if isinstance(axes, (list, np.ndarray, tuple)) else [axes]
        # skip the final boundary (no action after it)
        bnds = boundaries[:-1] if len(boundaries) > 1 else []
        for ax in ax_list:
            if not bnds:
                continue
            ymin, ymax = ax.get_ylim()
            for x, act_id in bnds:
                ax.axvline(x=x, color='k', linestyle=':', linewidth=1, alpha=0.5)
                # place label at top with slight padding
                ax.text(x, ymax, f'act {act_id}', rotation=90, va='top', ha='right',
                        fontsize=8, alpha=0.7,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0))

    # TCP Pose (position and orientation vector)
    if TP.size > 0:
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            axs[0].plot(T, TP[:, i], label=labels[i])
        axs[0].set_ylabel('Position [m]')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='best')

        labels_r = ['rx', 'ry', 'rz']
        for i in range(3):
            axs[1].plot(T, TP[:, 3 + i], label=labels_r[i])
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Rotation vector [rad]')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='best')
        # add boundaries on both subplots
        draw_boundaries([axs[0], axs[1]])
        fig.tight_layout()
        fig.savefig((graphs_dir / 'tcp_pose.png').as_posix(), dpi=150)
        plt.close(fig)

    # TCP Speed (spatial velocity)
    if TV.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        spd_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        for i in range(min(6, TV.shape[1])):
            ax.plot(T, TV[:, i], label=spd_labels[i] if i < len(spd_labels) else f'c{i}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('TCP speed [m/s, rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'tcp_speed.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Positions
    if JP.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(min(6, JP.shape[1])):
            ax.plot(T, JP[:, i], label=f'J{i+1}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint position [rad]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'joint_positions.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Speeds
    if JV.size > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(min(6, JV.shape[1])):
            ax.plot(T, JV[:, i], label=f'J{i+1}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint speed [rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncols=3)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((graphs_dir / 'joint_speeds.png').as_posix(), dpi=150)
        plt.close(fig)

    print("Saved plots to:")
    print(f" - {graphs_dir / 'tcp_pose.png'}")
    print(f" - {graphs_dir / 'tcp_speed.png'}")
    print(f" - {graphs_dir / 'joint_positions.png'}")
    print(f" - {graphs_dir / 'joint_speeds.png'}")


# --- Combine robot_plots and mujoco_plots for comparison ---
def _make_continuous(plots_list: list[PlotData]):
    # Sort by action index
    plots_sorted = sorted(plots_list, key=lambda p: p.act_idx)
    all_times: list[float] = []
    all_joint_pos: list[np.ndarray] = []
    all_joint_speed: list[np.ndarray] = []
    all_tcp_pose: list[np.ndarray] = []
    all_tcp_speed: list[np.ndarray] = []
    boundaries: list[tuple[float, int]] = []  # (time, act_idx)

    t_offset = 0.0
    for p in plots_sorted:
        if p.times is None or p.times.size == 0:
            continue
        times = p.times.ravel()
        dt = (times[1] - times[0]) if times.size > 1 else 0.0
        seg_times = (times + t_offset).tolist()

        all_times.extend(seg_times)
        if p.joint_position is not None and p.joint_position.size > 0:
            all_joint_pos.extend([row for row in p.joint_position])
        if p.joint_speed is not None and p.joint_speed.size > 0:
            all_joint_speed.extend([row for row in p.joint_speed])
        if p.tcp_pose is not None and p.tcp_pose.size > 0:
            all_tcp_pose.extend([row for row in p.tcp_pose])
        if p.tcp_speed is not None and p.tcp_speed.size > 0:
            all_tcp_speed.extend([row for row in p.tcp_speed])

        seg_end = seg_times[-1] if seg_times else t_offset
        boundaries.append((seg_end, getattr(p, 'act_idx', 0)))
        t_offset = seg_end + (dt if dt > 0 else 0)

    return {
        'T': np.array(all_times),
        'JP': np.array(all_joint_pos) if all_joint_pos else np.empty((0, 6)),
        'JV': np.array(all_joint_speed) if all_joint_speed else np.empty((0, 6)),
        'TP': np.array(all_tcp_pose) if all_tcp_pose else np.empty((0, 6)),
        'TV': np.array(all_tcp_speed) if all_tcp_speed else np.empty((0, 6)),
        'boundaries': boundaries,
    }


def combine_plots_for_comparison(robot_plots_list: list[PlotData], mujoco_plots_list: list[PlotData]):
    """Create continuous series from both robot and mujoco plots for side-by-side comparison.
    Returns a dict with keys 'robot' and 'mujoco', each containing T, JP, JV, TP, TV, boundaries.
    """
    return {
        'robot': _make_continuous(robot_plots_list),
        'mujoco': _make_continuous(mujoco_plots_list),
    }


def plot_comparison(combined: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    R = combined['robot']
    M = combined['mujoco']

    Tr, Jr, Vr, Tpr, Tvr = R['T'], R['JP'], R['JV'], R['TP'], R['TV']
    Tm, Jm, Vm, Tpm, Tvm = M['T'], M['JP'], M['JV'], M['TP'], M['TV']
    boundaries = R['boundaries']

    # --- Align lengths within each data source (robot, mujoco) to avoid mismatched x/y lengths ---
    def _truncate_common(T: np.ndarray, *arrays: np.ndarray):
        lengths = [len(T)] + [a.shape[0] for a in arrays if a.size > 0]
        if not lengths:
            return T, arrays
        common = min(lengths)
        if common == len(T) and all(a.shape[0] == common for a in arrays if a.size > 0):
            return T, arrays
        # Warn once
        print(f"[plot_comparison] Warning: truncating to common length {common} (time={len(T)}, arrays={[a.shape[0] for a in arrays]})")
        T_new = T[:common]
        arrays_new = tuple(a[:common] if a.shape[0] >= common else a for a in arrays)
        return T_new, arrays_new

    # Truncate robot series
    Tr, (Jr, Vr, Tpr, Tvr) = _truncate_common(Tr, Jr, Vr, Tpr, Tvr)
    # Truncate mujoco series
    Tm, (Jm, Vm, Tpm, Tvm) = _truncate_common(Tm, Jm, Vm, Tpm, Tvm)

    def draw_boundaries(axs):
        ax_list = axs if isinstance(axs, (list, tuple, np.ndarray)) else [axs]
        bnds = boundaries[:-1] if len(boundaries) > 1 else []
        for ax in ax_list:
            if not bnds:
                continue
            ymin, ymax = ax.get_ylim()
            for x, act_id in bnds:
                ax.axvline(x=x, color='k', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(x, ymax, f'act {act_id}', rotation=90, va='top', ha='right', fontsize=8,
                        alpha=0.7, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0))

    # TCP Pose comparison
    if Tpr.size > 0 or Tpm.size > 0:
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        labels = ['x', 'y', 'z']
        cmap = plt.get_cmap('tab10')
        pose_colors = [cmap(i) for i in range(6)]  # reuse colors for pos (0-2) and rot (3-5)
        for i in range(3):
            color = pose_colors[i]
            if Tpr.size > 0:
                axs[0].plot(Tr, Tpr[:, i], label=f'Robot {labels[i]}', linestyle='-', color=color)
            if Tpm.size > 0:
                axs[0].plot(Tm, Tpm[:, i], label=f'Mujoco {labels[i]}', linestyle='--', color=color)
        axs[0].set_ylabel('Position [m]')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(ncols=3, fontsize=8)

        labels_r = ['rx', 'ry', 'rz']
        for i in range(3):
            color = pose_colors[3 + i]
            if Tpr.size > 0:
                axs[1].plot(Tr, Tpr[:, 3 + i], label=f'Robot {labels_r[i]}', linestyle='-', color=color)
            if Tpm.size > 0:
                axs[1].plot(Tm, Tpm[:, 3 + i], label=f'Mujoco {labels_r[i]}', linestyle='--', color=color)
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Rotation vector [rad]')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(ncols=3, fontsize=8)
        draw_boundaries([axs[0], axs[1]])
        fig.tight_layout()
        fig.savefig((out_dir / 'compare_tcp_pose.png').as_posix(), dpi=150)
        plt.close(fig)

    # TCP Speed comparison
    if Tvr.size > 0 or Tvm.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        spd_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        for i in range(6):
            if Tvr.size > 0:
                ax.plot(Tr, Tvr[:, i], label=f'Robot {spd_labels[i]}', linestyle='-')
            if Tvm.size > 0:
                ax.plot(Tm, Tvm[:, i], label=f'Mujoco {spd_labels[i]}', linestyle='--')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('TCP speed [m/s, rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=3, fontsize=8)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((out_dir / 'compare_tcp_speed.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Position comparison
    if Jr.size > 0 or Jm.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        cmap = plt.get_cmap('tab10')
        joint_colors = [cmap(i) for i in range(10)]
        for i in range(6):
            color = joint_colors[i % len(joint_colors)]
            if Jr.size > 0:
                ax.plot(Tr, Jr[:, i], label=f'Robot J{i+1}', linestyle='-', color=color)
            if Jm.size > 0:
                ax.plot(Tm, Jm[:, i], label=f'Mujoco J{i+1}', linestyle='--', color=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint position [rad]')
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=3, fontsize=8)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((out_dir / 'compare_joint_positions.png').as_posix(), dpi=150)
        plt.close(fig)

    # Joint Speed comparison
    if Vr.size > 0 or Vm.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        cmap = plt.get_cmap('tab10')
        joint_colors = [cmap(i) for i in range(10)]
        for i in range(6):
            color = joint_colors[i % len(joint_colors)]
            if Vr.size > 0:
                ax.plot(Tr, Vr[:, i], label=f'Robot J{i+1}', linestyle='-', color=color)
            if Vm.size > 0:
                ax.plot(Tm, Vm[:, i], label=f'Mujoco J{i+1}', linestyle='--', color=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint speed [rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=3, fontsize=8)
        draw_boundaries(ax)
        fig.tight_layout()
        fig.savefig((out_dir / 'compare_joint_speeds.png').as_posix(), dpi=150)
        plt.close(fig)
