#!/usr/bin/env python3
"""Plot per-joint arm trajectories logged by spot_warehouse.py's ArmDataLogger.

For each arm joint, draws two stacked panels:
  - top:    commanded (cuRobo) position vs actual measured position
  - bottom: measured effort (torque) vs the joint's effort limit (dashed)

The effort panel is the torque diagnosis: if the arm can't lift the grasped
object, the measured effort pins against the dashed limit line while the actual
position stalls below the commanded position.

Usage:
    python3 plot_arm_trajectory.py /workspace/outputs/arm_gain_log_<ts>.csv
    python3 plot_arm_trajectory.py <csv> --save out.png      # headless
    python3 plot_arm_trajectory.py <csv> --joints arm_sh1,arm_el0
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to arm_gain_log CSV")
    ap.add_argument("--joints", default=None,
                    help="Comma-separated subset of joints to plot (default: all present)")
    ap.add_argument("--save", default=None, help="Save figure to this path instead of showing")
    args = ap.parse_args()

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    t = df["t_sim"].to_numpy()

    # Effort limits sidecar (written by ArmDataLogger.resolve_indices)
    limits = {}
    meta_path = args.csv + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            limits = json.load(fh).get("effort_limits", {})

    if args.joints:
        joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    else:
        joints = [j for j in ARM_JOINTS if f"{j}_pos" in df.columns]

    n = len(joints)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7), sharex=True, squeeze=False)

    for col, j in enumerate(joints):
        ax_pos = axes[0][col]
        ax_eff = axes[1][col]

        # --- position tracking ---
        if f"{j}_tgt" in df.columns:
            ax_pos.plot(t, df[f"{j}_tgt"], label="cuRobo cmd", color="tab:blue", lw=1.2)
        ax_pos.plot(t, df[f"{j}_pos"], label="actual", color="tab:orange", lw=1.2)
        ax_pos.set_title(j)
        ax_pos.grid(True, alpha=0.3)
        if col == 0:
            ax_pos.set_ylabel("position [rad]")
            ax_pos.legend(fontsize=8, loc="best")

        # --- effort vs limit ---
        if f"{j}_eff" in df.columns:
            ax_eff.plot(t, df[f"{j}_eff"], color="tab:red", lw=1.0, label="measured effort")
        lim = limits.get(j)
        if lim is not None and np.isfinite(lim):
            ax_eff.axhline(lim, ls="--", color="k", lw=1.0, alpha=0.7, label="effort limit")
            ax_eff.axhline(-lim, ls="--", color="k", lw=1.0, alpha=0.7)
        ax_eff.grid(True, alpha=0.3)
        ax_eff.set_xlabel("t_sim [s]")
        if col == 0:
            ax_eff.set_ylabel("effort [N·m]")
            ax_eff.legend(fontsize=8, loc="best")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"saved {args.save}")

    # --- task-level figure: did the object actually lift, or did the EE slip? ---
    if "obj_z" in df.columns:
        obj_mass = None
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                obj_mass = json.load(fh).get("object_mass_kg")
        tfig, tax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        # heights: EE vs object — divergence = slip
        if "ee_z" in df.columns:
            tax[0].plot(t, df["ee_z"], label="EE z (hand)", color="tab:green")
        tax[0].plot(t, df["obj_z"], label="object z", color="tab:purple")
        ttl = "EE vs object height"
        if obj_mass is not None and np.isfinite(obj_mass):
            ttl += f"  (object mass {obj_mass:.3f} kg)"
        tax[0].set_title(ttl); tax[0].set_ylabel("z [m]")
        tax[0].legend(fontsize=8); tax[0].grid(True, alpha=0.3)
        # object speed
        ospeed = np.sqrt(df.get("obj_vx", 0)**2 + df.get("obj_vy", 0)**2 + df.get("obj_vz", 0)**2)
        tax[1].plot(t, ospeed, color="tab:purple", label="object |vel|")
        if "obj_vz" in df.columns:
            tax[1].plot(t, df["obj_vz"], color="tab:red", lw=0.8, label="object vz")
        tax[1].set_ylabel("m/s"); tax[1].legend(fontsize=8); tax[1].grid(True, alpha=0.3)
        # gripper grasp engagement
        if "arm_f1x_pos" in df.columns:
            tax[2].plot(t, df["arm_f1x_pos"], color="tab:blue", label="finger pos (0=closed)")
            tax[2].plot(t, df["arm_f1x_eff"], color="tab:red", lw=0.8, label="finger effort")
        tax[2].set_ylabel("rad / N·m"); tax[2].set_xlabel("t_sim [s]")
        tax[2].legend(fontsize=8); tax[2].grid(True, alpha=0.3)
        tfig.tight_layout()
        if args.save:
            task_path = args.save.replace(".png", "_task.png")
            if task_path == args.save:
                task_path = args.save + "_task.png"
            tfig.savefig(task_path, dpi=130)
            print(f"saved {task_path}")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
