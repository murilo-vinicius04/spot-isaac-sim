from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import logging
import numpy as np
import os
import signal
import sys
import termios
import threading
import tty
from pathlib import Path

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/spot_warehouse_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SpotLocoPolicy(SpotArmFlatTerrainPolicy):
    """Drop-in replacement that matches the locomanipulation training obs layout.

    Training layout (69-dim):
      [lin_vel_b(3), ang_vel_b(3), gravity_b(3), cmd(3),
       leg_pos(12), leg_vel(12), arm_pos(7), arm_vel(7), last_action(19)]

    USD joint order: [fl_hy, fl_kn, fr_hy, fr_kn, hl_hy, hl_kn, hr_hy, hr_kn,
                      arm0_sh1, arm0_el0, arm0_el1, arm0_wr0, arm0_wr1, arm0_f1x, arm0_sh0,
                      fl_hx, fr_hx, hl_hx, hr_hx]
    Policy order:   [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy,
                      fl_kn, fr_kn, hl_kn, hr_kn, arm_sh0, arm_sh1, arm_el0, arm_el1,
                      arm_wr0, arm_wr1, arm_f1x]

    This subclass reorders joint data from USD order to policy order.
    Arm actions are zeroed — arm stays at default (Phase 2 training).
    """

    def __init__(self, *args, **kwargs):
        logger.info("=== SPOTLOCOPOLICY INITIALIZING ===")
        super().__init__(*args, **kwargs)

        # Override decimation to match standalone deploy: 200Hz physics / 4 = 50Hz policy
        self._decimation = 4
        logger.info(f"Decimation set to {self._decimation} (50Hz policy)")

        # USD joint order
        self.usd_joint_order = [
            'fl_hy', 'fl_kn', 'fr_hy', 'fr_kn', 'hl_hy', 'hl_kn', 'hr_hy', 'hr_kn',
            'arm0_sh1', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x', 'arm0_sh0',
            'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx'
        ]
        logger.info(f"USD joint order ({len(self.usd_joint_order)}): {self.usd_joint_order}")

        # Policy expected order
        self.policy_joint_order = [
            'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy',
            'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn', 'arm_sh0', 'arm_sh1', 'arm_el0', 'arm_el1',
            'arm_wr0', 'arm_wr1', 'arm_f1x'
        ]
        logger.info(f"Policy joint order ({len(self.policy_joint_order)}): {self.policy_joint_order}")

        # Override arm default positions to more stable configuration
        # The env.yaml defaults (arm0_sh1=-3.13, arm0_el0=3.13) are very extended and cause instability
        # Using more neutral stow position instead
        self._stable_arm_defaults = {
            'arm0_sh0': 0.0,     # shoulder rotation
            'arm0_sh1': 0.0,     # shoulder elevation (was -3.13, now neutral)
            'arm0_el0': 0.5,     # elbow flex (was 3.13, now moderate)
            'arm0_el1': 0.0,
            'arm0_wr0': 0.0,
            'arm0_wr1': 0.0,
            'arm0_f1x': 0.0
        }
        logger.info(f"Stable arm defaults: {self._stable_arm_defaults}")

        # Create mapping from USD indices to policy indices
        # Note: arm0_* in USD maps to arm_* in policy
        self.usd_to_policy_idx = {}
        for policy_idx, policy_name in enumerate(self.policy_joint_order):
            # Map arm_sh0 → arm0_sh0, etc.
            usd_name = policy_name.replace('arm_', 'arm0_') if policy_name.startswith('arm_') else policy_name
            if usd_name in self.usd_joint_order:
                usd_idx = self.usd_joint_order.index(usd_name)
                self.usd_to_policy_idx[usd_idx] = policy_idx

        logger.info(f"Joint mappings: {len(self.usd_to_policy_idx)} joints mapped")
        logger.info("=== SPOTLOCOPOLICY INITIALIZED ===\n")

    def _get_stable_arm_defaults(self):
        """Get stable arm default positions in USD order"""
        defaults_usd = []
        for joint_name in self.usd_joint_order:
            if joint_name in self._stable_arm_defaults:
                defaults_usd.append(self._stable_arm_defaults[joint_name])
            else:
                # Use inherited default for leg joints
                idx = self.usd_joint_order.index(joint_name)
                defaults_usd.append(self.default_pos[idx])
        defaults_array = np.array(defaults_usd)
        logger.debug(f"Stable defaults in USD order: {defaults_array}")
        return defaults_array

    def _reorder_joints(self, usd_data):
        """Reorder joint data from USD order to policy order"""
        if usd_data.shape[0] != len(self.usd_joint_order):
            logger.error(f"USD data shape mismatch! Expected {len(self.usd_joint_order)}, got {usd_data.shape[0]}")
            logger.error(f"USD data: {usd_data}")
            raise ValueError("Joint data length mismatch")

        policy_data = np.zeros_like(usd_data)
        for usd_idx, policy_idx in self.usd_to_policy_idx.items():
            policy_data[policy_idx] = usd_data[usd_idx]

        logger.debug(f"Reordered joints - USD→Policy: {len(self.usd_to_policy_idx)} joints")
        return policy_data

    def _compute_observation(self, command):
        """Compute 69-dim observation with proper joint ordering"""
        logger.debug(f"Computing observation with command: {command}")

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_BI = quat_to_rot_matrix(q_IB).T
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        jpos = self.robot.get_joint_positions()   # [19] in USD order
        jvel = self.robot.get_joint_velocities()

        logger.debug(f"USD joint positions: {jpos}")
        logger.debug(f"USD joint velocities: {jvel}")

        # Reorder joints from USD order to policy order
        jpos_policy = self._reorder_joints(jpos)
        jvel_policy = self._reorder_joints(jvel)
        default_pos_policy = self._reorder_joints(self.default_pos)

        logger.debug(f"Policy joint positions: {jpos_policy}")
        logger.debug(f"Policy joint velocities: {jvel_policy}")
        logger.debug(f"Policy default positions: {default_pos_policy}")

        obs = np.concatenate([
            lin_vel_b,                                      # 3
            ang_vel_b,                                      # 3
            gravity_b,                                      # 3
            command,                                        # 3
            jpos_policy[:12] - default_pos_policy[:12],     # 12  leg pos
            jvel_policy[:12],                               # 12  leg vel
            jpos_policy[12:19] - default_pos_policy[12:19], # 7   arm pos
            jvel_policy[12:19],                             # 7   arm vel
            self._previous_action,                              # 19
        ])  # total = 69

        logger.debug(f"Observation shape: {obs.shape}, expected: (69,)")
        logger.debug(f"Observation ranges: lin_vel=[{obs[0:3].min():.3f},{obs[0:3].max():.3f}] "
                    f"ang_vel=[{obs[3:6].min():.3f},{obs[3:6].max():.3f}] "
                    f"cmd=[{obs[9:12].min():.3f},{obs[9:12].max():.3f}]")

        return obs

    def forward(self, dt, command):
        """Forward pass with joint reordering"""
        logger.debug(f"Forward call - counter: {self._policy_counter}, decimation: {self._decimation}")

        if self._policy_counter % self._decimation == 0:
            logger.info(f"=== POLICY STEP {self._policy_counter} ===")
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            logger.info(f"Policy action: {self.action[:5]}... (first 5 of 19)")
            logger.info(f"Action scale: {self._action_scale}")
        else:
            logger.debug(f"Skipping policy computation (counter {self._policy_counter} % decimation {self._decimation} != 0)")

        # Build joint targets in policy order first, then map back to USD order
        policy_targets = np.zeros(19)

        # Apply leg actions (first 12 dimensions)
        default_pos_policy = self._reorder_joints(self.default_pos)
        leg_targets = default_pos_policy[:12] + self.action[:12] * self._action_scale

        policy_targets[:12] = leg_targets
        logger.debug(f"Leg targets (policy order): {leg_targets[:5]}... (first 5 of 12)")

        # Keep arm at stable defaults (last 7 dimensions)
        arm_defaults_policy = self._reorder_joints(self._get_stable_arm_defaults())
        policy_targets[12:19] = arm_defaults_policy[12:19]

        logger.debug(f"Arm targets (policy order): {policy_targets[12:19]}")

        # Map back from policy order to USD order
        usd_targets = np.zeros(19)
        for usd_idx, policy_idx in self.usd_to_policy_idx.items():
            usd_targets[usd_idx] = policy_targets[policy_idx]

        logger.debug(f"Final USD targets: {usd_targets}")
        logger.debug(f"Target ranges: [{usd_targets.min():.3f}, {usd_targets.max():.3f}]")

        # Check for any obviously wrong values
        if np.any(np.abs(usd_targets) > 10):
            logger.warning(f"LARGE joint targets detected! Max abs: {np.max(np.abs(usd_targets)):.3f}")
        if np.any(np.isnan(usd_targets)):
            logger.error(f"NaN values in joint targets! {usd_targets}")

        self.robot.apply_action(ArticulationAction(joint_positions=usd_targets))
        self._policy_counter += 1
