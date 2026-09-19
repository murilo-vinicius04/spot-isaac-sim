from pathlib import Path
from typing import Optional
import csv
import re

import numpy as np
import omni.kit.commands
import yaml
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from omni.physx import get_physx_simulation_interface


PHASE2_CONFIG_PATH = Path(__file__).with_name("spot_loco_phase2.yaml")


class SpotFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1


class SpotArmFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(69)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:31] = current_joint_pos - self.default_pos
        obs[31:50] = current_joint_vel
        # Previous Action
        obs[50:69] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1


def load_spot_loco_phase2_config(config_path: Optional[str] = None) -> dict:
    """Load shared Spot locomanipulation phase-2 configuration."""
    path = Path(config_path) if config_path is not None else PHASE2_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["policy_arm_order"] = config["policy_joint_order"][12:19]
    return config


def normalize_joint_name(name: str) -> str:
    """Normalize joint names across URDF and USD variants."""
    return name.replace("arm0_", "arm_")


def load_csv_poses(csv_path: str, arm_order: list[str], csv_joint_cols: list[str]) -> np.ndarray:
    """Load reachability poses and return a [N, 7] array in arm_order."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = [float(row[col]) for col in csv_joint_cols]
            reordered = []
            for joint_name in arm_order:
                if joint_name in csv_joint_cols:
                    reordered.append(vals[csv_joint_cols.index(joint_name)])
                else:
                    reordered.append(0.0)
            rows.append(reordered)
    return np.asarray(rows, dtype=np.float32)


def apply_trained_gains(robot, phase2_config: dict) -> None:
    """Apply trained phase-2 gains to a SingleArticulation robot."""
    joint_indices = []
    kps = []
    kds = []
    efforts = []
    gain_cfg = phase2_config["trained_gains"]

    for joint_name in robot.dof_names:
        normalized = normalize_joint_name(joint_name)
        if re.fullmatch(r".*_h[xy]", normalized):
            gains = gain_cfg["legs_hxy"]
        elif re.fullmatch(r".*_kn", normalized):
            gains = gain_cfg["legs_kn"]
        elif normalized in gain_cfg:
            gains = gain_cfg[normalized]
        else:
            continue

        joint_indices.append(robot.get_dof_index(joint_name))
        kps.append(gains["kp"])
        kds.append(gains["kd"])
        efforts.append(gains["effort"])

    robot._articulation_view.initialize()
    robot._articulation_view.set_gains(
        kps=np.asarray(kps, dtype=np.float32),
        kds=np.asarray(kds, dtype=np.float32),
        joint_indices=joint_indices,
    )
    robot._articulation_view.set_max_efforts(
        values=np.asarray(efforts, dtype=np.float32),
        joint_indices=joint_indices,
    )


class SpotLocoPolicy(SpotArmFlatTerrainPolicy):
    """Shared phase-2 Spot locomanipulation policy wrapper."""

    def __init__(
        self,
        *args,
        phase2_config: Optional[dict] = None,
        phase2_config_path: Optional[str] = None,
        physics_dt: Optional[float] = None,
        obs_mode: str = "loco",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obs_mode = obs_mode
        self.phase2_config = phase2_config or load_spot_loco_phase2_config(phase2_config_path)
        self._decimation = int(self.phase2_config["decimation"])
        self._physics_dt = float(physics_dt if physics_dt is not None else self.phase2_config["physics_dt"])
        self._action_scale = float(self.phase2_config["action_scale"]) if obs_mode == "loco" else 0.2
        self.policy_joint_order = list(self.phase2_config["policy_joint_order"])
        self.robot_joint_order = None
        self.robot_to_policy_idx = None
        self.arm_goal_policy = None
        self.arm_commanded_policy = None
        arm_vel = float(self.phase2_config["arm_tracking"]["max_vel"])
        self.arm_max_step = arm_vel * self._physics_dt * self._decimation

    def _setup_joint_ordering(self):
        """Build mapping from articulation DOF order to policy order."""
        self.robot_joint_order = list(self.robot.dof_names)
        self.robot_to_policy_idx = {}
        normalized_joint_names = [normalize_joint_name(name) for name in self.robot_joint_order]
        for robot_idx, joint_name in enumerate(normalized_joint_names):
            if joint_name in self.policy_joint_order:
                self.robot_to_policy_idx[robot_idx] = self.policy_joint_order.index(joint_name)

        if len(self.robot_to_policy_idx) != len(self.policy_joint_order):
            missing = [name for name in self.policy_joint_order if name not in normalized_joint_names]
            raise RuntimeError(f"Failed to map all policy joints. Missing: {missing}")

    def ensure_joint_ordering_ready(self):
        """Initialize joint ordering when articulation metadata becomes available."""
        if self.robot_to_policy_idx is not None:
            return
        if self.robot is None or self.robot.dof_names is None:
            raise RuntimeError("Robot DOF names are not available yet.")
        self._setup_joint_ordering()

    def initialize(
        self,
        physics_sim_view=None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        """Initialize without relying on the stock env.yaml joint-property loader."""
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        get_physx_simulation_interface().flush_changes()
        self.robot.get_articulation_controller().switch_control_mode(control_mode)

        self.ensure_joint_ordering_ready()
        self.default_pos = self._build_default_joint_positions()
        self.default_vel = np.zeros(len(self.robot_joint_order), dtype=np.float32)
        default_pos_policy = self._reorder_joints(self.default_pos)
        self.arm_goal_policy = default_pos_policy[12:19].copy()
        self.arm_commanded_policy = default_pos_policy[12:19].copy()

        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(self.default_vel)

        if set_articulation_props:
            self._set_articulation_props()

    def _build_default_joint_positions(self) -> np.ndarray:
        """Build default joint positions from shared phase-2 config."""
        import fnmatch
        raw_init_joint_pos = (
            self.policy_env_params.get("scene", {})
            .get("robot", {})
            .get("init_state", {})
            .get("joint_pos", {})
        )
        # Normalize env.yaml keys (arm0_* -> arm_*) so they match URDF joint names
        init_joint_pos = {normalize_joint_name(k): v for k, v in raw_init_joint_pos.items()}

        if self.obs_mode == "arm":
            # For the arm policy, use env.yaml defaults exclusively (no phase-2 overrides).
            # env.yaml keys may be regex/glob patterns so use fnmatch matching.
            current_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
            default_pos = current_pos.copy()
            for idx, joint_name in enumerate(self.robot_joint_order):
                normalized = normalize_joint_name(joint_name)
                for pattern, value in init_joint_pos.items():
                    glob_pattern = pattern.replace(".", "*") + "*"
                    if fnmatch.fnmatch(normalized, glob_pattern):
                        default_pos[idx] = value
                        break
            return default_pos

        leg_defaults = self.phase2_config["standalone_leg_defaults"]
        arm_defaults = self.phase2_config["stable_arm_defaults"]
        current_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        default_pos = current_pos.copy()
        for idx, joint_name in enumerate(self.robot_joint_order):
            normalized = normalize_joint_name(joint_name)
            if normalized in leg_defaults:
                default_pos[idx] = leg_defaults[normalized]
            elif normalized in arm_defaults:
                default_pos[idx] = arm_defaults[normalized]
            elif joint_name in init_joint_pos:
                default_pos[idx] = init_joint_pos[joint_name]
            elif normalized in init_joint_pos:
                default_pos[idx] = init_joint_pos[normalized]
        return default_pos

    def _reorder_joints(self, robot_data):
        """Reorder joint arrays from articulation DOF order to policy order."""
        self.ensure_joint_ordering_ready()
        policy_data = np.zeros_like(robot_data)
        for robot_idx, policy_idx in self.robot_to_policy_idx.items():
            policy_data[policy_idx] = robot_data[robot_idx]
        return policy_data

    def set_arm_goal(self, arm_goal_policy: np.ndarray) -> None:
        """Set a new 7-DOF arm goal in policy order."""
        self.arm_goal_policy = np.asarray(arm_goal_policy, dtype=np.float32).copy()

    def set_default_arm_pose(self) -> None:
        """Return the arm goal to the configured stow pose."""
        default_pos_policy = self._reorder_joints(self.default_pos)
        self.arm_goal_policy = default_pos_policy[12:19].copy()

    def _compute_observation(self, command):
        if self.obs_mode == "arm":
            lin_vel_i = self.robot.get_linear_velocity()
            ang_vel_i = self.robot.get_angular_velocity()
            _, q_ib = self.robot.get_world_pose()
            r_bi = quat_to_rot_matrix(q_ib).T
            lin_vel_b = r_bi @ lin_vel_i
            ang_vel_b = r_bi @ ang_vel_i
            gravity_b = r_bi @ np.array([0.0, 0.0, -1.0])
            jpos = self._reorder_joints(self.robot.get_joint_positions())
            jvel = self._reorder_joints(self.robot.get_joint_velocities())
            default_policy = self._reorder_joints(self.default_pos)
            obs = np.zeros(69)
            obs[:3] = lin_vel_b
            obs[3:6] = ang_vel_b
            obs[6:9] = gravity_b
            obs[9:12] = command
            obs[12:31] = jpos - default_policy   # all 19 joints in policy order
            obs[31:50] = jvel
            obs[50:69] = self._previous_action
            return obs

        lin_vel_i = self.robot.get_linear_velocity()
        ang_vel_i = self.robot.get_angular_velocity()
        _, q_ib = self.robot.get_world_pose()

        r_bi = quat_to_rot_matrix(q_ib).T
        lin_vel_b = r_bi @ lin_vel_i
        ang_vel_b = r_bi @ ang_vel_i
        gravity_b = r_bi @ np.array([0.0, 0.0, -1.0])

        jpos_policy = self._reorder_joints(self.robot.get_joint_positions())
        jvel_policy = self._reorder_joints(self.robot.get_joint_velocities())
        default_pos_policy = self._reorder_joints(self.default_pos)

        return np.concatenate([
            lin_vel_b,
            ang_vel_b,
            gravity_b,
            command,
            jpos_policy[:12] - default_pos_policy[:12],
            jvel_policy[:12],
            jpos_policy[12:19] - default_pos_policy[12:19],
            jvel_policy[12:19],
            self._previous_action,
        ])

    def forward(self, dt, command):
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
            if self.obs_mode == "loco":
                if self.arm_goal_policy is not None and self.arm_commanded_policy is not None:
                    delta = self.arm_goal_policy - self.arm_commanded_policy
                    self.arm_commanded_policy = self.arm_commanded_policy + np.clip(
                        delta, -self.arm_max_step, self.arm_max_step
                    )

        if self.obs_mode == "arm":
            # Action is in policy order (all 19); reorder to robot DOF order before applying
            default_policy = self._reorder_joints(self.default_pos)
            policy_targets = default_policy + self.action * self._action_scale
            robot_targets = np.zeros(len(self.robot_joint_order))
            for robot_idx, policy_idx in self.robot_to_policy_idx.items():
                robot_targets[robot_idx] = policy_targets[policy_idx]
            self.robot.apply_action(ArticulationAction(joint_positions=robot_targets))
        else:
            policy_targets = np.zeros(len(self.policy_joint_order))
            default_pos_policy = self._reorder_joints(self.default_pos)
            policy_targets[:12] = default_pos_policy[:12] + self.action[:12] * self._action_scale

            if self.arm_commanded_policy is None:
                policy_targets[12:19] = self._reorder_joints(self.default_pos)[12:19]
            else:
                policy_targets[12:19] = self.arm_commanded_policy

            robot_targets = np.zeros(len(self.robot_joint_order))
            for robot_idx, policy_idx in self.robot_to_policy_idx.items():
                robot_targets[robot_idx] = policy_targets[policy_idx]

            self.robot.apply_action(ArticulationAction(joint_positions=robot_targets))

        self._policy_counter += 1
