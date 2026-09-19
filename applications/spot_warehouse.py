import sys
from isaacsim import SimulationApp
# --zed-operator: the anim-graph runtime and the ZED streamer extension must load at
# kit BOOT (like a GUI session that already has them enabled). Enabling them later
# half-loads omni.anim.graph.core ("Aborting Python node registration") and creating a
# character from a late-applied binding crashes kit natively.
_APP_CONFIG = {"headless": False}
if "--zed-operator" in sys.argv:
    _APP_CONFIG["extra_args"] = [
        "--ext-folder", "/workspace/zed-isaac-sim/exts",
        "--enable", "omni.anim.graph.bundle",
        "--enable", "sl.sensor.camera",
    ]
simulation_app = SimulationApp(_APP_CONFIG)

import carb
import csv
import logging
import numpy as np
import os
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension

# Custom warehouse (simple_warehouse + table + clutter objects pre-arranged)
WAREHOUSE_USD = str(Path(__file__).resolve().parent.parent / "assets" / "clutter" / "warehouse.usd")

# --- Optional ZED-operator demo scene (enabled with --zed-operator) --------------------
# Spawns a construction-worker character (anim.usd) whose arm IKs to a draggable
# wrist_target, watched by a virtual ZED that streams to the ZED SDK over IPC. With the
# zed wrapper container + wrist_detector running, the robot mirrors the operator's motion
# (for the teleop-loop README GIF). Dev-only paths; poses are rough defaults, tune in GUI.
ANIM_USD = "/workspace/anim.usd"
ZED_EXTS_DIR = "/workspace/zed-isaac-sim/exts"
ZED_X_USD = "/workspace/zed-isaac-sim/exts/sl.sensor.camera/data/usd/ZED_X.usdc"
# World coords (Spot spawns at (-2.5, 0, 0.7)); rotations are ZYX degrees.
OPERATOR_POS = (0.0, 1.6, 0.0)
OPERATOR_ROT_ZYX = (0.0, 0.0, 180.0)
ZED_POS = (0.6, 0.6, 1.4)
ZED_ROT_ZYX = (20.0, -10.0, 210.0)
ZED_STREAM_PORT = 30000
# Live-tuned framing persists here (written by the 'G' hotkey, auto-loaded next launch),
# so dragging the operator/ZED and walking the robot into place survives across runs
# without editing the constants above. Dev-only; absent file => the constants are used.
ZED_OPERATOR_POSE_FILE = "/workspace/zed_operator_poses.json"


def _load_zed_operator_poses():
    """Return the saved framing dict (operator/zed_x flat 4x4 + robot_spawn pos/quat) or {}."""
    import json
    try:
        with open(ZED_OPERATOR_POSE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}

from spot_cameras import ALL_CAMERAS, create_spot_cameras, initialize_cameras
from spot_ros_bridge import ROSBridgeBuilder
from spot_arm_subscriber import SpotArmCommandSubscriber
from spot_policy import (
    SpotLocoPolicy,
    apply_trained_gains,
    load_csv_poses,
    load_spot_loco_phase2_config,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

enable_extension("isaacsim.ros2.bridge")
try:
    enable_extension("isaacsim.asset.importer.urdf")
except Exception:
    try:
        enable_extension("omni.importer.urdf")
    except Exception:
        enable_extension("omni.isaac.urdf")

try:
    from isaacsim.asset.importer.urdf import _urdf
    _URDF_IMPORTER_FLAVOR = "isaacsim"
except ImportError:
    try:
        from omni.importer.urdf import _urdf
        _URDF_IMPORTER_FLAVOR = "omni.importer"
    except ImportError:
        from omni.isaac.urdf import _urdf
        _URDF_IMPORTER_FLAVOR = "omni.isaac"

if _URDF_IMPORTER_FLAVOR == "isaacsim":
    import omni.kit.commands


ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]


def _arm_gains(sh: float, el0_kd: float, el1_wr: float) -> dict:
    """Build an arm gain table parameterised by shoulder, el0, and el1/wrist kd."""
    return {
        "arm_sh0": (120.0, sh),
        "arm_sh1": (120.0, sh),
        "arm_el0": (120.0, el0_kd),
        "arm_el1": (100.0, el1_wr),
        "arm_wr0": (100.0, el1_wr),
        "arm_wr1": (100.0, el1_wr),
        "arm_f1x": (16.0, 0.32),
    }


# Sweep presets (rotated each interval). Each entry is (name, gains_dict).
# Format key: sh<kd_sh>_el0<kd_el0>_w<kd_el1_wr>
ARM_GAIN_SWEEP = [
    ("sh2_el0-2_w2",   _arm_gains(2.0, 2.0, 2.0)),   # original baseline
    ("sh4_el0-3_w3",   _arm_gains(4.0, 3.0, 3.0)),
    ("sh6_el0-5_w3",   _arm_gains(6.0, 5.0, 3.0)),   # prev recommendation (with wrist bump)
    ("sh8_el0-6_w4",   _arm_gains(8.0, 6.0, 4.0)),   # current NEW
    ("sh10_el0-8_w5",  _arm_gains(10.0, 8.0, 5.0)),  # near/over critical
]


def apply_arm_gains(robot, gains: dict) -> None:
    """Set kp/kd on the specified arm joints of a SingleArticulation."""
    indices, kps, kds = [], [], []
    for jname, (kp, kd) in gains.items():
        try:
            idx = robot.get_dof_index(jname)
        except Exception:
            continue
        indices.append(idx)
        kps.append(kp)
        kds.append(kd)
    if not indices:
        return
    robot._articulation_view.set_gains(
        kps=np.asarray(kps, dtype=np.float32),
        kds=np.asarray(kds, dtype=np.float32),
        joint_indices=indices,
    )


class ArmDataLogger:
    """Per-physics-step CSV logger for arm joint state, target, and active gain set."""

    def __init__(self, path: str, joint_names: list[str]) -> None:
        self._joint_names = list(joint_names)
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        header = [
            "t_sim", "gain_mode", "tgt_source",
            "pose_idx", "difficulty",
            "cmd_vx", "cmd_vy", "cmd_wz",
            "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
            "proj_grav_x", "proj_grav_y", "proj_grav_z",
            # grasped-object world pose+vel and end-effector world pose, so the
            # lift can be judged on the TASK (did the bottle rise / slip?) not
            # just joint torques. obj_* from the bottle rigid body, ee_* from
            # arm_link_wr1 (hand/palm). NaN when the handles aren't available.
            "obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz",
            "ee_x", "ee_y", "ee_z",
        ]
        for j in self._joint_names:
            header += [f"{j}_tgt", f"{j}_pos", f"{j}_vel", f"{j}_eff"]
        self._writer.writerow(header)
        self._t = 0.0
        self._joint_indices = None  # resolved lazily once robot.dof_names is ready
        self._policy_idx = None     # index into policy arm order

    def resolve_indices(self, robot, policy_arm_order: list[str], extra_meta: dict | None = None) -> None:
        self._joint_indices = [robot.get_dof_index(j) for j in self._joint_names]
        self._policy_idx = [policy_arm_order.index(j) for j in self._joint_names]
        # Capture the per-joint effort (torque) limits once and drop them in a
        # sidecar JSON so the plot script can draw saturation lines. If the arm
        # can't lift the object, the measured effort will be pinned at this limit.
        try:
            max_eff = np.asarray(robot._articulation_view.get_max_efforts()).reshape(-1)
            limits = {j: float(max_eff[i]) for j, i in zip(self._joint_names, self._joint_indices)}
        except Exception as exc:  # pragma: no cover - depends on Isaac build
            print(f"[ArmLogger] could not read effort limits: {exc}")
            limits = {j: float("nan") for j in self._joint_names}
        try:
            import json
            meta = {"effort_limits": limits, "joint_names": self._joint_names}
            if extra_meta:
                meta.update(extra_meta)
            with open(self._path + ".meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            print(f"[ArmLogger] effort limits: {limits}")
            if extra_meta:
                print(f"[ArmLogger] extra meta: {extra_meta}")
        except Exception:
            pass

    def log(
        self,
        dt: float,
        gain_mode: str,
        robot,
        target_array,
        tgt_source: str,
        pose_idx: int,
        difficulty: str,
        cmd: np.ndarray,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        projected_gravity_b: np.ndarray,
        obj_pos: np.ndarray | None = None,
        obj_vel: np.ndarray | None = None,
        ee_pos: np.ndarray | None = None,
    ) -> None:
        """target_array is 7-DOF in policy arm order (sh0,sh1,el0,el1,wr0,wr1,f1x)."""
        self._t += dt
        pos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        vel = np.asarray(robot.get_joint_velocities(), dtype=np.float32)
        try:
            eff = np.asarray(robot.get_measured_joint_efforts(), dtype=np.float32).reshape(-1)
        except Exception:
            eff = np.full(pos.shape, np.nan, dtype=np.float32)
        row = [
            f"{self._t:.4f}", gain_mode, tgt_source,
            int(pose_idx), difficulty,
            f"{float(cmd[0]):.4f}", f"{float(cmd[1]):.4f}", f"{float(cmd[2]):.4f}",
            f"{float(base_lin_vel_b[0]):.5f}", f"{float(base_lin_vel_b[1]):.5f}", f"{float(base_lin_vel_b[2]):.5f}",
            f"{float(base_ang_vel_b[0]):.5f}", f"{float(base_ang_vel_b[1]):.5f}", f"{float(base_ang_vel_b[2]):.5f}",
            f"{float(projected_gravity_b[0]):.5f}", f"{float(projected_gravity_b[1]):.5f}", f"{float(projected_gravity_b[2]):.5f}",
        ]
        op = np.asarray(obj_pos).reshape(-1) if obj_pos is not None else np.full(3, np.nan)
        ov = np.asarray(obj_vel).reshape(-1) if obj_vel is not None else np.full(3, np.nan)
        ep = np.asarray(ee_pos).reshape(-1) if ee_pos is not None else np.full(3, np.nan)
        row += [
            f"{op[0]:.5f}", f"{op[1]:.5f}", f"{op[2]:.5f}",
            f"{ov[0]:.5f}", f"{ov[1]:.5f}", f"{ov[2]:.5f}",
            f"{ep[0]:.5f}", f"{ep[1]:.5f}", f"{ep[2]:.5f}",
        ]
        for ji, pi in zip(self._joint_indices, self._policy_idx):
            tgt = float(target_array[pi]) if target_array is not None else float("nan")
            row += [f"{tgt:.5f}", f"{pos[ji]:.5f}", f"{vel[ji]:.5f}", f"{eff[ji]:.5f}"]
        self._writer.writerow(row)
        self._fh.flush()  # stream to disk each step so a sim crash mid-lift loses nothing

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def _import_spot_from_urdf(urdf_path: str, prim_path: str) -> str:
    """Import the Spot URDF using the Isaac Sim URDF importer."""
    logger.info("Importing Spot URDF: %s", urdf_path)
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.default_drive_strength = 60.0
    import_config.default_position_drive_damping = 1.5
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    if _URDF_IMPORTER_FLAVOR == "isaacsim":
        dest_path = str(Path("/tmp") / f"{Path(urdf_path).stem}_imported.usd")
        result, _ = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            dest_path=dest_path,
        )
        if not result:
            raise RuntimeError(f"URDFParseAndImportFile failed for {urdf_path}")
        add_reference_to_stage(dest_path, prim_path)
        return prim_path

    urdf_dir = os.path.dirname(urdf_path)
    urdf_file = os.path.basename(urdf_path)
    urdf_interface = _urdf.acquire_urdf_interface()
    imported_robot = urdf_interface.parse_urdf(urdf_dir, urdf_file, import_config)
    return urdf_interface.import_robot(urdf_dir, urdf_file, imported_robot, import_config, prim_path)


def _start_keyboard_listener(callbacks: dict) -> None:
    """Read single keypresses from stdin (raw mode) in a daemon thread."""

    def _loop():
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.buffer.read(1)
                if ch == b"\x03":
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                if ch == b"\x1b":
                    rest = sys.stdin.buffer.read(2)
                    seq = {b"[A": "UP", b"[B": "DOWN", b"[C": "RIGHT", b"[D": "LEFT"}.get(rest)
                    if seq and seq in callbacks:
                        callbacks[seq]()
                    continue
                key = ch.decode("utf-8", errors="ignore").upper()
                if key in callbacks:
                    callbacks[key]()
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    threading.Thread(target=_loop, daemon=True).start()


class SpotRunner:
    def __init__(self, physics_dt, render_dt, obs_mode: str = "loco", policy_path: str | None = None,
                 log_csv: str | None = None, gain_switch_s: float = 0.0,
                 grasp_object: str = "SM_BottlePlasticB_01", zed_operator: bool = False,
                 wrist_demo: bool = False, arm_gains: str | None = None) -> None:
        self._zed_operator = bool(zed_operator)
        self._wrist_demo = bool(wrist_demo)
        self._arm_gains_preset = arm_gains
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._phase2_config = load_spot_loco_phase2_config()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Custom warehouse: simple_warehouse + real table + clutter objects
        # pre-arranged on top (drill, jar, etc.)
        prim = define_prim("/World/Warehouse", "Xform")
        prim.GetReferences().AddReference(WAREHOUSE_USD)

        base_dir = Path(__file__).resolve().parent.parent
        policy_params_path = os.path.join(base_dir, "policies/spot_arm/params", "env.yaml")
        urdf_path = "/workspace/spot-locomanipulation/source/spot_loco/spot_loco/assets/spot/spot_with_arm.urdf"
        _import_spot_from_urdf(urdf_path=urdf_path, prim_path="/World/Spot")

        if obs_mode == "loco":
            policy_pt = policy_path or (
                "/workspace/spot-locomanipulation/logs/rsl_rl/spot_locomanipulation"
                "/2026-05-31_16-22-52/exported/policy.pt"
            )
        else:
            policy_pt = str(Path(base_dir) / "policies/spot_arm/models/spot_arm_policy.pt")
        spawn_pos = np.array([-2.5, 0.0, 0.7])
        spawn_quat = None
        if self._zed_operator:
            rs = _load_zed_operator_poses().get("robot_spawn")
            if rs:
                import math
                saved = np.array(rs.get("pos", spawn_pos), dtype=float)
                # Restore only x/y (+ yaw below) and keep the known-good drop height:
                # replaying the raw live pose (standing height + mid-stance orientation)
                # spawns the robot intersecting the ground and physics launches it.
                spawn_pos = np.array([saved[0], saved[1], 0.7])
                if rs.get("quat"):
                    w, x, y, z = [float(v) for v in rs["quat"]]
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    spawn_quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0,
                                           math.sin(yaw / 2.0)])
                print(f"[ZED-operator] robot spawn loaded: xy={saved[:2].tolist()} "
                      f"z=0.7 yaw-only", flush=True)
        self._spot = SpotLocoPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=None,
            policy_path=policy_pt,
            policy_params_path=policy_params_path,
            position=spawn_pos,
            orientation=spawn_quat,
            physics_dt=physics_dt,
            phase2_config=self._phase2_config,
            obs_mode=obs_mode,
        )
        self._trained_gains_applied = False
        self._arm_poses = load_csv_poses(
            self._phase2_config["csv_path"],
            self._phase2_config["policy_arm_order"],
            self._phase2_config["csv_joint_cols"],
        )
        self._active_difficulty = "easy"
        self._arm_sequence = list(self._phase2_config["difficulty_groups"][self._active_difficulty])
        self._pose_idx = 0
        self._arm_routine_idx = -1
        self._arm_ticks = 0
        self._policy_hz = int(round(1.0 / (physics_dt * self._spot._decimation)))
        self._pose_hold_ticks = int(self._phase2_config["arm_tracking"]["pose_hold_s"] * self._policy_hz)
        self._default_hold_ticks = int(self._phase2_config["arm_tracking"]["default_hold_s"] * self._policy_hz)

        self._cameras = create_spot_cameras("/World/Spot", ALL_CAMERAS)
        self._ros_bridge = None
        self._arm_sub = SpotArmCommandSubscriber(self._spot)

        # --- arm gain experiment ---
        self._physics_dt = physics_dt
        self._gain_switch_s = float(gain_switch_s)
        self._gain_sweep = ARM_GAIN_SWEEP
        self._gain_idx = 0  # current index into the sweep
        self._gain_last_switch_t = 0.0
        self._sim_t = 0.0
        self._arm_logger = ArmDataLogger(log_csv, ARM_JOINTS) if log_csv else None
        self._arm_logger_ready = False
        # Physics-backed handles for the grasped object + end-effector, resolved
        # on the first physics step (USD xforms are stale under Fabric, so these
        # must read from the physics sim, not the stage).
        # Which object to grasp/log/stabilize. Accept either a full prim path or a
        # short name resolved under the warehouse stage, so it's no longer hardcoded.
        self._grasp_obj_path = (
            grasp_object if grasp_object.startswith("/")
            else f"/World/Warehouse/{grasp_object}"
        )
        self._ee_link_name = "arm_link_wr1"  # hand/palm; stable (the finger moves)
        self._grasp_obj = None
        self._ee_link_idx = None  # link index into the articulation (read-only, non-invasive)

        self._vx_sens = 1.5
        self._vy_sens = 0.8
        self._wz_sens = 1.5
        self._base_command = np.zeros(3)
        self.needs_reset = False
        self.first_step = True
        self._bridge_delay = 0

        # Object-only reset: snapshot each warehouse rigid body's spawn pose so a
        # keypress can restore the clutter without resetting Spot.
        self._object_reset_requested = False
        self._object_handles = {}        # prim path -> SingleRigidPrim
        self._object_initial_state = {}  # prim path -> (pos[3], quat_wxyz[4])

        # ZED-operator scene must exist BEFORE world.reset(): the rig's IsaacImuSensor
        # (which gates the ZED streamer exec chain) only registers with the physics
        # sensor interface at sim init; created after reset it never produces readings
        # ("no valid sensor reading") and the stream never starts. v4 worked because the
        # whole rig lived in the stage before pressing Play.
        if self._zed_operator:
            self._setup_zed_operator()

    def _set_vel(self, vx: float, vy: float, wz: float) -> None:
        self._base_command = np.array([vx, vy, wz])
        print(f"[Vel] cmd=[{vx:.1f}, {vy:.1f}, {wz:.1f}]")

    def _set_arm_pose_from_sequence(self, idx_in_sequence: int) -> None:
        pose_idx = self._arm_sequence[idx_in_sequence % len(self._arm_sequence)]
        self._pose_idx = pose_idx
        self._spot.set_arm_goal(self._arm_poses[pose_idx])
        print(
            f"[ArmPose] -> pose {pose_idx:3d} "
            f"({self._active_difficulty}, {idx_in_sequence % len(self._arm_sequence) + 1}/{len(self._arm_sequence)})"
        )

    def _advance_arm_routine(self) -> None:
        self._arm_routine_idx = (self._arm_routine_idx + 1) % len(self._arm_sequence)
        self._set_arm_pose_from_sequence(self._arm_routine_idx)
        self._arm_ticks = 0

    def _set_default_arm(self) -> None:
        self._arm_routine_idx = -1
        self._arm_ticks = 0
        self._spot.set_default_arm_pose()
        print("[ArmPose] -> default (stow)")

    def _scale_velocity(self, factor: float) -> None:
        self._vx_sens = round(self._vx_sens * factor, 3)
        self._vy_sens = round(self._vy_sens * factor, 3)
        self._wz_sens = round(self._wz_sens * factor, 3)
        print(f"[Vel] sens -> vx={self._vx_sens:.2f}  vy={self._vy_sens:.2f}  wz={self._wz_sens:.2f}")

    def _switch_difficulty(self, name: str) -> None:
        self._active_difficulty = name
        self._arm_sequence = list(self._phase2_config["difficulty_groups"][name])
        self._arm_routine_idx = -1
        self._arm_ticks = 0
        print(f"[Difficulty] -> {name.upper()} ({len(self._arm_sequence)} poses)")

    def setup(self) -> None:
        callbacks = {
            "UP": lambda: self._set_vel(self._vx_sens, 0.0, 0.0),
            "DOWN": lambda: self._set_vel(-self._vx_sens, 0.0, 0.0),
            "LEFT": lambda: self._set_vel(0.0, self._vy_sens, 0.0),
            "RIGHT": lambda: self._set_vel(0.0, -self._vy_sens, 0.0),
            "Z": lambda: self._set_vel(0.0, 0.0, self._wz_sens),
            "X": lambda: self._set_vel(0.0, 0.0, -self._wz_sens),
            "L": lambda: self._set_vel(0.0, 0.0, 0.0),
            "N": lambda: self._set_arm_pose_from_sequence(
                self._arm_sequence.index(self._pose_idx) + 1 if self._pose_idx in self._arm_sequence else 0
            ),
            "P": lambda: self._set_arm_pose_from_sequence(
                self._arm_sequence.index(self._pose_idx) - 1 if self._pose_idx in self._arm_sequence else 0
            ),
            "0": self._set_default_arm,
            "+": lambda: self._scale_velocity(1.2),
            "-": lambda: self._scale_velocity(1 / 1.2),
            "E": lambda: self._switch_difficulty("easy"),
            "M": lambda: self._switch_difficulty("medium"),
            "H": lambda: self._switch_difficulty("hard"),
            "R": self._request_object_reset,
            "T": self._request_full_reset,
        }
        if self._zed_operator:
            callbacks["G"] = self._request_pose_save
        _start_keyboard_listener(callbacks)
        print("[Keyboard] terminal raw-stdin listener started.")
        print("  Arrow Up/Down  -> forward / backward")
        print("  Arrow Left/Right -> strafe left / right")
        print("  Z / X          -> yaw left / right")
        print("  L              -> stop (zero velocity)")
        print("  N / P          -> next / previous arm pose")
        print("  0              -> arm to default stow")
        print("  + / -          -> scale all velocity sens x1.2 / x0.83")
        print("  E / M / H      -> easy / medium / hard arm sets")
        print("  R              -> reset all warehouse objects (Spot untouched)")
        print("  T              -> full reset (Spot to spawn + objects + re-init)")
        print("  Ctrl-C         -> quit")
        if self._zed_operator:
            print("  G              -> save operator/ZED/robot framing to disk")

        self._world.add_physics_callback("spot_forward", callback_fn=self.on_physics_step)


    def _setup_zed_operator(self) -> None:
        """Spawn the operator character + a virtual ZED that streams to the ZED SDK (IPC),
        for the teleop-loop demo/GIF. Gated behind --zed-operator so normal runs are
        untouched. The operator (anim.usd) has a self-contained animation graph with a
        TwoBoneIK on its arm that follows /World/Operator/wrist_target; drag that target by
        hand while recording. The zed wrapper container receives the stream and
        wrist_detector maps the operator's wrist motion onto the robot arm. The OPERATOR_*/
        ZED_* poses are rough defaults; aim/reposition in the GUI."""
        import omni.usd
        import omni.kit.app
        import omni.graph.core as og
        from pxr import Sdf, Usd, UsdGeom, UsdSkel, Gf

        stage = omni.usd.get_context().get_stage()

        # Extensions (anim-graph runtime + ZED streamer) are enabled at module load, before
        # the stage exists (see the --zed-operator block near the top of the file).

        def _place(prim_path, pos, rot_zyx):
            # Set a single double-precision transform (matrix) op. Referenced assets
            # (e.g. ZED_X.usdc) ship their own translate/rotateZYX/scale ops in double
            # precision; re-adding typed ops would raise a precision/typeName clash, so we
            # clear the op order and write one xformOp:transform instead.
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
            xf.ClearXformOpOrder()
            rx, ry, rz = rot_zyx
            rot = (Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
                   * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                   * Gf.Rotation(Gf.Vec3d(1, 0, 0), rx))
            xf.AddTransformOp().Set(Gf.Matrix4d(rot, Gf.Vec3d(*pos)))

        # Operator character (self-contained anim graph + arm IK -> wrist_target).
        add_reference_to_stage(ANIM_USD, "/World/Operator")
        _place("/World/Operator", OPERATOR_POS, OPERATOR_ROT_ZYX)

        _saved = _load_zed_operator_poses()

        def _apply_saved_matrix(prim_path, key):
            m = _saved.get(key)
            if not m:
                return
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*[float(v) for v in m]))
            print(f"[ZED-operator] {prim_path} pose loaded from saved framing", flush=True)

        _apply_saved_matrix("/World/Operator", "operator")
        # Diagnostic: dump what composed under /World/Operator (typed prims only). The
        # SkelRoot search below depends on the character's S3 payload having resolved.
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Operator")):
            path = str(prim.GetPath())
            if "ActionGraph_01/" in path or "AnimationGraph_01/" in path:
                continue
            print(f"[ZED-operator] prim {path} [{prim.GetTypeName()}]", flush=True)

        # anim.usd stores the character, the AnimationGraph (idle clip + TwoBoneIK), and
        # the ActionGraph feeding wrist_target into it, but NOT the skeleton<->graph
        # binding: AnimationGraphAPI was applied live in the authoring GUI session and is
        # not saved in the asset. Without it the character stays in T-pose and the graph
        # nodes log "Invalid character". Recreate the binding on every SkelRoot found.
        import omni.kit.commands
        skel_roots = [
            prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Operator"))
            if prim.IsA(UsdSkel.Root)
        ]
        anim_graph_prim = stage.GetPrimAtPath("/World/Operator/AnimationGraph_01")
        if not skel_roots or not anim_graph_prim:
            print(f"[ZED-operator] SkelRoots={[str(p.GetPath()) for p in skel_roots]} "
                  f"AnimationGraph={bool(anim_graph_prim)} — binding NOT applied, "
                  f"character will stay in T-pose", flush=True)
        else:
            print(f"[ZED-operator] binding AnimationGraph to SkelRoot(s): "
                  f"{[str(p.GetPath()) for p in skel_roots]}", flush=True)
            try:
                omni.kit.commands.execute(
                    "ApplyAnimationGraphAPICommand",
                    paths=[p.GetPath() for p in skel_roots],
                    animation_graph_path=anim_graph_prim.GetPath(),
                )
            except Exception as exc:
                print(f"[ZED-operator] ApplyAnimationGraphAPICommand failed ({exc}); "
                      f"applying schema manually", flush=True)
                for p in skel_roots:
                    p.AddAppliedSchema("AnimationGraphAPI")
                    p.CreateRelationship("animationGraph").SetTargets(
                        [anim_graph_prim.GetPath()])
        simulation_app.update()
        # anim.usd's ActionGraph->AnimationGraph wiring is broken as-saved (authored
        # live in the GUI, never persisted): the write node's graph rel targets the
        # nonexistent /World/AnimationGraph, its variableName is empty, and the
        # wrist_position variable is not declared on the graph prim. Repair all three,
        # otherwise wrist_target never drives the arm's TwoBoneIK.
        wag = stage.GetPrimAtPath(
            "/World/Operator/ActionGraph_01/write_animation_graph_variable")
        if wag and wag.IsValid() and anim_graph_prim:
            wag.GetRelationship("inputs:graph").SetTargets(
                [Sdf.Path("/World/Operator/AnimationGraph_01")])
            vn = wag.GetAttribute("inputs:variableName")
            # the input is CONNECTED to a read_variable node (graph var "varName",
            # never authored) and a connected input ignores the local value: sever it.
            vn.ClearConnections()
            vn.Set("wrist_position")
            vattr = anim_graph_prim.GetAttribute("anim:graph:variable:wrist_position")
            if not vattr or not vattr.IsValid():
                anim_graph_prim.CreateAttribute(
                    "anim:graph:variable:wrist_position",
                    Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.0, 0.0, 0.0))
            print("[ZED-operator] repaired wrist ActionGraph wiring "
                  "(graph rel + variableName + variable decl)", flush=True)

        # Scripted wrist demo: capture the wrist_target's base local translate so
        # on_physics_step can oscillate it (up-down) without a human dragging it.
        self._wrist_demo_path = "/World/Operator/wrist_target"
        wt = stage.GetPrimAtPath(self._wrist_demo_path)
        self._wrist_demo_base = None
        if wt and wt.IsValid():
            for op in UsdGeom.Xformable(wt).GetOrderedXformOps():
                if "translate" in op.GetOpName():
                    self._wrist_demo_base = Gf.Vec3d(op.Get())
                    self._wrist_demo_op = op
                    break
        if self._wrist_demo and self._wrist_demo_base is None:
            print("[ZED-operator] wrist demo requested but wrist_target translate op "
                  "not found", flush=True)
        if self._wrist_demo and wt and wt.IsValid():
            # Demo drives the anim-graph variable DIRECTLY (set_variable), so disable
            # the per-tick ActionGraph write that would overwrite it with the static
            # wrist_target pose.
            # ActionGraph stays ACTIVE: it is the per-frame variable writer; we move
            # wrist_target with TransformPrimSRT (the exact mechanism GUI drags use).
            m = UsdGeom.Xformable(wt).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            self._wrist_demo_world = Gf.Vec3d(m.ExtractTranslation())
            # Re-base the orbit NEAR the character: at the saved wrist_target distance
            # the +-0.3 m z sweep only tilts the fully-extended arm ~10 deg (invisible).
            # 0.45 m out from the skeleton, same direction, makes the sweep produce
            # large elbow/shoulder motion like a real drag.
            sk = skel_roots[-1] if skel_roots else None
            if sk and sk.IsValid():
                skw = UsdGeom.Xformable(sk).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()).ExtractTranslation()
                d = Gf.Vec3d(self._wrist_demo_world[0] - skw[0],
                             self._wrist_demo_world[1] - skw[1], 0.0)
                if d.GetLength() > 1e-3:
                    d = d / d.GetLength()
                    self._wrist_demo_world = Gf.Vec3d(
                        skw[0] + 0.45 * d[0], skw[1] + 0.45 * d[1], 1.05)
            print(f"[ZED-operator] wrist demo: direct variable drive, world base="
                  f"{tuple(round(v,3) for v in self._wrist_demo_world)}", flush=True)

        # Arm the one-shot runtime probe (see on_physics_step): logs whether the
        # anim-graph runtime actually created a Character from the binding.
        self._zed_probe_skel = str(skel_roots[-1].GetPath()) if skel_roots else None
        self._zed_probe_done = False
        if skel_roots:
            sr = skel_roots[-1]
            print(f"[ZED-operator] post-binding schemas on {sr.GetPath()}: "
                  f"{sr.GetAppliedSchemas()}", flush=True)

        # Virtual ZED X, posed to frame the operator (and the arm if it fits).
        add_reference_to_stage(ZED_X_USD, "/World/ZED_X")
        _place("/World/ZED_X", ZED_POS, ZED_ROT_ZYX)
        # ZED_X.usdc ships an ENABLED rigid body with gravity on, so as a free body it
        # falls. Disable ONLY gravity (v4 does the same): the rigid body must STAY enabled
        # because the rig's Imu_Sensor reads from it, and the extension's exec chain that
        # triggers the ZED streamer node runs through IsaacReadIMU. rigidBodyEnabled=False
        # kills the IMU ("no valid sensor reading") and the stream never starts.
        zed_prim = stage.GetPrimAtPath("/World/ZED_X")
        zed_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
        _apply_saved_matrix("/World/ZED_X", "zed_x")
        self._save_poses_requested = False

        # Streaming graph: OnPlaybackTick -> ZED_Camera helper (IPC to 127.0.0.1).
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": "/World/ZED_ActionGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("ZEDCamera", "sl.sensor.camera.ZED_Camera"),
                ],
                keys.SET_VALUES: [
                    ("ZEDCamera.inputs:cameraModel", "ZED_X"),
                    ("ZEDCamera.inputs:resolution", "HD1080"),
                    # NETWORK (H264 over the port), matching the zed wrapper's sim_mode
                    # receiver. IPC (shared-mem) does NOT feed sim_mode -> the receiver
                    # gets "Failed to retrieve camera settings from sender".
                    ("ZEDCamera.inputs:transportLayerMode", "NETWORK"),
                    ("ZEDCamera.inputs:streamingPort", ZED_STREAM_PORT),
                    ("ZEDCamera.inputs:fps", 30),
                    # Default 10 Mbps smears moving regions into macroblock ghosts at
                    # 1080p (identical corruption at publisher and viewer = encoder-side,
                    # not DDS). Loopback bandwidth is free: give the encoder headroom.
                    ("ZEDCamera.inputs:bitrate", 40000),
                    ("ZEDCamera.inputs:chunkSize", 16084),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "ZEDCamera.inputs:execIn"),
                ],
            },
        )
        # cameraPrim is a target (relationship) input; set it on the USD directly.
        node = stage.GetPrimAtPath("/World/ZED_ActionGraph/ZEDCamera")
        rel = node.GetRelationship("inputs:cameraPrim") if node else None
        if rel:
            rel.SetTargets([Sdf.Path("/World/ZED_X")])

        logger.info("[ZED-operator] operator + virtual ZED ready (IPC port %d). Drag "
                    "/World/Operator/wrist_target to move the arm; run the zed container "
                    "+ wrist_detector to drive the robot.", ZED_STREAM_PORT)

    def _request_pose_save(self) -> None:
        # Keyboard thread just flags; the write happens on the physics thread.
        self._save_poses_requested = True
        print("[ZED-operator] pose save requested (writing on next physics step)")

    def _save_zed_operator_poses(self) -> None:
        """Snapshot operator/ZED local transforms + robot base pose to
        ZED_OPERATOR_POSE_FILE so the next --zed-operator launch restores this framing."""
        import json
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()

        def _local_matrix(prim_path):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            m = UsdGeom.Xformable(prim).GetLocalTransformation()
            return [float(m.GetRow(r)[c]) for r in range(4) for c in range(4)]

        data = {"operator": _local_matrix("/World/Operator"),
                "zed_x": _local_matrix("/World/ZED_X")}
        try:
            pos, quat = self._spot.robot.get_world_pose()
            data["robot_spawn"] = {
                "pos": [float(v) for v in np.asarray(pos).reshape(-1)[:3]],
                "quat": [float(v) for v in np.asarray(quat).reshape(-1)[:4]],
            }
        except Exception as exc:
            print(f"[ZED-operator] could not read robot pose: {exc}", flush=True)
        try:
            with open(ZED_OPERATOR_POSE_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"[ZED-operator] framing saved to {ZED_OPERATOR_POSE_FILE}", flush=True)
        except Exception as exc:
            print(f"[ZED-operator] save failed: {exc}", flush=True)

    def _request_object_reset(self) -> None:
        """Flag an object-only reset; the actual reset runs in the physics step
        (setting poses from the keyboard thread is not physics-safe)."""
        self._object_reset_requested = True
        print("[ObjReset] requested — will restore warehouse objects on next step")

    def _request_full_reset(self) -> None:
        """Flag a full world reset: Spot returns to its spawn pose and the whole
        scene re-initializes (same path used when the sim is stopped/restarted).
        This resets Spot AND the objects, and re-creates the ROS bridge/cameras.
        The flag is consumed in on_physics_step (reset is not thread-safe here)."""
        self.needs_reset = True
        print("[FullReset] requested — Spot to spawn + scene re-init on next step")

    def _capture_object_initial_state(self) -> None:
        """Snapshot the spawn pose of every dynamic rigid body under the
        warehouse so they can later be restored without touching Spot."""
        try:
            from pxr import UsdPhysics
            import omni.usd
            from isaacsim.core.prims import SingleRigidPrim
        except Exception as exc:
            print(f"[ObjReset] import failed, object reset disabled: {exc}")
            return

        stage = omni.usd.get_context().get_stage()
        paths = [
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Warehouse")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        for i, path in enumerate(paths):
            try:
                rp = SingleRigidPrim(prim_path=path, name=f"obj_reset_{i}")
                try:
                    rp.initialize()
                except Exception:
                    pass  # already initialized by the world reset
                pos, quat = rp.get_world_pose()
                self._object_handles[path] = rp
                self._object_initial_state[path] = (np.asarray(pos), np.asarray(quat))
            except Exception as exc:
                print(f"[ObjReset] could not capture {path}: {exc}")
        print(f"[ObjReset] captured spawn pose of {len(self._object_handles)} object(s)")

    def _reset_objects(self) -> None:
        """Restore every captured object to its spawn pose with zero velocity."""
        n = 0
        for path, rp in self._object_handles.items():
            pos, quat = self._object_initial_state[path]
            try:
                rp.set_world_pose(position=pos, orientation=quat)
                rp.set_linear_velocity(np.zeros(3))
                rp.set_angular_velocity(np.zeros(3))
                n += 1
            except Exception as exc:
                print(f"[ObjReset] reset failed on {path}: {exc}")
        print(f"[ObjReset] restored {n} object(s) to spawn pose")

    def _init_grasp_handles(self) -> float:
        """Bind the grasped object as a rigid prim and resolve the EE link index.

        The object is a standalone rigid body, so a SingleRigidPrim handle is
        safe. The EE link is part of the robot articulation — we must NOT wrap it
        as its own rigid body (that splits it from the articulation and detaches
        the finger). We only record its link index and READ its pose from the
        articulation's existing physics view in _read_grasp_state.

        Returns the object mass (kg), or NaN if unavailable. Best-effort.
        """
        obj_mass = float("nan")
        try:
            from isaacsim.core.prims import SingleRigidPrim
            self._grasp_obj = SingleRigidPrim(prim_path=self._grasp_obj_path, name="grasp_obj")
            try:
                self._grasp_obj.initialize()
            except Exception:
                pass  # already initialized by the world reset
            obj_mass = float(self._grasp_obj.get_mass())
            print(f"[GraspLog] object {self._grasp_obj_path} mass={obj_mass:.3f} kg")
        except Exception as exc:
            print(f"[GraspLog] could not bind object {self._grasp_obj_path}: {exc}")
            self._grasp_obj = None

        # EE link index only — never a separate rigid body.
        try:
            self._ee_link_idx = self._spot.robot._articulation_view.get_link_index(self._ee_link_name)
            print(f"[GraspLog] EE link '{self._ee_link_name}' index={self._ee_link_idx} (read-only)")
        except Exception as exc:
            print(f"[GraspLog] could not resolve EE link '{self._ee_link_name}': {exc}")
            self._ee_link_idx = None
        return obj_mass

    def _stabilize_grasp_physics(self) -> None:
        """Make the squeeze grasp stable: high friction on the object + gripper
        pads, no restitution, and more solver iterations so the pressed object
        doesn't penetrate/jitter/slip. All best-effort and non-invasive (material
        binding + solver-count properties, never a new rigid body).
        """
        try:
            import omni.usd
            from pxr import UsdPhysics, PhysxSchema
            from isaacsim.core.api.materials import PhysicsMaterial
            from isaacsim.core.prims import SingleGeometryPrim
        except Exception as exc:
            print(f"[GraspPhys] import failed, skipping stabilization: {exc}")
            return

        # 1) high-friction, zero-restitution material
        mat = None
        try:
            mat = PhysicsMaterial(
                prim_path="/World/Physics/grasp_high_friction",
                name="grasp_high_friction",
                static_friction=1.4, dynamic_friction=1.2, restitution=0.0,
            )
        except Exception as exc:
            print(f"[GraspPhys] material create failed: {exc}")

        stage = omni.usd.get_context().get_stage()

        # Collect every dynamic rigid body under the warehouse — these are the
        # grasp candidates. Static colliders (floor/walls/table) carry no
        # RigidBodyAPI so they're excluded automatically; the robot lives under
        # /World/Spot and is excluded by the path prefix. This makes the grasp
        # physics work for ANY object you reach for, not just one hardcoded prim.
        grasp_objects = [
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Warehouse")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        print(f"[GraspPhys] {len(grasp_objects)} dynamic rigid body(ies) under /World/Warehouse")

        # 2) bind the friction material to every grasp candidate's collider
        if mat is not None:
            applied = 0
            for path in grasp_objects:
                try:
                    SingleGeometryPrim(prim_path=path).apply_physics_material(mat)
                    applied += 1
                except Exception as exc:
                    print(f"[GraspPhys] material apply failed on {path}: {exc}")
            print(f"[GraspPhys] friction material on {applied}/{len(grasp_objects)} object(s)")

            # 3) bind it to the gripper pads (finger + fixed jaw colliders)
            pads = 0
            for prim in stage.Traverse():
                path = prim.GetPath().pathString
                low = path.lower()
                if prim.HasAPI(UsdPhysics.CollisionAPI) and ("fngr" in low or "jaw" in low):
                    try:
                        SingleGeometryPrim(prim_path=path).apply_physics_material(mat)
                        pads += 1
                    except Exception:
                        pass
            print(f"[GraspPhys] friction material on {pads} gripper pad collider(s)")

        # 4) more solver iterations on the robot articulation (resolves the squeeze)
        try:
            av = self._spot.robot._articulation_view
            av.set_solver_position_iteration_counts(np.array([32]))
            av.set_solver_velocity_iteration_counts(np.array([4]))
            print("[GraspPhys] robot solver iters -> pos=32 vel=4")
        except Exception as exc:
            print(f"[GraspPhys] robot solver iter set failed: {exc}")

        # 5) more solver iterations on every grasp object too
        objs_done = 0
        for path in grasp_objects:
            try:
                obj_prim = stage.GetPrimAtPath(path)
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(obj_prim)
                rb.CreateSolverPositionIterationCountAttr(32)
                rb.CreateSolverVelocityIterationCountAttr(4)
                objs_done += 1
            except Exception as exc:
                print(f"[GraspPhys] object solver iter set failed on {path}: {exc}")
        print(f"[GraspPhys] object solver iters -> pos=32 vel=4 on {objs_done} body(ies)")

    @staticmethod
    def _to_numpy(x):
        """Convert a warp/torch/np array to numpy without assuming the backend."""
        if x is None:
            return None
        if hasattr(x, "numpy"):
            try:
                return x.numpy()
            except Exception:
                pass
        if hasattr(x, "detach"):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                pass
        return np.asarray(x)

    def _read_grasp_state(self):
        """Return (obj_pos[3], obj_vel[3], ee_pos[3]) world-frame, NaN on failure.

        EE pose is read non-invasively from the articulation's physics view link
        transforms — no extra rigid body is created.
        """
        obj_pos = obj_vel = ee_pos = None
        if self._grasp_obj is not None:
            try:
                obj_pos, _ = self._grasp_obj.get_world_pose()
                obj_vel = self._grasp_obj.get_linear_velocity()
            except Exception:
                obj_pos = obj_vel = None
        if self._ee_link_idx is not None:
            try:
                pv = self._spot.robot._articulation_view._physics_view
                xf = np.asarray(self._to_numpy(pv.get_link_transforms()))  # pos3 + quat4
                # shape is (num_envs, num_links, 7) for a view, or (num_links, 7)
                link_row = xf[0, self._ee_link_idx] if xf.ndim == 3 else xf[self._ee_link_idx]
                ee_pos = link_row[:3]
            except Exception:
                ee_pos = None
        return obj_pos, obj_vel, ee_pos

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self._spot.ensure_joint_ordering_ready()
            if not self._trained_gains_applied:
                apply_trained_gains(self._spot.robot, self._phase2_config)
                self._trained_gains_applied = True
                if self._arm_gains_preset:
                    match = [g for n, g in ARM_GAIN_SWEEP if n == self._arm_gains_preset]
                    if match:
                        apply_arm_gains(self._spot.robot, match[0])
                        print(f"[ArmGains] preset override -> {self._arm_gains_preset}")
                    else:
                        print(f"[ArmGains] unknown preset {self._arm_gains_preset}; "
                              f"valid: {[n for n, _ in ARM_GAIN_SWEEP]}")
            # only override gains when a sweep is requested
            if self._gain_switch_s > 0.0:
                name, gains = self._gain_sweep[self._gain_idx]
                apply_arm_gains(self._spot.robot, gains)
                print(f"[ArmGains] starting sweep at preset[{self._gain_idx}] = {name}")
            else:
                print("[ArmGains] no sweep — using gains from spot_loco_phase2.yaml")
            obj_mass = self._init_grasp_handles()
            self._stabilize_grasp_physics()
            self._capture_object_initial_state()
            if self._arm_logger is not None:
                policy_arm_order = self._phase2_config.get("policy_arm_order", ARM_JOINTS)
                self._arm_logger.resolve_indices(
                    self._spot.robot, policy_arm_order,
                    extra_meta={"object_path": self._grasp_obj_path,
                                "object_mass_kg": obj_mass,
                                "ee_link": self._ee_link_name},
                )
                self._arm_logger_ready = True
                print(f"[ArmLogger] writing CSV: {self._arm_logger._path}")
            self._set_default_arm()
            initialize_cameras(list(self._cameras.values()), enable_depth=True)
            self._arm_sub.start()
            self.first_step = False
            self._bridge_delay = 120  # wait ~120 physics steps (0.6 s) for render products to be ready
        elif self._bridge_delay > 0:
            self._bridge_delay -= 1
            if self._bridge_delay == 0:
                self._ros_bridge = ROSBridgeBuilder(
                    robot_prim_path="/World/Spot",
                    articulation_root_path="/World/Spot/body",
                    cameras=self._cameras,
                )
                if self._ros_bridge.success:
                    logger.info("ROS2 bridge ready — cameras publishing at ~10 Hz")
                else:
                    logger.warning("ROS2 bridge failed: %s", self._ros_bridge.error)
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
            self._bridge_delay = 0
            self._ros_bridge = None
        else:
            if self._object_reset_requested:
                self._reset_objects()
                self._object_reset_requested = False
            if self._spot._policy_counter % self._spot._decimation == 0:
                if self._arm_sub.arm_override:
                    self._arm_ticks = 0
                else:
                    hold_ticks = self._default_hold_ticks if self._arm_routine_idx == -1 else self._pose_hold_ticks
                    if self._arm_ticks >= hold_ticks:
                        self._advance_arm_routine()
                    self._arm_ticks += 1
            self._spot.forward(step_size, self._base_command)
            self._arm_sub.update(step_size)
            if self._arm_sub.arm_override:
                action = self._arm_sub.get_arm_action()
                if action is not None:
                    self._spot.robot.apply_action(action)

            # advance sim time, rotate through gain sweep, log arm state
            self._sim_t += step_size
            # Scripted operator wrist motion (up-down sinusoid) for hands-free demo runs.
            if (self._zed_operator and self._wrist_demo
                    and getattr(self, "_wrist_demo_base", None) is not None
                    and (int(self._sim_t / step_size) % 10 == 0)):
                import math
                A, P = 0.30, 8.0  # amplitude [m], period [s]
                z = self._wrist_demo_base[2] + A * math.sin(2.0 * math.pi * self._sim_t / P)
                # Drive the anim-graph variable directly (same mechanism the GUI /
                # People ext uses). The saved ActionGraph pipe is broken as-authored and
                # is deactivated in demo mode.
                try:
                    import carb
                    import omni.kit.commands
                    from pxr import Gf as _Gf
                    zt = self._wrist_demo_base[2] + A * math.sin(
                        2.0 * math.pi * self._sim_t / P)
                    omni.kit.commands.execute(
                        "TransformPrimSRT",
                        path=self._wrist_demo_path,
                        new_translation=_Gf.Vec3d(float(self._wrist_demo_base[0]),
                                                  float(self._wrist_demo_base[1]),
                                                  float(zt)),
                    )
                    import omni.anim.graph.core as ag
                    if not hasattr(self, "_wrist_demo_char") or self._wrist_demo_char is None:
                        self._wrist_demo_char = ag.get_character(self._zed_probe_skel)
                    c = self._wrist_demo_char
                    if c is not None and getattr(self, "_wrist_demo_world", None) is not None:
                        w = self._wrist_demo_world
                        zw = w[2] + A * math.sin(2.0 * math.pi * self._sim_t / P)
                        c.set_variable("wrist_position",
                                       carb.Float3(float(w[0]), float(w[1]), float(zw)))
                        if not getattr(self, "_wrist_demo_ok", False):
                            self._wrist_demo_ok = True
                            print(f"[ZED-operator] set_variable drive ACTIVE "
                                  f"(base z={w[2]:.3f})", flush=True)
                        self._wrist_demo_n = getattr(self, "_wrist_demo_n", 0) + 1
                        if self._wrist_demo_n % 40 == 0:  # ~every 2s
                            try:
                                rb = c.get_variable("wrist_position")
                            except Exception as _e:
                                rb = f"get_var_err:{_e}"
                            print(f"[WRIST-DRV] t={self._sim_t:.1f} wrote_z={zw:.3f} "
                                  f"readback={rb}", flush=True)
                except Exception as exc:
                    if not getattr(self, "_wrist_demo_err", False):
                        self._wrist_demo_err = True
                        print(f"[ZED-operator] set_variable failed: {exc}", flush=True)
            if self._zed_operator and getattr(self, "_save_poses_requested", False):
                self._save_poses_requested = False
                self._save_zed_operator_poses()
            if self._gain_switch_s > 0.0 and (self._sim_t - self._gain_last_switch_t) >= self._gain_switch_s:
                self._gain_idx = (self._gain_idx + 1) % len(self._gain_sweep)
                name, gains = self._gain_sweep[self._gain_idx]
                apply_arm_gains(self._spot.robot, gains)
                self._gain_last_switch_t = self._sim_t
                print(f"[ArmGains] t={self._sim_t:.2f}s -> preset[{self._gain_idx}] = {name}")
            if self._arm_logger_ready:
                # Prefer the live ROS/curobo commanded target when override is active,
                # otherwise fall back to the policy's arm_goal_policy.
                if self._arm_sub.arm_override and self._arm_sub._arm_position is not None:
                    target_array = self._arm_sub._arm_position
                    tgt_source = "ros"
                else:
                    target_array = getattr(self._spot, "arm_goal_policy", None)
                    tgt_source = "policy"
                preset_name = self._gain_sweep[self._gain_idx][0]
                # Replicate spot_policy._compute_observation's base-frame transform so
                # logged base state matches what the policy actually sees.
                lin_vel_i = self._spot.robot.get_linear_velocity()
                ang_vel_i = self._spot.robot.get_angular_velocity()
                _, q_ib = self._spot.robot.get_world_pose()
                r_bi = quat_to_rot_matrix(q_ib).T
                base_lin_vel_b = r_bi @ lin_vel_i
                base_ang_vel_b = r_bi @ ang_vel_i
                projected_gravity_b = r_bi @ np.array([0.0, 0.0, -1.0])
                obj_pos, obj_vel, ee_pos = self._read_grasp_state()
                self._arm_logger.log(
                    step_size,
                    preset_name,
                    self._spot.robot,
                    target_array,
                    tgt_source,
                    pose_idx=self._pose_idx if self._arm_routine_idx != -1 else -1,
                    difficulty=self._active_difficulty if self._arm_routine_idx != -1 else "default",
                    cmd=self._base_command,
                    base_lin_vel_b=base_lin_vel_b,
                    base_ang_vel_b=base_ang_vel_b,
                    projected_gravity_b=projected_gravity_b,
                    obj_pos=obj_pos,
                    obj_vel=obj_vel,
                    ee_pos=ee_pos,
                )

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-mode", choices=["loco", "arm"], default="loco",
                        help="loco: locomanipulation policy (default); arm: original spot_arm_policy")
    parser.add_argument("--policy", type=str, default=None, help="Path to policy .pt file (loco mode only)")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Path to CSV for per-step arm joint logging (target/pos/vel + gain_mode). "
                             "Pass 'auto' to write to /workspace/outputs/arm_gain_log_<timestamp>.csv")
    parser.add_argument("--gain-switch-s", type=float, default=0.0,
                        help="Toggle arm gains between ORIG and NEW every N seconds (0 = no switching)")
    parser.add_argument("--grasp-object", type=str, default="SM_BottlePlasticB_01",
                        help="Object to grasp/log/stabilize: a short name resolved under "
                             "/World/Warehouse, or a full prim path. The high-friction grasp "
                             "material and per-step obj_* logging bind to this prim.")
    parser.add_argument("--wrist-demo", action="store_true",
                        help="With --zed-operator: script an up-down motion on the "
                             "operator's wrist_target (hands-free demo/recording).")
    parser.add_argument("--arm-gains", type=str, default=None,
                        help="Override arm gains with an ARM_GAIN_SWEEP preset name "
                             "(e.g. sh10_el0-8_w5) after the trained gains are applied.")
    parser.add_argument("--zed-operator", action="store_true",
                        help="Spawn the operator character (anim.usd) + a virtual ZED that "
                             "streams to the ZED SDK over IPC, for the teleop-loop demo/GIF. "
                             "Off by default; normal runs are unaffected.")
    args, _ = parser.parse_known_args()

    log_csv = args.log_csv
    if log_csv == "auto":
        log_csv = f"/workspace/outputs/arm_gain_log_{int(time.time())}.csv"

    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotRunner(
        physics_dt=physics_dt,
        render_dt=render_dt,
        obs_mode=args.obs_mode,
        policy_path=args.policy,
        log_csv=log_csv,
        gain_switch_s=args.gain_switch_s,
        grasp_object=args.grasp_object,
        zed_operator=args.zed_operator,
        wrist_demo=args.wrist_demo,
        arm_gains=args.arm_gains,
    )
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    try:
        runner.run()
    finally:
        if runner._arm_logger is not None:
            runner._arm_logger.close()
    runner._world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
