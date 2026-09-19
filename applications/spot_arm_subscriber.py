"""
ROS2 arm joint command subscriber for Spot in Isaac Sim.

Subscribes to /arm/joint_command (sensor_msgs/JointState) and drives the arm
joints directly via ArticulationAction at physics rate (200 Hz), decoupled from
the loco policy. Also publishes /joint_states at physics rate.

Position mode: message.position non-empty → set as arm goal immediately.
Velocity mode: message.velocity non-empty, message.position empty/zeros →
               integrate velocities each update() call at 200 Hz.

Joint names must match the policy arm joint names:
    arm_sh0, arm_sh1, arm_el0, arm_el1, arm_wr0, arm_wr1, arm_f1x
Unspecified joints retain their current commanded position.
"""

import threading
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spot_policy import SpotLocoPolicy

ARM_JOINT_NAMES = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]
DEFAULT_TOPIC = "/arm/joint_command"


class SpotArmCommandSubscriber:
    """Receives arm joint commands over ROS2 and drives the arm at physics rate."""

    def __init__(
        self,
        spot: "SpotLocoPolicy",
        topic: str = DEFAULT_TOPIC,
        node_name: str = "spot_arm_command_subscriber",
    ):
        self._spot = spot
        self._topic = topic
        self._node_name = node_name

        self._arm_position: Optional[np.ndarray] = None  # 7-DOF, ARM_JOINT_NAMES order
        self._vel_cmd: Optional[np.ndarray] = None        # 7-DOF velocity
        self.arm_override: bool = False
        self._lock = threading.Lock()
        self._arm_indices_cache: Optional[List[int]] = None

        self._node = None
        self._executor = None
        self._thread = None
        self._started = False
        self._js_pub = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the rclpy node in a daemon thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="SpotRosNode")
        self._thread.start()

    def update(self, step_size: float) -> None:
        """Call every physics step (200 Hz): integrate velocity and publish joint states."""
        with self._lock:
            vel = self._vel_cmd
            if self.arm_override and vel is not None and self._arm_position is not None:
                self._arm_position = self._arm_position + vel * step_size

        self._publish_joint_states()

    def get_arm_action(self):
        """Return ArticulationAction for arm joints, or None if no override active."""
        from isaacsim.core.utils.types import ArticulationAction
        with self._lock:
            pos = self._arm_position.copy() if self._arm_position is not None else None
        if pos is None:
            return None
        return ArticulationAction(joint_positions=pos, joint_indices=self._get_arm_indices())

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_arm_indices(self) -> List[int]:
        """Robot DOF indices for arm joints in ARM_JOINT_NAMES (= policy arm) order."""
        if self._arm_indices_cache is None:
            inv = {v: k for k, v in self._spot.robot_to_policy_idx.items()}
            self._arm_indices_cache = [inv[12 + i] for i in range(7)]
        return self._arm_indices_cache

    def _init_arm_position(self) -> None:
        """Seed _arm_position from current commanded policy state (no-jump on takeover)."""
        current = self._spot.arm_commanded_policy
        self._arm_position = current.copy() if current is not None else np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)

    def _publish_joint_states(self) -> None:
        if self._js_pub is None or self._node is None:
            return
        try:
            from sensor_msgs.msg import JointState
            msg = JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name = list(self._spot.robot.dof_names)
            positions = self._spot.robot.get_joint_positions()
            velocities = self._spot.robot.get_joint_velocities()
            if positions is not None:
                msg.position = positions.tolist()
            if velocities is not None:
                msg.velocity = velocities.tolist()
            self._js_pub.publish(msg)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import JointState

            rclpy.init()
            self._node = rclpy.create_node(self._node_name)
            self._node.create_subscription(JointState, self._topic, self._on_joint_state, 10)
            self._js_pub = self._node.create_publisher(JointState, "/joint_states", 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._node.get_logger().info(f"[SpotArmSub] Listening on {self._topic}")
            self._executor.spin()
        except Exception as exc:
            print(f"[SpotArmSub] ERROR in subscriber thread: {exc}")
        finally:
            if self._node is not None:
                self._node.destroy_node()

    def _on_joint_state(self, msg) -> None:
        names = list(msg.name)
        positions = list(msg.position) if msg.position else []
        velocities = list(msg.velocity) if msg.velocity else []

        has_positions = len(positions) > 0 and any(p != 0.0 for p in positions)
        has_velocities = len(velocities) > 0 and any(v != 0.0 for v in velocities)

        if has_positions:
            self._apply_position_cmd(names, positions)
            with self._lock:
                self._vel_cmd = None
            self.arm_override = True
        elif has_velocities:
            if not self.arm_override:
                self._init_arm_position()
            self._update_velocity_cmd(names, velocities)
            self.arm_override = True
        else:
            # All zeros → release ROS control back to the auto-routine
            with self._lock:
                pos = self._arm_position.copy() if self._arm_position is not None else None
                self._vel_cmd = None
            if pos is not None:
                self._spot.set_arm_goal(pos)
            self.arm_override = False

    def _apply_position_cmd(self, names, positions) -> None:
        if not self.arm_override:
            self._init_arm_position()
        with self._lock:
            new_pos = self._arm_position.copy() if self._arm_position is not None else np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        for name, pos in zip(names, positions):
            if name in ARM_JOINT_NAMES:
                new_pos[ARM_JOINT_NAMES.index(name)] = pos
        with self._lock:
            self._arm_position = new_pos

    def _update_velocity_cmd(self, names, velocities) -> None:
        vel = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        for name, v in zip(names, velocities):
            if name in ARM_JOINT_NAMES:
                vel[ARM_JOINT_NAMES.index(name)] = v
        with self._lock:
            self._vel_cmd = vel
