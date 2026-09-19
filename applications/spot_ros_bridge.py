"""
ROS2 OmniGraph Action Graph builder for Spot in Isaac Sim.

Publishes:
  /clock                                    — rosgraph_msgs/Clock
  /joint_states                             — sensor_msgs/JointState
  /odom                                     — nav_msgs/Odometry
  /tf                                       — tf2_msgs/TFMessage (world→body→all links/cameras)
  /camera/{name}/image                 — sensor_msgs/Image  (RGB)
  /depth_registered/{name}/image       — sensor_msgs/Image  (depth)
  /camera/{name}/camera_info           — sensor_msgs/CameraInfo
  /depth_registered/{name}/camera_info — sensor_msgs/CameraInfo

Usage:
    bridge = ROSBridgeBuilder(
        robot_prim_path="/World/Spot",
        cameras=cameras,          # Dict[str, Camera] from create_spot_cameras()
    )
    if bridge.success:
        print("ROS2 bridge ready")

Adapted from:
    spot-sim/scripts/spot_isaacsim/omnigraph/builder.py
"""

from typing import Any, Dict, List, Optional, Tuple


class ROSBridgeBuilder:
    """Builds an Isaac Sim OmniGraph that publishes Spot sensors over ROS2."""

    def __init__(
        self,
        robot_prim_path: str,
        cameras: Dict[str, Any],
        articulation_root_path: Optional[str] = None,
        enable_joint_state: bool = False,
        enable_tf: bool = True,
        enable_odometry: bool = True,
        graph_path: str = "/ActionGraph_ROS",
        camera_step: int = 6,
    ):
        """
        Args:
            robot_prim_path: USD path to the robot reference root (e.g. "/World/Spot").
                Used for TF traversal — should cover all robot links.
            cameras: Dict[name, Camera] from create_spot_cameras()
            articulation_root_path: USD path to the ArticulationRootAPI prim
                (e.g. "/World/Spot/body"). Used for JointState and odometry publishing.
                Defaults to robot_prim_path if not provided.
            enable_joint_state: Publish /joint_states
            enable_tf: Publish /tf for robot + camera frames
            enable_odometry: Publish /odom (nav_msgs/Odometry) + odom→body TF
            graph_path: OmniGraph prim path (recreated if exists)
            camera_step: IsaacSimulationGate step — fires every N render ticks.
                         Default 6 ≈ 10 Hz at 60 Hz render rate.
        """
        self.robot_prim_path = robot_prim_path
        self.articulation_root_path = articulation_root_path or robot_prim_path
        self.cameras = cameras or {}
        self.enable_joint_state = enable_joint_state
        self.enable_tf = enable_tf
        self.enable_odometry = enable_odometry
        self.graph_path = graph_path
        self.camera_step = camera_step

        self._camera_paths = self._extract_camera_paths()
        self._validate_prims()

        self.success, self.error = self._build()

    # -------------------------------------------------------------------------
    # Prim extraction / validation
    # -------------------------------------------------------------------------

    def _extract_camera_paths(self) -> Dict[str, Tuple[str, str]]:
        """Return {name: (prim_path, render_product_path)} for each camera."""
        paths = {}
        for name, camera in self.cameras.items():
            try:
                paths[name] = (camera.prim_path, camera.get_render_product_path())
            except Exception as exc:
                print(f"[ROSBridgeBuilder] WARN: could not get paths for camera '{name}': {exc}")
        return paths

    def _validate_prims(self) -> None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(self.robot_prim_path).IsValid():
            raise ValueError(f"[ROSBridgeBuilder] Robot prim not found: {self.robot_prim_path}")
        for name, (cam_path, _) in self._camera_paths.items():
            if not stage.GetPrimAtPath(cam_path).IsValid():
                print(f"[ROSBridgeBuilder] WARN: camera prim not found: {cam_path} (name={name})")

    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------

    def _build(self) -> Tuple[bool, Optional[str]]:
        try:
            self._ensure_graph_container()

            create_nodes, connect, set_values = [], [], []

            for spec_fn in [self._common_spec, self._joint_state_spec, self._tf_spec, self._odometry_spec]:
                if spec_fn is self._joint_state_spec and not self.enable_joint_state:
                    continue
                if spec_fn is self._tf_spec and not self.enable_tf:
                    continue
                if spec_fn is self._odometry_spec and not self.enable_odometry:
                    continue
                c, conn, v = spec_fn()
                create_nodes.extend(c)
                connect.extend(conn)
                set_values.extend(v)

            for name, (prim_path, render_path) in self._camera_paths.items():
                c, conn, v = self._camera_tf_spec(name=name, prim_path=prim_path)
                create_nodes.extend(c)
                connect.extend(conn)
                set_values.extend(v)

            for name, (_, render_path) in self._camera_paths.items():
                for fn, kwargs in [
                    (self._camera_image_spec, {
                        "name": name,
                        "render_path": render_path,
                        "rgb_topic": f"/camera/{name}/image",
                        "depth_topic": f"/depth_registered/{name}/image",
                        "frame_id": name,
                    }),
                    (self._camera_info_spec, {
                        "name": name,
                        "render_path": render_path,
                        "rgb_info_topic": f"/camera/{name}/camera_info",
                        "depth_info_topic": f"/depth_registered/{name}/camera_info",
                        "frame_id": name,
                    }),
                ]:
                    c, conn, v = fn(**kwargs)
                    create_nodes.extend(c)
                    connect.extend(conn)
                    set_values.extend(v)

            import omni.graph.core as og
            keys = og.Controller.Keys
            og.Controller.edit(
                {"graph_path": self.graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: create_nodes,
                    keys.CONNECT: connect,
                    keys.SET_VALUES: set_values,
                },
            )
            return True, None

        except Exception as exc:
            return False, str(exc)

    def _ensure_graph_container(self) -> None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        existing = stage.GetPrimAtPath(self.graph_path)
        if existing.IsValid():
            stage.RemovePrim(self.graph_path)

    # -------------------------------------------------------------------------
    # Node specification helpers
    # -------------------------------------------------------------------------

    def _common_spec(self) -> Tuple[List, List, List]:
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("PubClock",       "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("CameraGate",     "isaacsim.core.nodes.IsaacSimulationGate"),
        ]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
            ("OnPlaybackTick.outputs:tick", "CameraGate.inputs:execIn"),
        ]
        set_values = [
            ("CameraGate.inputs:step", self.camera_step),
        ]
        return create_nodes, connect, set_values

    def _joint_state_spec(self) -> Tuple[List, List, List]:
        import usdrt.Sdf
        create_nodes = [("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubJointState.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubJointState.inputs:timeStamp"),
        ]
        set_values = [
            ("PubJointState.inputs:targetPrim", [usdrt.Sdf.Path(self.articulation_root_path)]),
            ("PubJointState.inputs:topicName", "/joint_states"),
        ]
        return create_nodes, connect, set_values

    def _tf_spec(self) -> Tuple[List, List, List]:
        import usdrt.Sdf
        create_nodes = [("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree")]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubTF.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
        ]
        target_prims = [usdrt.Sdf.Path(self.articulation_root_path)]
        set_values = [
            ("PubTF.inputs:targetPrims", target_prims),
            ("PubTF.inputs:topicName", "/tf"),
        ]
        return create_nodes, connect, set_values

    def _odometry_spec(self) -> Tuple[List, List, List]:
        import usdrt.Sdf
        create_nodes = [
            ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("PubOdom",     "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ]
        connect = [
            ("OnPlaybackTick.outputs:tick",       "ComputeOdom.inputs:execIn"),
            ("ComputeOdom.outputs:execOut",        "PubOdom.inputs:execIn"),
            ("ComputeOdom.outputs:position",       "PubOdom.inputs:position"),
            ("ComputeOdom.outputs:orientation",    "PubOdom.inputs:orientation"),
            ("ComputeOdom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
            ("ComputeOdom.outputs:angularVelocity","PubOdom.inputs:angularVelocity"),
            ("ReadSimTime.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
        ]
        set_values = [
            ("ComputeOdom.inputs:chassisPrim",  [usdrt.Sdf.Path(self.articulation_root_path)]),
            ("PubOdom.inputs:topicName",        "/odom"),
            ("PubOdom.inputs:odomFrameId",      "odom"),
            ("PubOdom.inputs:chassisFrameId",   "body"),
        ]
        return create_nodes, connect, set_values

    def _camera_tf_spec(self, name: str, prim_path: str) -> Tuple[List, List, List]:
        """Dedicated ROS2PublishTransformTree node for one camera prim.

        Publishes world→<name> using Isaac Sim's own coordinate handling.
        Each camera gets its own node so it never shares targetPrims with the
        articulation root and cannot give any frame two parents.
        """
        import usdrt.Sdf
        node = f"PubTF_{name}"
        create_nodes = [(node, "isaacsim.ros2.bridge.ROS2PublishTransformTree")]
        connect = [
            ("OnPlaybackTick.outputs:tick",        f"{node}.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
        ]
        set_values = [
            (f"{node}.inputs:targetPrims", [usdrt.Sdf.Path(prim_path)]),
            (f"{node}.inputs:topicName",   "/tf"),
        ]
        return create_nodes, connect, set_values

    def _camera_image_spec(
        self,
        name: str,
        render_path: str,
        rgb_topic: str,
        depth_topic: str,
        frame_id: str,
    ) -> Tuple[List, List, List]:
        rgb_node   = f"PubCamera_{name}_rgb"
        depth_node = f"PubCamera_{name}_depth"
        create_nodes = [
            (rgb_node,   "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        connect = [
            ("CameraGate.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            ("CameraGate.outputs:execOut", f"{depth_node}.inputs:execIn"),
        ]
        set_values = [
            (f"{rgb_node}.inputs:frameId",           frame_id),
            (f"{rgb_node}.inputs:topicName",          rgb_topic),
            (f"{rgb_node}.inputs:renderProductPath",  render_path),
            (f"{rgb_node}.inputs:type",               "rgb"),
            (f"{depth_node}.inputs:frameId",          frame_id),
            (f"{depth_node}.inputs:topicName",        depth_topic),
            (f"{depth_node}.inputs:renderProductPath", render_path),
            (f"{depth_node}.inputs:type",             "depth"),
        ]
        return create_nodes, connect, set_values

    def _camera_info_spec(
        self,
        name: str,
        render_path: str,
        rgb_info_topic: str,
        depth_info_topic: str,
        frame_id: str,
    ) -> Tuple[List, List, List]:
        rgb_node   = f"PubCameraInfo_{name}_rgb"
        depth_node = f"PubCameraInfo_{name}_depth"
        create_nodes = [
            (rgb_node,   "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        connect = [
            ("CameraGate.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            ("CameraGate.outputs:execOut", f"{depth_node}.inputs:execIn"),
        ]
        set_values = [
            (f"{rgb_node}.inputs:frameId",           frame_id),
            (f"{rgb_node}.inputs:topicName",          rgb_info_topic),
            (f"{rgb_node}.inputs:renderProductPath",  render_path),
            (f"{depth_node}.inputs:frameId",          frame_id),
            (f"{depth_node}.inputs:topicName",        depth_info_topic),
            (f"{depth_node}.inputs:renderProductPath", render_path),
        ]
        return create_nodes, connect, set_values
