"""
Spot camera configurations and factory for Isaac Sim.

Provides pre-configured CameraConfig instances for all 6 Spot onboard cameras
(hand + 5 body fisheyes) with transforms baked from the URDF joint chain.

Usage:
    cameras = create_spot_cameras("/World/Spot", ALL_CAMERAS)  # before world.reset()
    # ... world.reset() ...
    initialize_cameras(list(cameras.values()), enable_depth=True)  # after world.reset()

Adapted from:
    spot-sim/scripts/spot_isaacsim/scene/cameras.py
    spot-sim/scripts/spot_isaacsim/spot_config/cameras/rgbd_cameras.py
"""

import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Isaac Sim RGB camera specification."""

    name: str
    prim_path: str                                        # relative to robot prim (e.g. "body/frontleft_cam")
    resolution: Tuple[int, int]                           # (width, height)
    frequency: int = 10                                   # Hz
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, -math.pi / 2, 0.0)  # radians
    focal_length: float = 18.0                            # mm (USD units)
    horizontal_aperture: float = 20.955                   # mm
    clipping_range: Optional[Tuple[float, float]] = (0.01, 100.0)


# ---------------------------------------------------------------------------
# URDF joint chain for Spot body cameras
# Encodes the transform chain from the `body` link to each fisheye mount.
# Source: spot_with_arm.urdf joint definitions.
# ---------------------------------------------------------------------------

_CAMERA_MOUNT_JOINTS = {
    "head": {
        "parent": None,
        "xyz": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    },
    "frontleft": {
        "parent": "head",
        "xyz": [0.41275, 0.03719, 0.02395],
        "rpy": [-2.589351, 1.137527, -3.136510],
    },
    "frontleft_fisheye": {
        "parent": "frontleft",
        "xyz": [0.07825, 0.00035, 0.00200],
        "rpy": [-0.005921, 0.000265, 0.012604],
    },
    "frontright": {
        "parent": "head",
        "xyz": [0.41262, -0.03788, 0.02454],
        "rpy": [2.633190, 1.143761, -3.106119],
    },
    "frontright_fisheye": {
        "parent": "frontright",
        "xyz": [0.07805, 0.00055, 0.00224],
        "rpy": [0.001813, 0.000403, 0.009670],
    },
    "left_fisheye": {
        "parent": None,
        "xyz": [0.0, 0.1, 0.0],
        "rpy": [0.0, 0.0, 1.57],
    },
    "right_fisheye": {
        "parent": None,
        "xyz": [0.0, -0.1, 0.0],
        "rpy": [0.0, 0.0, -1.57],
    },
    "back_fisheye": {
        "parent": None,
        "xyz": [-0.45, 0.0, 0.0],
        "rpy": [0.0, 0.0, 3.14],
    },
}


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat3(rpy)
    T[:3, 3] = xyz
    return T


@functools.lru_cache(maxsize=None)
def _body_T(name):
    """Recursively compose transforms from body frame to mount link `name`."""
    entry = _CAMERA_MOUNT_JOINTS[name]
    T_local = _make_T(entry["xyz"], entry["rpy"])
    if entry["parent"] is None:
        return T_local
    return _body_T(entry["parent"]) @ T_local


def _mat3_to_rpy(R):
    """Extract ZYX Euler angles (roll, pitch, yaw) from rotation matrix."""
    r = np.arctan2(R[2, 1], R[2, 2])
    p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    y = np.arctan2(R[1, 0], R[0, 0])
    return (r, p, y)


def _compute_body_cam_transform(mount_name, local_xyz, local_rpy):
    """
    Compute the body-relative transform for a camera.

    Composes: T_body_mount (from URDF joint chain) @ T_local_offset.
    Returns: (translation_tuple, orientation_rpy_tuple)
    """
    T_body_mount = _body_T(mount_name)
    T_local = _make_T(local_xyz, local_rpy)
    T_final = T_body_mount @ T_local
    translation = tuple(T_final[:3, 3].tolist())
    orientation_rpy = _mat3_to_rpy(T_final[:3, :3])
    return translation, orientation_rpy


# ---------------------------------------------------------------------------
# Pre-configured camera instances
# ---------------------------------------------------------------------------

hand_camera_config = CameraConfig(
    name="hand",
    prim_path="arm_link_wr1/hand",
    resolution=(640, 480),
    translation=(0.15, 0.0, 0.05),
    orientation_rpy=(0.0, 0.0, 0.0),
    focal_length=18.0,
)

_fl_xyz, _fl_rpy = _compute_body_cam_transform("frontleft_fisheye", [0.0, 0.0, 0.01], [0.0, -np.pi / 2, 0.0])
frontleft_camera_config = CameraConfig(
    name="frontleft",
    prim_path="body/frontleft",
    resolution=(640, 480),
    translation=_fl_xyz,
    orientation_rpy=_fl_rpy,
    focal_length=10.8,
)

_fr_xyz, _fr_rpy = _compute_body_cam_transform("frontright_fisheye", [0.0, 0.0, 0.01], [0.0, -np.pi / 2, 0.0])
frontright_camera_config = CameraConfig(
    name="frontright",
    prim_path="body/frontright",
    resolution=(640, 480),
    translation=_fr_xyz,
    orientation_rpy=_fr_rpy,
    focal_length=10.8,
)

_l_xyz, _l_rpy = _compute_body_cam_transform("left_fisheye", [0.05, 0.0, 0.02], [0.0, 0.0, 0.0])
left_camera_config = CameraConfig(
    name="left",
    prim_path="body/left",
    resolution=(640, 480),
    translation=_l_xyz,
    orientation_rpy=_l_rpy,
    focal_length=10.8,
)

_r_xyz, _r_rpy = _compute_body_cam_transform("right_fisheye", [0.05, 0.0, 0.02], [0.0, 0.0, 0.0])
right_camera_config = CameraConfig(
    name="right",
    prim_path="body/right",
    resolution=(640, 480),
    translation=_r_xyz,
    orientation_rpy=_r_rpy,
    focal_length=10.8,
)

_b_xyz, _b_rpy = _compute_body_cam_transform("back_fisheye", [0.06, 0.0, 0.0], [0.0, 0.0, 0.0])
back_camera_config = CameraConfig(
    name="rear",
    prim_path="body/rear",
    resolution=(640, 480),
    translation=_b_xyz,
    orientation_rpy=_b_rpy,
    focal_length=10.8,
)

ALL_CAMERAS: List[CameraConfig] = [
    hand_camera_config,
    frontleft_camera_config,
    frontright_camera_config,
    left_camera_config,
    right_camera_config,
    back_camera_config,
]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_spot_cameras(
    robot_prim_path: str,
    configs: List[CameraConfig],
) -> Dict[str, "Camera"]:
    """
    Create Isaac Sim Camera objects from a list of CameraConfigs.

    Must be called BEFORE world.reset(). Cameras are not yet initialized —
    call initialize_cameras() after world.reset().

    Args:
        robot_prim_path: USD path to the robot articulation root (e.g. "/World/Spot")
        configs: List of CameraConfig instances to create

    Returns:
        Dict mapping camera name to Camera object
    """
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    cameras = {}
    for cfg in configs:
        orientation = rot_utils.euler_angles_to_quats(
            np.array(cfg.orientation_rpy), degrees=False
        )
        camera = Camera(
            prim_path=f"{robot_prim_path}/{cfg.prim_path}",
            name=cfg.name,
            resolution=cfg.resolution,
            frequency=cfg.frequency,
            translation=np.array(cfg.translation),
            orientation=orientation,
        )
        camera.set_focal_length(cfg.focal_length)
        camera.set_horizontal_aperture(cfg.horizontal_aperture)
        if cfg.clipping_range is not None:
            camera.set_clipping_range(*cfg.clipping_range)
        cameras[cfg.name] = camera

    return cameras


def initialize_cameras(cameras: List["Camera"], enable_depth: bool = True) -> None:
    """
    Initialize cameras after world.reset().

    Args:
        cameras: List of Camera objects to initialize
        enable_depth: Whether to enable depth output (distance to image plane)
    """
    for camera in cameras:
        camera.initialize()
        if enable_depth:
            camera.add_distance_to_image_plane_to_frame()
