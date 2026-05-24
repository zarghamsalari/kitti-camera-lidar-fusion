"""Parse KITTI calibration files and perform coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KittiCalibration:
    """Parsed KITTI calibration matrices."""

    P2: np.ndarray          # 3x4 — left colour camera projection
    R0_rect: np.ndarray     # 3x3 — rectification rotation
    Tr_velo_to_cam: np.ndarray  # 3x4 — Velodyne to camera 0


def load_calibration(calib_path: str | Path) -> KittiCalibration:
    """Load a KITTI calibration file and return parsed matrices."""
    data: dict[str, np.ndarray] = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in value.split()])

    P2 = data["P2"].reshape(3, 4)
    R0_rect = data["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = data["Tr_velo_to_cam"].reshape(3, 4)

    return KittiCalibration(P2=P2, R0_rect=R0_rect, Tr_velo_to_cam=Tr_velo_to_cam)


def lidar_to_camera(points_lidar: np.ndarray, calib: KittiCalibration) -> np.ndarray:
    """Transform Nx3 LiDAR points to camera coordinates.

    Returns Nx3 array in camera frame (x-right, y-down, z-forward).
    """
    n = points_lidar.shape[0]
    # Homogeneous: Nx4
    ones = np.ones((n, 1), dtype=points_lidar.dtype)
    pts_hom = np.hstack([points_lidar[:, :3], ones])  # Nx4

    # Velodyne → camera 0: (3x4) @ (4xN) → 3xN
    pts_cam0 = (calib.Tr_velo_to_cam @ pts_hom.T)  # 3xN

    # Rectification: (3x3) @ (3xN) → 3xN
    pts_rect = calib.R0_rect @ pts_cam0  # 3xN

    return pts_rect.T  # Nx3


def camera_to_image(points_camera: np.ndarray, calib: KittiCalibration) -> np.ndarray:
    """Project Nx3 camera-frame points onto image plane using P2.

    Returns Nx3 array of [u, v, depth].
    """
    n = points_camera.shape[0]
    ones = np.ones((n, 1), dtype=points_camera.dtype)
    pts_hom = np.hstack([points_camera, ones])  # Nx4

    # P2 (3x4) @ (4xN) → 3xN
    pts_img = (calib.P2 @ pts_hom.T)  # 3xN

    # Normalise by depth
    depth = pts_img[2, :]
    u = pts_img[0, :] / depth
    v = pts_img[1, :] / depth

    return np.stack([u, v, depth], axis=1)  # Nx3


def project_lidar_to_image(
    points_lidar: np.ndarray,
    calib: KittiCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: LiDAR → camera → image.

    Returns:
        uv: Mx2 pixel coordinates of valid points
        depth: M depths
        valid_mask: N boolean mask (True = kept)
    """
    pts_cam = lidar_to_camera(points_lidar, calib)

    # Filter points behind camera (z <= 0)
    valid = pts_cam[:, 2] > 0
    pts_cam_valid = pts_cam[valid]

    if pts_cam_valid.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            valid,
        )

    uvd = camera_to_image(pts_cam_valid, calib)
    uv = uvd[:, :2]
    depth = uvd[:, 2]

    return uv, depth, valid
