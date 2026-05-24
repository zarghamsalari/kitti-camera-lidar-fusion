"""Project LiDAR points onto a camera image and render overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lidar_fusion.calibration import KittiCalibration, load_calibration, project_lidar_to_image
from lidar_fusion.pointcloud import filter_by_range, filter_forward_points, load_velodyne_bin


def depth_colour_map(depth: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """Map depth values to RGB colours via a matplotlib colourmap.

    Returns Nx3 uint8 array.
    """
    cmap = plt.cm.get_cmap(cmap_name)
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colours = (cmap(d_norm)[:, :3] * 255).astype(np.uint8)
    return colours


def draw_lidar_on_image(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Draw projected LiDAR points on an image, coloured by depth."""
    img = image.copy()
    h, w = img.shape[:2]

    # Clip to image bounds
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[mask]
    depth = depth[mask]

    colours = depth_colour_map(depth)
    for i in range(len(uv)):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        colour = tuple(int(c) for c in colours[i])
        cv2.circle(img, (u, v), radius, colour, -1)

    return img


def run_projection(
    image_path: str | Path,
    velodyne_path: str | Path,
    calib_path: str | Path,
    out_path: str | Path,
    max_distance: float = 80.0,
) -> np.ndarray:
    """Full projection pipeline: load data, project, draw, save."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    points = load_velodyne_bin(velodyne_path)
    points = filter_forward_points(points)
    points = filter_by_range(points, max_distance)

    calib = load_calibration(calib_path)
    uv, depth, _ = project_lidar_to_image(points, calib)

    result = draw_lidar_on_image(image, uv, depth)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)

    return result
