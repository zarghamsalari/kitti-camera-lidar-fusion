"""Frustum filtering: extract LiDAR points inside 2D bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrustumResult:
    """Result of filtering LiDAR points through a 2D bounding box."""

    points_3d: np.ndarray       # Kx3 filtered LiDAR points (camera frame)
    points_lidar: np.ndarray    # Kx3 filtered LiDAR points (LiDAR frame)
    median_depth: float
    mean_depth: float
    center_3d: np.ndarray       # [x, y, z] approximate 3D centre
    num_points: int
    bbox: np.ndarray            # original [x1, y1, x2, y2, conf, cls]


def points_in_bbox(projected_uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return boolean mask for projected points inside a 2D bounding box.

    Args:
        projected_uv: Mx2 pixel coordinates.
        bbox: [x1, y1, x2, y2, ...] — only first 4 values used.

    Returns:
        Boolean mask of length M.
    """
    x1, y1, x2, y2 = bbox[:4]
    mask = (
        (projected_uv[:, 0] >= x1)
        & (projected_uv[:, 0] <= x2)
        & (projected_uv[:, 1] >= y1)
        & (projected_uv[:, 1] <= y2)
    )
    return mask


def frustum_filter(
    points_lidar: np.ndarray,
    projected_uv: np.ndarray,
    valid_mask: np.ndarray,
    bbox: np.ndarray,
) -> FrustumResult:
    """Filter LiDAR points that project inside a 2D bounding box.

    Args:
        points_lidar: Nx4 original LiDAR points.
        projected_uv: Mx2 projected pixel coords (M = sum of valid_mask).
        valid_mask: N boolean mask from projection (True = in front of camera).
        bbox: [x1, y1, x2, y2, confidence, class_id].

    Returns:
        FrustumResult with filtered points and depth statistics.
    """
    # Map valid-only indices back to full array
    valid_indices = np.where(valid_mask)[0]

    if projected_uv.shape[0] == 0:
        return _empty_result(bbox)

    inside = points_in_bbox(projected_uv, bbox)

    if not inside.any():
        return _empty_result(bbox)

    # Get the original LiDAR points that are inside the box
    selected_indices = valid_indices[inside]
    filtered_lidar = points_lidar[selected_indices, :3]

    # Depth from the projected points (already computed)
    # Re-derive from LiDAR x-coordinate (forward direction in LiDAR frame)
    depths = filtered_lidar[:, 0]  # x is forward in Velodyne frame

    return FrustumResult(
        points_3d=filtered_lidar,
        points_lidar=filtered_lidar,
        median_depth=float(np.median(depths)),
        mean_depth=float(np.mean(depths)),
        center_3d=filtered_lidar.mean(axis=0),
        num_points=int(filtered_lidar.shape[0]),
        bbox=np.asarray(bbox, dtype=np.float64),
    )


def estimate_3d_center(filtered_points: np.ndarray) -> np.ndarray:
    """Estimate 3D centre as the mean of filtered points."""
    if filtered_points.shape[0] == 0:
        return np.zeros(3)
    return filtered_points[:, :3].mean(axis=0)


def estimate_depth_stats(filtered_points: np.ndarray) -> dict[str, float]:
    """Return depth statistics for filtered LiDAR points."""
    if filtered_points.shape[0] == 0:
        return {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    depths = filtered_points[:, 0]  # x is forward in Velodyne
    return {
        "median": float(np.median(depths)),
        "mean": float(np.mean(depths)),
        "min": float(np.min(depths)),
        "max": float(np.max(depths)),
        "count": int(filtered_points.shape[0]),
    }


def compute_3d_extent(filtered_points: np.ndarray) -> dict[str, float]:
    """Compute axis-aligned 3D bounding extent of filtered LiDAR points.

    Returns dict with x/y/z min, max, extent, centre, and total box volume.
    """
    if filtered_points.shape[0] == 0:
        return {
            "x_min": 0.0, "x_max": 0.0, "x_extent": 0.0,
            "y_min": 0.0, "y_max": 0.0, "y_extent": 0.0,
            "z_min": 0.0, "z_max": 0.0, "z_extent": 0.0,
            "center": [0.0, 0.0, 0.0],
            "volume": 0.0,
        }
    pts = filtered_points[:, :3]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extents = maxs - mins
    return {
        "x_min": float(mins[0]), "x_max": float(maxs[0]), "x_extent": float(extents[0]),
        "y_min": float(mins[1]), "y_max": float(maxs[1]), "y_extent": float(extents[1]),
        "z_min": float(mins[2]), "z_max": float(maxs[2]), "z_extent": float(extents[2]),
        "center": pts.mean(axis=0).tolist(),
        "volume": float(extents[0] * extents[1] * extents[2]),
    }


def _empty_result(bbox: np.ndarray) -> FrustumResult:
    return FrustumResult(
        points_3d=np.empty((0, 3)),
        points_lidar=np.empty((0, 3)),
        median_depth=0.0,
        mean_depth=0.0,
        center_3d=np.zeros(3),
        num_points=0,
        bbox=np.asarray(bbox, dtype=np.float64),
    )
