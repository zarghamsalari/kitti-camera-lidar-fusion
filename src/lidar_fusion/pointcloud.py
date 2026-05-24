"""Load and filter KITTI Velodyne point clouds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_velodyne_bin(path: str | Path) -> np.ndarray:
    """Load a KITTI .bin point cloud file.

    Returns Nx4 float32 array: [x, y, z, intensity].
    """
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return points


def filter_forward_points(points: np.ndarray) -> np.ndarray:
    """Keep only points in front of the sensor (x > 0)."""
    return points[points[:, 0] > 0]


def filter_by_range(points: np.ndarray, max_distance: float = 80.0) -> np.ndarray:
    """Keep points within a Euclidean distance from the sensor."""
    dist = np.linalg.norm(points[:, :3], axis=1)
    return points[dist <= max_distance]


def make_bev_plot(
    points: np.ndarray,
    xlim: tuple[float, float] = (-40, 40),
    ylim: tuple[float, float] = (0, 80),
    point_size: float = 0.3,
) -> plt.Figure:
    """Create a bird's-eye-view scatter plot of a point cloud.

    Uses LiDAR x (lateral) as plot-x and LiDAR y (forward) as plot-y,
    coloured by height (z).
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap="viridis",
        s=point_size,
        edgecolors="none",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Bird's-Eye View")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="height (m)")
    plt.tight_layout()
    return fig
