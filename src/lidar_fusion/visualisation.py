"""Visualisation utilities for LiDAR fusion outputs."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lidar_fusion.frustum import FrustumResult
from lidar_fusion.projection import depth_colour_map


def draw_boxes_with_lidar(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    results: list[FrustumResult],
    valid_mask: np.ndarray,
    points_lidar: np.ndarray,
) -> np.ndarray:
    """Draw 2D detection boxes and filtered LiDAR points on an image."""
    img = image.copy()
    h, w = img.shape[:2]

    # Draw all projected points faintly
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for i in np.where(in_frame)[0]:
        u, v = int(uv[i, 0]), int(uv[i, 1])
        cv2.circle(img, (u, v), 1, (100, 100, 100), -1)

    # Draw each detection box and its filtered LiDAR points
    colours_box = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    for idx, res in enumerate(results):
        colour = colours_box[idx % len(colours_box)]
        x1, y1, x2, y2 = res.bbox[:4].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Label with depth and count
        label = f"d={res.median_depth:.1f}m n={res.num_points}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Highlight filtered points in box colour
        if res.num_points > 0:
            from lidar_fusion.calibration import project_lidar_to_image, load_calibration

            # Re-project filtered points — use the valid_mask indices
            # We already have the bbox-filtered lidar points; find their uv
            inside_mask = _find_inside_indices(uv, res.bbox)
            for i in np.where(inside_mask)[0]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(img, (u, v), 3, colour, -1)

    return img


def _find_inside_indices(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return boolean mask for uv points inside bbox."""
    x1, y1, x2, y2 = bbox[:4]
    return (
        (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    )


def make_frustum_bev(
    points_all: np.ndarray,
    results: list[FrustumResult],
) -> plt.Figure:
    """BEV plot with all points grey and frustum-filtered points highlighted."""
    fig, ax = plt.subplots(figsize=(8, 10))

    # All points in grey
    ax.scatter(
        points_all[:, 0],
        points_all[:, 1],
        c="lightgrey",
        s=0.3,
        edgecolors="none",
        label="all",
    )

    colours = ["green", "blue", "red", "orange", "cyan", "magenta"]
    for idx, res in enumerate(results):
        if res.num_points > 0:
            c = colours[idx % len(colours)]
            ax.scatter(
                res.points_lidar[:, 0],
                res.points_lidar[:, 1],
                c=c,
                s=3,
                label=f"box {idx} (d={res.median_depth:.1f}m)",
            )

    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("BEV — Frustum-Filtered Points")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig
