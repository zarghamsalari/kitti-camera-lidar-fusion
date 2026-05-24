"""Visualisation utilities for LiDAR fusion outputs."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from lidar_fusion.frustum import FrustumResult, compute_3d_extent
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


def _cuboid_edges(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return x, y, z lists for the 12 edges of an axis-aligned cuboid.

    Edges are separated by ``None`` so Plotly draws them as disconnected lines.
    """
    corners = [
        (xmin, ymin, zmin), (xmax, ymin, zmin),
        (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax),
        (xmax, ymax, zmax), (xmin, ymax, zmax),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs


def make_3d_scene(
    points: np.ndarray,
    results: list[FrustumResult],
    class_names: list[str] | None = None,
    max_points: int = 8000,
) -> go.Figure:
    """Build an interactive Plotly 3D scatter of the LiDAR scene.

    Parameters
    ----------
    points : Nx4 full (filtered-forward) point cloud.
    results : frustum-filtered results per object.
    class_names : optional per-result class labels.
    max_points : downsample background cloud to this many points.
    """
    fig = go.Figure()

    # Downsample background cloud
    pts = points[:, :3]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(pts.shape[0], max_points, replace=False)
        pts_ds = pts[idx]
    else:
        pts_ds = pts

    fig.add_trace(go.Scatter3d(
        x=pts_ds[:, 0], y=pts_ds[:, 1], z=pts_ds[:, 2],
        mode="markers",
        marker=dict(size=1, color="lightgrey", opacity=0.4),
        name="LiDAR scene",
        hoverinfo="skip",
    ))

    colours = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#1abc9c", "#9b59b6"]

    for i, r in enumerate(results):
        if r.num_points == 0:
            continue
        c = colours[i % len(colours)]
        label = class_names[i] if class_names and i < len(class_names) else f"obj {i}"

        # Frustum points
        fig.add_trace(go.Scatter3d(
            x=r.points_lidar[:, 0],
            y=r.points_lidar[:, 1],
            z=r.points_lidar[:, 2],
            mode="markers",
            marker=dict(size=2.5, color=c),
            name=f"{label} ({r.num_points} pts)",
        ))

        # 3D centre marker
        fig.add_trace(go.Scatter3d(
            x=[r.center_3d[0]], y=[r.center_3d[1]], z=[r.center_3d[2]],
            mode="markers",
            marker=dict(size=6, color=c, symbol="diamond"),
            name=f"{label} centre",
            showlegend=False,
        ))

        # 3D cuboid from axis-aligned extent
        ext = compute_3d_extent(r.points_lidar)
        if ext["volume"] > 0:
            ex, ey, ez = _cuboid_edges(
                ext["x_min"], ext["x_max"],
                ext["y_min"], ext["y_max"],
                ext["z_min"], ext["z_max"],
            )
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color=c, width=3),
                name=f"{label} extent",
                showlegend=False,
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x (forward, m)",
            yaxis_title="y (left, m)",
            zaxis_title="z (up, m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        legend=dict(font=dict(size=10)),
    )
    return fig
