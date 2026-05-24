"""Tests for frustum filtering."""

from __future__ import annotations

import numpy as np

from lidar_fusion.frustum import (
    estimate_3d_center,
    estimate_depth_stats,
    frustum_filter,
    points_in_bbox,
)


def test_points_in_bbox_basic():
    uv = np.array([[100, 200], [150, 250], [300, 400], [50, 50]], dtype=np.float64)
    bbox = np.array([80, 180, 200, 300, 0.9, 2])
    mask = points_in_bbox(uv, bbox)
    assert mask[0] is np.bool_(True)   # (100,200) inside
    assert mask[1] is np.bool_(True)   # (150,250) inside
    assert mask[2] is np.bool_(False)  # (300,400) outside
    assert mask[3] is np.bool_(False)  # (50,50) outside


def test_points_in_bbox_edge():
    """Points exactly on bbox boundary should be included."""
    uv = np.array([[100, 200]], dtype=np.float64)
    bbox = np.array([100, 200, 300, 400, 0.9, 2])
    mask = points_in_bbox(uv, bbox)
    assert mask[0] is np.bool_(True)


def test_frustum_filter_basic():
    # 5 LiDAR points, all in front of camera
    pts_lidar = np.array([
        [10, 2, 0, 0.5],
        [20, -1, 1, 0.5],
        [30, 5, -1, 0.5],
        [15, 0, 0, 0.5],
        [40, 3, 2, 0.5],
    ], dtype=np.float64)

    # Simulated projection: all valid
    valid_mask = np.ones(5, dtype=bool)
    projected_uv = np.array([
        [100, 200],
        [150, 250],
        [300, 400],
        [120, 220],
        [500, 100],
    ], dtype=np.float64)

    bbox = np.array([80, 180, 200, 300, 0.9, 2])  # should catch pts 0, 1, 3

    result = frustum_filter(pts_lidar, projected_uv, valid_mask, bbox)
    assert result.num_points == 3
    assert result.median_depth > 0


def test_frustum_filter_empty_bbox():
    pts = np.array([[10, 2, 0, 0.5]], dtype=np.float64)
    valid_mask = np.ones(1, dtype=bool)
    uv = np.array([[100, 200]], dtype=np.float64)
    bbox = np.array([500, 500, 600, 600, 0.5, 0])  # far from the point

    result = frustum_filter(pts, uv, valid_mask, bbox)
    assert result.num_points == 0
    assert result.points_3d.shape == (0, 3)
    assert result.median_depth == 0.0


def test_frustum_filter_no_valid_points():
    pts = np.array([[10, 2, 0, 0.5]], dtype=np.float64)
    valid_mask = np.zeros(1, dtype=bool)  # nothing valid
    uv = np.empty((0, 2), dtype=np.float64)
    bbox = np.array([0, 0, 1000, 1000, 0.5, 0])

    result = frustum_filter(pts, uv, valid_mask, bbox)
    assert result.num_points == 0


def test_estimate_3d_center_basic():
    pts = np.array([[10, 2, 0], [20, 4, 2]])
    center = estimate_3d_center(pts)
    np.testing.assert_array_almost_equal(center, [15, 3, 1])


def test_estimate_3d_center_empty():
    center = estimate_3d_center(np.empty((0, 3)))
    np.testing.assert_array_equal(center, [0, 0, 0])


def test_estimate_depth_stats():
    pts = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]])
    stats = estimate_depth_stats(pts)
    assert stats["median"] == 20.0
    assert stats["mean"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    assert stats["count"] == 3


def test_estimate_depth_stats_empty():
    stats = estimate_depth_stats(np.empty((0, 3)))
    assert stats["count"] == 0
