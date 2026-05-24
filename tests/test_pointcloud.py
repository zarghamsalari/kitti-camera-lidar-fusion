"""Tests for point cloud loading and filtering."""

from __future__ import annotations

import numpy as np

from lidar_fusion.pointcloud import (
    filter_by_range,
    filter_forward_points,
    load_velodyne_bin,
)


def test_load_velodyne_shape(sample_velodyne_file):
    pts = load_velodyne_bin(sample_velodyne_file)
    assert pts.shape == (10, 4)
    assert pts.dtype == np.float32


def test_filter_forward_removes_negative_x():
    pts = np.array([
        [10, 0, 0, 0.5],
        [-5, 0, 0, 0.5],
        [20, 1, 1, 0.5],
    ], dtype=np.float32)
    filtered = filter_forward_points(pts)
    assert filtered.shape[0] == 2
    assert (filtered[:, 0] > 0).all()


def test_filter_by_range():
    pts = np.array([
        [10, 0, 0, 0.5],
        [100, 0, 0, 0.5],
        [5, 3, 0, 0.5],
    ], dtype=np.float32)
    filtered = filter_by_range(pts, max_distance=50.0)
    assert filtered.shape[0] == 2  # only first and third within 50m


def test_filter_forward_all_behind():
    pts = np.array([[-1, 0, 0, 0], [-2, 3, 1, 0]], dtype=np.float32)
    filtered = filter_forward_points(pts)
    assert filtered.shape[0] == 0


def test_load_velodyne_values(sample_velodyne_file):
    pts = load_velodyne_bin(sample_velodyne_file)
    # All x values should be positive (generated in conftest with uniform(5,50))
    assert (pts[:, 0] > 0).all()
