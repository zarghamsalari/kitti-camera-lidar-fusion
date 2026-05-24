"""Tests for KITTI calibration parsing and projection."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_fusion.calibration import (
    KittiCalibration,
    camera_to_image,
    lidar_to_camera,
    load_calibration,
    project_lidar_to_image,
)


def test_load_calibration_shapes(sample_calib_file):
    calib = load_calibration(sample_calib_file)
    assert calib.P2.shape == (3, 4)
    assert calib.R0_rect.shape == (3, 3)
    assert calib.Tr_velo_to_cam.shape == (3, 4)


def test_load_calibration_values(sample_calib_file):
    calib = load_calibration(sample_calib_file)
    # R0_rect should be identity in our fixture
    np.testing.assert_array_almost_equal(calib.R0_rect, np.eye(3))


def test_lidar_to_camera_shape(sample_calib_file):
    calib = load_calibration(sample_calib_file)
    pts = np.array([[10.0, 2.0, -0.5], [20.0, -1.0, 1.0]])
    result = lidar_to_camera(pts, calib)
    assert result.shape == (2, 3)


def test_lidar_to_camera_identity_transform():
    """With identity-like transform, verify coordinate mapping."""
    # Tr_velo_to_cam: x_cam = -y_velo, y_cam = -z_velo, z_cam = x_velo
    calib = KittiCalibration(
        P2=np.eye(3, 4),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=np.float64),
    )
    pts = np.array([[10.0, 5.0, -1.0]])
    cam = lidar_to_camera(pts, calib)
    # z_cam = x_velo = 10 (forward in camera)
    assert cam[0, 2] == pytest.approx(10.0)


def test_camera_to_image_shape():
    calib = KittiCalibration(
        P2=np.eye(3, 4),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.eye(3, 4),
    )
    pts_cam = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    result = camera_to_image(pts_cam, calib)
    assert result.shape == (2, 3)


def test_camera_to_image_simple_projection():
    """With identity P2, u = x/z, v = y/z."""
    calib = KittiCalibration(
        P2=np.eye(3, 4),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.eye(3, 4),
    )
    pts_cam = np.array([[5.0, 10.0, 20.0]])
    result = camera_to_image(pts_cam, calib)
    assert result[0, 0] == pytest.approx(0.25)  # u = 5/20
    assert result[0, 1] == pytest.approx(0.50)  # v = 10/20
    assert result[0, 2] == pytest.approx(20.0)  # depth


def test_project_lidar_to_image_filters_behind_camera():
    """Points behind the camera (z_cam <= 0) should be filtered."""
    # With this Tr, z_cam = x_velo
    calib = KittiCalibration(
        P2=np.eye(3, 4),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=np.float64),
    )
    # First point: x_velo=10 → z_cam=10 (in front), second: x_velo=-5 → z_cam=-5 (behind)
    pts = np.array([[10.0, 2.0, 1.0, 0.5], [-5.0, 2.0, 1.0, 0.5]])
    uv, depth, valid = project_lidar_to_image(pts, calib)

    assert valid[0] is np.bool_(True)
    assert valid[1] is np.bool_(False)
    assert uv.shape[0] == 1  # only one valid point
    assert depth.shape[0] == 1


def test_project_lidar_empty():
    calib = KittiCalibration(
        P2=np.eye(3, 4),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.eye(3, 4),
    )
    pts = np.array([[-10.0, 0.0, 0.0, 0.5]])  # behind camera
    uv, depth, valid = project_lidar_to_image(pts, calib)
    assert uv.shape == (0, 2)
    assert depth.shape == (0,)
