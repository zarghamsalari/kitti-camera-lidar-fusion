"""Tests for LiDAR-to-image projection rendering."""

from __future__ import annotations

import numpy as np

from lidar_fusion.projection import depth_colour_map, draw_lidar_on_image


def test_depth_colour_map_shape():
    depth = np.array([5.0, 10.0, 20.0, 40.0])
    colours = depth_colour_map(depth)
    assert colours.shape == (4, 3)
    assert colours.dtype == np.uint8


def test_depth_colour_map_empty():
    colours = depth_colour_map(np.array([]))
    assert colours.shape == (0, 3)


def test_draw_lidar_on_image_preserves_shape():
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    uv = np.array([[100.0, 200.0], [500.0, 300.0], [9999.0, 9999.0]])
    depth = np.array([10.0, 20.0, 30.0])
    result = draw_lidar_on_image(img, uv, depth)
    assert result.shape == img.shape
    # Out-of-bounds point should be ignored; in-bounds should add colour
    assert result[200, 100].sum() > 0  # first point drawn
    assert result[0, 0].sum() == 0     # corner untouched


def test_draw_lidar_does_not_mutate_input():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    original = img.copy()
    uv = np.array([[50.0, 50.0]])
    depth = np.array([10.0])
    draw_lidar_on_image(img, uv, depth)
    np.testing.assert_array_equal(img, original)
