"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_calib_file(tmp_path: Path) -> Path:
    """Create a minimal KITTI calibration file with identity-like matrices."""
    content = (
        "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "P2: 721.5377 0.0 609.5593 44.8573 0.0 721.5377 172.854 0.2164 0.0 0.0 1.0 0.0027\n"
        "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    path = tmp_path / "calib.txt"
    path.write_text(content)
    return path


@pytest.fixture
def sample_velodyne_file(tmp_path: Path) -> Path:
    """Create a tiny .bin with 10 points in front of the sensor."""
    rng = np.random.default_rng(42)
    points = np.zeros((10, 4), dtype=np.float32)
    points[:, 0] = rng.uniform(5, 50, 10)   # x forward
    points[:, 1] = rng.uniform(-10, 10, 10)  # y lateral
    points[:, 2] = rng.uniform(-1, 2, 10)    # z height
    points[:, 3] = rng.uniform(0, 1, 10)     # intensity
    path = tmp_path / "test.bin"
    points.tofile(str(path))
    return path
