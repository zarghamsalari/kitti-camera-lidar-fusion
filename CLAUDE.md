# CLAUDE.md — kitti-camera-lidar-fusion

## Project purpose

Classical KITTI camera-LiDAR fusion for learning robotic perception, 3D localisation, and future digital inspection robotics foundations for AISUS. This is educational — do not overclaim industrial readiness.

## Repository layout

- `src/lidar_fusion/` — core library (calibration, pointcloud, projection, frustum, visualisation, yolo_detect)
- `scripts/` — CLI demo scripts (run_projection_demo.py, run_frustum_demo.py)
- `streamlit_app/app.py` — interactive Streamlit demo
- `tests/` — pytest test suite
- `configs/` — YAML configuration
- `docs/` — technical documentation
- `data/` — KITTI data (not committed)

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Commands

- `make test` — run tests
- `make lidar-projection` — batch LiDAR projection
- `make lidar-frustum` — batch frustum filtering
- `make demo` — launch Streamlit demo

## Rules — DO NOT

- Download KITTI data or add large files
- Add YOLO weights or model files to the repo
- Implement PointPillars, VoxelNet, BEVFusion, CenterPoint, SLAM, robot control, or deep 3D detector training
- Build a full robot or overclaim industrial readiness
- Touch folders outside this repository

## Rules — DO

- Keep the project lightweight, educational, and honest
- Keep YOLO optional only (graceful fallback when ultralytics is not installed)
- Focus on classical sensor fusion: calibration, point cloud loading, LiDAR projection, frustum filtering, 3D centre/depth estimation, visualisation, tests, Streamlit demo, and documentation
- Run `make test` after changes to verify nothing breaks
