# KITTI Camera-LiDAR Fusion

Classical camera-LiDAR fusion pipeline for KITTI 3D Object Detection.

## Overview

This project implements a lightweight, explainable camera-LiDAR fusion demo for autonomous driving perception:

1. Load KITTI camera images and Velodyne `.bin` point clouds
2. Parse KITTI calibration files (P2, R0_rect, Tr_velo_to_cam)
3. Project LiDAR points onto the camera image plane
4. Run YOLOv8 2D object detection
5. Filter LiDAR points inside each 2D detection box (frustum filtering)
6. Estimate approximate object depth and 3D centre
7. Visualise results: projection overlay, detection+LiDAR overlay, bird's-eye view

No deep 3D detection (PointPillars, VoxelNet, BEVFusion) is used.
This is a classical geometry + 2D detection approach.

## Setup

```bash
python -m pip install -e ".[dev]"
```

### KITTI Data

Download from [KITTI 3D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and extract to:

```
data/kitti3d/training/
  image_2/       # left colour camera images
  velodyne/      # Velodyne .bin point clouds
  calib/         # calibration files
  label_2/       # ground truth labels (optional)
```

## Usage

### LiDAR Projection

```bash
python scripts/run_projection_demo.py \
  --image data/kitti3d/training/image_2/000001.png \
  --velodyne data/kitti3d/training/velodyne/000001.bin \
  --calib data/kitti3d/training/calib/000001.txt \
  --out outputs/lidar_projection/000001_projection.png
```

Or batch via config:

```bash
make lidar-projection
```

### Frustum Filtering Demo

```bash
python scripts/run_frustum_demo.py \
  --image data/kitti3d/training/image_2/000001.png \
  --velodyne data/kitti3d/training/velodyne/000001.bin \
  --calib data/kitti3d/training/calib/000001.txt \
  --out-dir outputs/lidar_frustum
```

Or batch via config:

```bash
make lidar-frustum
```

### Streamlit Demo

```bash
make demo
```

## Testing

```bash
make test
```

## Project Structure

```
src/lidar_fusion/
  calibration.py     # KITTI calib parsing, coordinate transforms
  pointcloud.py      # Velodyne .bin loader, filtering, BEV plot
  projection.py      # LiDAR-to-image projection and rendering
  frustum.py         # 2D bbox frustum filtering, depth estimation
  visualisation.py   # Combined overlay and BEV visualisation
  yolo_detect.py     # YOLOv8 2D detection wrapper

scripts/
  run_projection_demo.py   # CLI: project LiDAR onto image
  run_frustum_demo.py      # CLI: YOLO + frustum filtering

streamlit_app/
  app.py             # Interactive demo

configs/
  kitti_lidar.yaml   # Dataset and model configuration
```

## Week 2 Context

This project is the Week 2 deliverable in a multi-week autonomous driving perception series.
Week 1 covered 2D detection and multi-object tracking on KITTI MOT
([kitti-tracking](https://github.com/zarghamsalari/kitti-tracking)).
This week extends into 3D sensor fusion using camera-LiDAR geometry.

See [docs/week2_lidar_fusion.md](docs/week2_lidar_fusion.md) for technical details.
