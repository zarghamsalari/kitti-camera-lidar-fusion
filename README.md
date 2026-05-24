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

No deep 3D detection networks are used.
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
  run_kitti_sample.py      # CLI: real KITTI sample fusion (label-based)

streamlit_app/
  app.py             # Interactive demo

configs/
  kitti_lidar.yaml   # Dataset and model configuration
```

## Running on a Real KITTI Sample

Place at least one KITTI object-detection sample under:

```
data/kitti/object/training/
  image_2/000000.png
  velodyne/000000.bin
  calib/000000.txt
  label_2/000000.txt
```

Local KITTI data is git-ignored and never committed to the repository.

Run the classical fusion pipeline on a single sample:

```bash
python scripts/run_kitti_sample.py \
  --sample-id 000000 \
  --data-root data/kitti/object/training \
  --output-dir outputs
```

This reads the ground-truth 2D label (no YOLO weights needed), projects LiDAR
points onto the image, filters LiDAR points inside the labelled bounding box,
and estimates depth and a 3D centre. Outputs are saved to `outputs/`.

This demonstrates the **perception and sensor-fusion layer** — the same
pipeline that a digital inspection robot would use:

| KITTI component | Robot equivalent |
|---|---|
| Left colour camera (`image_2`) | Inspection robot RGB / video camera |
| Velodyne LiDAR (`velodyne`) | Robot 3D LiDAR / depth / spatial sensor |
| Calibration files (`calib`) | Multi-sensor robot calibration |
| 2D box + LiDAR frustum filtering | First step toward 3D object / anomaly localisation |

## Project Positioning

This repository demonstrates a classical camera-LiDAR fusion pipeline for robotic perception and 3D spatial localisation. Using real KITTI object-detection data, it projects LiDAR points into the camera image, associates 2D object labels with 3D point-cloud evidence, performs frustum filtering, and estimates approximate object depth and 3D centre.

The project is positioned as a transparent foundation for digital inspection robotics, where cameras, LiDAR, calibration, and spatial reasoning are required before higher-level autonomy, anomaly mapping, inspection planning, or operator dashboards.

This repository focuses specifically on camera-LiDAR fusion: projecting 3D LiDAR evidence into the camera frame, associating 2D object regions with spatial point-cloud evidence, and producing interpretable depth and 3D-centre estimates.

See [docs/lidar_fusion_technical_overview.md](docs/lidar_fusion_technical_overview.md) for technical details.

## Current Scope and Limitations

- This is a classical perception and sensor-fusion baseline implementation.
- It does not include robot hardware, navigation, manipulation, or industrial validation.
- It uses ground-truth KITTI labels or optional 2D detections (YOLOv8) rather than a trained 3D detector.
- No deep 3D detection networks are used; 3D localisation relies entirely on classical geometry.
- It is designed as a transparent classical baseline before advanced 3D detection or robotic deployment.
