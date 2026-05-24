"""CLI script: YOLO detection + LiDAR frustum filtering on KITTI frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from lidar_fusion.calibration import load_calibration, project_lidar_to_image
from lidar_fusion.frustum import frustum_filter
from lidar_fusion.pointcloud import filter_by_range, filter_forward_points, load_velodyne_bin
from lidar_fusion.projection import draw_lidar_on_image
from lidar_fusion.visualisation import draw_boxes_with_lidar, make_frustum_bev
from lidar_fusion.yolo_detect import detect_objects


def process_frame(
    image_path: Path,
    velodyne_path: Path,
    calib_path: Path,
    out_dir: Path,
    frame_id: str,
    weights: str = "yolov8m.pt",
    max_distance: float = 80.0,
    min_points: int = 3,
) -> list[dict]:
    """Process a single frame: detect, project, filter, save outputs."""
    image = cv2.imread(str(image_path))
    points = load_velodyne_bin(velodyne_path)
    points = filter_forward_points(points)
    points = filter_by_range(points, max_distance)
    calib = load_calibration(calib_path)

    # Project LiDAR onto image
    uv, depth, valid_mask = project_lidar_to_image(points, calib)

    # 1) Projection overlay
    proj_img = draw_lidar_on_image(image, uv, depth)
    cv2.imwrite(str(out_dir / f"{frame_id}_projection.png"), proj_img)

    # 2) YOLO detections
    detections = detect_objects(image_path, weights=weights)

    # 3) Frustum filter each detection
    results = []
    for det in detections:
        res = frustum_filter(points, uv, valid_mask, det)
        if res.num_points >= min_points:
            results.append(res)

    # 4) Boxes + filtered LiDAR overlay
    box_img = draw_boxes_with_lidar(image, uv, depth, results, valid_mask, points)
    cv2.imwrite(str(out_dir / f"{frame_id}_boxes_lidar.png"), box_img)

    # 5) BEV plot
    bev_fig = make_frustum_bev(points, results)
    bev_fig.savefig(str(out_dir / f"{frame_id}_bev.png"), dpi=150)

    # 6) JSON summary
    summary = []
    for r in results:
        summary.append({
            "bbox": r.bbox[:4].tolist(),
            "confidence": float(r.bbox[4]) if len(r.bbox) > 4 else 0.0,
            "class_id": int(r.bbox[5]) if len(r.bbox) > 5 else -1,
            "median_depth_m": r.median_depth,
            "mean_depth_m": r.mean_depth,
            "center_3d": r.center_3d.tolist(),
            "num_points": r.num_points,
        })

    with open(out_dir / f"{frame_id}_detections.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] {frame_id}: {len(results)} objects with depth")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR frustum filtering demo")
    parser.add_argument("--image", type=str)
    parser.add_argument("--velodyne", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--weights", type=str, default="yolov8m.pt")
    parser.add_argument("--out-dir", type=str, default="outputs/lidar_frustum")
    parser.add_argument("--config", type=str, help="YAML config (overrides above)")
    args = parser.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        ds = cfg["dataset"]
        root = Path(ds["root"])
        out_dir = Path(cfg["output"]["frustum_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = cfg["detector"]["weights"]
        max_dist = cfg["lidar"]["max_distance"]
        min_pts = cfg["lidar"]["min_points_per_box"]

        for sid in ds["sample_ids"]:
            img = root / ds["image_dir"] / f"{sid}.png"
            vel = root / ds["velodyne_dir"] / f"{sid}.bin"
            cal = root / ds["calib_dir"] / f"{sid}.txt"
            if not img.exists():
                print(f"[skip] {img} not found")
                continue
            process_frame(img, vel, cal, out_dir, sid, weights, max_dist, min_pts)
    else:
        if not all([args.image, args.velodyne, args.calib]):
            parser.error("Provide --image, --velodyne, --calib or --config")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_id = Path(args.image).stem
        process_frame(
            Path(args.image), Path(args.velodyne), Path(args.calib),
            out_dir, frame_id, args.weights,
        )


if __name__ == "__main__":
    main()
