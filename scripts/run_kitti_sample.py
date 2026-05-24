"""Run classical camera-LiDAR fusion on a real KITTI sample.

Parses KITTI ground-truth labels (no YOLO required), projects LiDAR onto the
image, performs frustum filtering for the selected object, and saves an
inspection-style report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running directly from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np

from lidar_fusion.calibration import load_calibration, project_lidar_to_image
from lidar_fusion.frustum import frustum_filter
from lidar_fusion.pointcloud import filter_by_range, filter_forward_points, load_velodyne_bin
from lidar_fusion.projection import draw_lidar_on_image


# ---------------------------------------------------------------------------
# KITTI label parsing
# ---------------------------------------------------------------------------

def parse_kitti_labels(label_path: Path) -> list[dict]:
    """Parse a KITTI label_2 file into a list of object dicts.

    Each line: class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
    """
    objects: list[dict] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 15:
            continue
        objects.append({
            "class": parts[0],
            "truncated": float(parts[1]),
            "occluded": int(parts[2]),
            "alpha": float(parts[3]),
            "bbox": [float(parts[4]), float(parts[5]),
                     float(parts[6]), float(parts[7])],
            "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],
            "location": [float(parts[11]), float(parts[12]), float(parts[13])],
            "rotation_y": float(parts[14]),
        })
    return objects


def select_object(objects: list[dict]) -> dict:
    """Pick the first Car; fall back to the first labelled object."""
    for obj in objects:
        if obj["class"] == "Car":
            return obj
    # Skip DontCare
    for obj in objects:
        if obj["class"] != "DontCare":
            return obj
    return objects[0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_sample(
    sample_id: str,
    data_root: Path,
    output_dir: Path,
    max_distance: float = 80.0,
) -> dict:
    """Process one KITTI sample through the classical fusion pipeline."""
    image_path = data_root / "image_2" / f"{sample_id}.png"
    velodyne_path = data_root / "velodyne" / f"{sample_id}.bin"
    calib_path = data_root / "calib" / f"{sample_id}.txt"
    label_path = data_root / "label_2" / f"{sample_id}.txt"

    for p in [image_path, velodyne_path, calib_path, label_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    image = cv2.imread(str(image_path))
    points = load_velodyne_bin(velodyne_path)
    points = filter_forward_points(points)
    points = filter_by_range(points, max_distance)
    calib = load_calibration(calib_path)

    # --- Parse labels and select object ---
    objects = parse_kitti_labels(label_path)
    if not objects:
        raise ValueError(f"No objects in {label_path}")
    obj = select_object(objects)
    x1, y1, x2, y2 = obj["bbox"]

    # --- Project LiDAR onto image ---
    uv, depth, valid_mask = project_lidar_to_image(points, calib)

    # --- Save projection overlay ---
    proj_img = draw_lidar_on_image(image.copy(), uv, depth)
    # Draw the selected 2D bounding box
    cv2.rectangle(proj_img, (int(x1), int(y1)), (int(x2), int(y2)),
                  color=(0, 255, 0), thickness=2)
    cv2.putText(proj_img, obj["class"], (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out_image_path = output_dir / f"{sample_id}_projection.png"
    cv2.imwrite(str(out_image_path), proj_img)

    # --- Frustum filter ---
    bbox_array = np.array([x1, y1, x2, y2, 1.0, 0.0], dtype=np.float64)
    result = frustum_filter(points, uv, valid_mask, bbox_array)

    # --- Build report ---
    report = {
        "sample_id": sample_id,
        "object_class": obj["class"],
        "bbox_2d": {"left": x1, "top": y1, "right": x2, "bottom": y2},
        "gt_location_cam": obj["location"],
        "gt_dimensions_hwl": obj["dimensions"],
        "lidar_points_in_box": result.num_points,
        "median_depth_m": round(result.median_depth, 3),
        "mean_depth_m": round(result.mean_depth, 3),
        "estimated_3d_centre": [round(v, 3) for v in result.center_3d.tolist()],
        "total_projected_points": int(uv.shape[0]),
        "output_image": str(out_image_path),
        "context": {
            "camera": "KITTI left colour camera (image_2) = inspection robot RGB camera",
            "lidar": "Velodyne HDL-64E = robot 3D spatial sensor",
            "calibration": "P2 + R0_rect + Tr_velo_to_cam = multi-sensor robot calibration",
            "pipeline": "2D bbox + LiDAR frustum = first step toward 3D anomaly/object localisation",
        },
    }

    out_report_path = output_dir / f"{sample_id}_report.json"
    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=2)

    report["output_report"] = str(out_report_path)

    # --- Console summary ---
    print()
    print("=" * 60)
    print("  KITTI Sample Fusion Report")
    print("=" * 60)
    print(f"  Sample ID           : {sample_id}")
    print(f"  Object class        : {obj['class']}")
    print(f"  2D bounding box     : left={x1:.1f}, top={y1:.1f}, "
          f"right={x2:.1f}, bottom={y2:.1f}")
    print(f"  LiDAR points in box : {result.num_points}")
    print(f"  Median depth (m)    : {result.median_depth:.3f}")
    print(f"  Estimated 3D centre : {report['estimated_3d_centre']}")
    print(f"  Output image        : {out_image_path}")
    print(f"  Output report       : {out_report_path}")
    print("=" * 60)
    print()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classical camera-LiDAR fusion on a real KITTI sample",
    )
    parser.add_argument(
        "--sample-id", type=str, default="000000",
        help="KITTI sample ID, e.g. 000000",
    )
    parser.add_argument(
        "--data-root", type=str, default="data/kitti/object/training",
        help="Root directory containing image_2/, velodyne/, calib/, label_2/",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for output images and reports",
    )
    parser.add_argument(
        "--max-distance", type=float, default=80.0,
        help="Maximum LiDAR range in metres",
    )
    args = parser.parse_args()

    run_sample(
        sample_id=args.sample_id,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()
