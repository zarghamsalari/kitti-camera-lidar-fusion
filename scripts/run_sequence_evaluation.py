"""Sequence evaluation: classical frustum-based 3D localisation on KITTI.

Runs the camera-LiDAR fusion pipeline across a range of KITTI frames,
compares estimated depth and 3D centre against ground-truth labels,
and produces per-object CSV results, summary statistics, failure-case
analysis, and evaluation figures.

Two evaluation modes separate fusion error from detection error:
  --detection-source labels   Use KITTI GT 2D boxes (evaluates fusion only)
  --detection-source yolo     Use external YOLO detections (evaluates full stack)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lidar_fusion.calibration import load_calibration, project_lidar_to_image
from lidar_fusion.depth_cleaning import METHODS as CLEANING_METHODS, apply_cleaning
from lidar_fusion.frustum import compute_3d_extent, frustum_filter
from lidar_fusion.pointcloud import filter_by_range, filter_forward_points, load_velodyne_bin


# ── KITTI label parsing ──────────────────────────────────────────────

def parse_kitti_labels(label_path: Path) -> list[dict]:
    """Parse KITTI label_2 file.  Returns list of object dicts."""
    objects: list[dict] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] == "DontCare":
            continue
        objects.append({
            "class": parts[0],
            "truncated": float(parts[1]),
            "occluded": int(parts[2]),
            "bbox": [float(parts[4]), float(parts[5]),
                     float(parts[6]), float(parts[7])],
            "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],
            "location": [float(parts[11]), float(parts[12]), float(parts[13])],
            "rotation_y": float(parts[14]),
        })
    return objects


def load_external_detections(det_path: Path, frame_id: str) -> list[dict]:
    """Load external detections JSON, filtered by frame_id."""
    records = json.loads(det_path.read_text())
    out: list[dict] = []
    for r in records:
        rid = str(r.get("frame_id", "")).zfill(6)
        if rid != frame_id:
            continue
        if "bbox" in r:
            bbox = r["bbox"]
            if isinstance(bbox, str):
                bbox = json.loads(bbox)
            xmin, ymin, xmax, ymax = [float(v) for v in bbox[:4]]
        else:
            xmin = float(r.get("xmin", r.get("x1", 0)))
            ymin = float(r.get("ymin", r.get("y1", 0)))
            xmax = float(r.get("xmax", r.get("x2", 0)))
            ymax = float(r.get("ymax", r.get("y2", 0)))
        out.append({
            "class": str(r.get("class_name", "unknown")),
            "confidence": float(r.get("confidence", 0.0)),
            "bbox": [xmin, ymin, xmax, ymax],
        })
    return out


# ── PCA yaw estimation (experimental) ───────────────────────────────

def estimate_bev_yaw(points: np.ndarray) -> float:
    """Estimate yaw angle from BEV PCA of frustum points (experimental).

    Uses the x-y plane (bird's-eye view) principal component direction.
    Returns angle in radians.  Not used for official metrics.
    """
    if points.shape[0] < 3:
        return 0.0
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(principal[1], principal[0]))


# ── Core evaluation ─────────────────────────────────────────────────

RESULT_FIELDS = [
    "frame_id", "object_id", "class_name", "detection_source", "depth_method",
    "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
    "raw_point_count", "cleaned_point_count", "point_retention_ratio",
    "raw_median_depth", "cleaned_median_depth", "depth_shift_raw_to_cleaned",
    "raw_depth_iqr", "cleaned_depth_iqr",
    "estimated_x", "estimated_y", "estimated_z",
    "gt_x", "gt_y", "gt_z",
    "depth_error", "centre_error",
    "x_extent", "y_extent", "z_extent", "volume",
    "pca_yaw_rad",
    "status", "failure_reason",
]


def _iqr(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def evaluate_object(
    points: np.ndarray,
    uv: np.ndarray,
    valid_mask: np.ndarray,
    obj: dict,
    frame_id: str,
    obj_idx: int,
    det_source: str,
    method: str,
    gt_location: list[float] | None,
) -> dict:
    """Evaluate one object with one cleaning method.  Returns a result dict."""
    bbox = obj["bbox"]
    bbox_array = np.array(bbox + [obj.get("confidence", 1.0), 0.0], dtype=np.float64)

    result = {
        "frame_id": frame_id,
        "object_id": obj_idx,
        "class_name": obj["class"],
        "detection_source": det_source,
        "depth_method": method,
        "bbox_left": bbox[0],
        "bbox_top": bbox[1],
        "bbox_right": bbox[2],
        "bbox_bottom": bbox[3],
    }

    # Raw frustum filter
    filt = frustum_filter(points, uv, valid_mask, bbox_array)
    raw_count = filt.num_points

    if raw_count == 0:
        result.update({
            "raw_point_count": 0, "cleaned_point_count": 0,
            "point_retention_ratio": 0.0,
            "raw_median_depth": 0.0, "cleaned_median_depth": 0.0,
            "depth_shift_raw_to_cleaned": 0.0,
            "raw_depth_iqr": 0.0, "cleaned_depth_iqr": 0.0,
            "estimated_x": 0.0, "estimated_y": 0.0, "estimated_z": 0.0,
            "gt_x": gt_location[0] if gt_location else "",
            "gt_y": gt_location[1] if gt_location else "",
            "gt_z": gt_location[2] if gt_location else "",
            "depth_error": "", "centre_error": "",
            "x_extent": 0.0, "y_extent": 0.0, "z_extent": 0.0, "volume": 0.0,
            "pca_yaw_rad": 0.0,
            "status": "fail", "failure_reason": "no_frustum_points",
        })
        return result

    raw_pts = filt.points_lidar
    raw_depths = raw_pts[:, 0]
    raw_median = float(np.median(raw_depths))
    raw_iqr_val = _iqr(raw_depths)

    # Apply cleaning
    inlier_mask = apply_cleaning(raw_pts, method)
    cleaned_pts = raw_pts[inlier_mask]
    cleaned_count = int(cleaned_pts.shape[0])

    if cleaned_count == 0:
        # Cleaning removed everything -- fall back to raw
        cleaned_pts = raw_pts
        cleaned_count = raw_count

    cleaned_depths = cleaned_pts[:, 0]
    cleaned_median = float(np.median(cleaned_depths))
    cleaned_iqr_val = _iqr(cleaned_depths)
    cleaned_centre = cleaned_pts.mean(axis=0)
    ext = compute_3d_extent(cleaned_pts)
    yaw = estimate_bev_yaw(cleaned_pts)

    retention = cleaned_count / raw_count if raw_count > 0 else 0.0

    result.update({
        "raw_point_count": raw_count,
        "cleaned_point_count": cleaned_count,
        "point_retention_ratio": round(retention, 4),
        "raw_median_depth": round(raw_median, 4),
        "cleaned_median_depth": round(cleaned_median, 4),
        "depth_shift_raw_to_cleaned": round(cleaned_median - raw_median, 4),
        "raw_depth_iqr": round(raw_iqr_val, 4),
        "cleaned_depth_iqr": round(cleaned_iqr_val, 4),
        "estimated_x": round(float(cleaned_centre[0]), 4),
        "estimated_y": round(float(cleaned_centre[1]), 4),
        "estimated_z": round(float(cleaned_centre[2]), 4),
        "x_extent": round(ext["x_extent"], 4),
        "y_extent": round(ext["y_extent"], 4),
        "z_extent": round(ext["z_extent"], 4),
        "volume": round(ext["volume"], 4),
        "pca_yaw_rad": round(yaw, 4),
    })

    # GT comparison (KITTI GT location is in camera frame: x-right, y-down, z-forward)
    # Our estimates are in Velodyne frame: x-forward, y-left, z-up
    # Depth comparison: estimated x (Velodyne forward) vs GT z (camera forward)
    if gt_location:
        gt_x, gt_y, gt_z = gt_location
        result["gt_x"] = round(gt_x, 4)
        result["gt_y"] = round(gt_y, 4)
        result["gt_z"] = round(gt_z, 4)

        # Depth: Velodyne x ≈ camera z (forward distance)
        depth_err = abs(cleaned_median - gt_z)
        result["depth_error"] = round(depth_err, 4)

        # 3D centre error: compare in camera frame convention
        # Rough mapping: Velo(x,y,z) ≈ Cam(z, -x, -y) approximately
        # But since both are approximate, use Euclidean on GT cam vs est Velo
        # projected into a comparable space:
        # est_cam_z ≈ est_velo_x, est_cam_x ≈ -est_velo_y, est_cam_y ≈ -est_velo_z
        est_cam = np.array([-cleaned_centre[1], -cleaned_centre[2], cleaned_centre[0]])
        gt_cam = np.array([gt_x, gt_y, gt_z])
        centre_err = float(np.linalg.norm(est_cam - gt_cam))
        result["centre_error"] = round(centre_err, 4)
        result["status"] = "valid"
        result["failure_reason"] = ""
    else:
        result["gt_x"] = ""
        result["gt_y"] = ""
        result["gt_z"] = ""
        result["depth_error"] = ""
        result["centre_error"] = ""
        result["status"] = "valid_no_gt"
        result["failure_reason"] = ""

    return result


# ── Summary and figures ──────────────────────────────────────────────

def compute_summary(rows: list[dict]) -> dict:
    """Compute aggregate metrics from result rows."""
    valid = [r for r in rows if r["status"] == "valid"]
    failed = [r for r in rows if r["status"] == "fail"]

    depth_errors = [r["depth_error"] for r in valid if isinstance(r["depth_error"], (int, float))]
    centre_errors = [r["centre_error"] for r in valid if isinstance(r["centre_error"], (int, float))]

    de = np.array(depth_errors) if depth_errors else np.array([])
    ce = np.array(centre_errors) if centre_errors else np.array([])

    # Per-class metrics
    classes = sorted({r["class_name"] for r in rows})
    per_class: dict[str, dict] = {}
    for cls in classes:
        cls_valid = [r for r in valid if r["class_name"] == cls]
        cls_de = np.array([r["depth_error"] for r in cls_valid
                           if isinstance(r["depth_error"], (int, float))])
        per_class[cls] = {
            "count": len([r for r in rows if r["class_name"] == cls]),
            "valid": len(cls_valid),
            "depth_mae": round(float(cls_de.mean()), 4) if len(cls_de) > 0 else None,
            "depth_rmse": round(float(np.sqrt((cls_de ** 2).mean())), 4) if len(cls_de) > 0 else None,
        }

    # Per-method metrics
    methods = sorted({r["depth_method"] for r in rows})
    per_method: dict[str, dict] = {}
    for m in methods:
        m_valid = [r for r in valid if r["depth_method"] == m]
        m_de = np.array([r["depth_error"] for r in m_valid
                         if isinstance(r["depth_error"], (int, float))])
        m_ce = np.array([r["centre_error"] for r in m_valid
                         if isinstance(r["centre_error"], (int, float))])
        per_method[m] = {
            "count": len([r for r in rows if r["depth_method"] == m]),
            "valid": len(m_valid),
            "depth_mae": round(float(m_de.mean()), 4) if len(m_de) > 0 else None,
            "depth_rmse": round(float(np.sqrt((m_de ** 2).mean())), 4) if len(m_de) > 0 else None,
            "centre_mae": round(float(m_ce.mean()), 4) if len(m_ce) > 0 else None,
            "centre_rmse": round(float(np.sqrt((m_ce ** 2).mean())), 4) if len(m_ce) > 0 else None,
        }

    n_total = len(rows)
    n_unique_objs = len({(r["frame_id"], r["object_id"]) for r in rows})
    unique_methods = len(methods)

    return {
        "frames_processed": len({r["frame_id"] for r in rows}),
        "objects_processed": n_unique_objs,
        "total_rows": n_total,
        "depth_methods_evaluated": methods,
        "valid_objects": len(valid),
        "failed_objects": len(failed),
        "valid_frustum_rate": round(len(valid) / n_total, 4) if n_total > 0 else 0.0,
        "depth_mae": round(float(de.mean()), 4) if len(de) > 0 else None,
        "depth_rmse": round(float(np.sqrt((de ** 2).mean())), 4) if len(de) > 0 else None,
        "centre_mae": round(float(ce.mean()), 4) if len(ce) > 0 else None,
        "centre_rmse": round(float(np.sqrt((ce ** 2).mean())), 4) if len(ce) > 0 else None,
        "per_class_metrics": per_class,
        "per_method_metrics": per_method,
        "notes": {
            "3d_iou": "Not computed. Current cuboids are axis-aligned; KITTI GT boxes "
                      "are oriented. 3D IoU requires yaw/orientation estimation.",
            "coordinate_mapping": "Estimates are in Velodyne frame (x-forward, y-left, z-up). "
                                  "GT is in camera frame (x-right, y-down, z-forward). "
                                  "Depth error uses Velodyne-x vs Camera-z. Centre error "
                                  "uses an approximate Velo-to-cam mapping.",
            "pca_yaw": "Experimental BEV PCA yaw is reported but not used in metrics.",
        },
    }


def generate_figures(rows: list[dict], fig_dir: Path) -> None:
    """Generate evaluation plots and save to fig_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r["status"] == "valid"]
    if not valid:
        return

    depth_errors = np.array([r["depth_error"] for r in valid
                             if isinstance(r["depth_error"], (int, float))])
    centre_errors = np.array([r["centre_error"] for r in valid
                              if isinstance(r["centre_error"], (int, float))])
    gt_depths = np.array([r["gt_z"] for r in valid
                          if isinstance(r["gt_z"], (int, float))])
    est_depths = np.array([r["cleaned_median_depth"] for r in valid
                           if isinstance(r["gt_z"], (int, float))])
    point_counts = np.array([r["cleaned_point_count"] for r in valid
                             if isinstance(r["depth_error"], (int, float))])

    # 1. Predicted vs GT depth scatter
    if len(gt_depths) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_depths, est_depths, alpha=0.6, s=20)
        lims = [0, max(gt_depths.max(), est_depths.max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=1, label="ideal")
        ax.set_xlabel("GT Depth -- camera z (m)")
        ax.set_ylabel("Estimated Depth -- Velodyne x (m)")
        ax.set_title("Predicted vs Ground-Truth Depth")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(fig_dir / "depth_scatter.png", dpi=150)
        plt.close(fig)

    # 2. Depth error histogram
    if len(depth_errors) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(depth_errors, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Depth Error (m)")
        ax.set_ylabel("Count")
        ax.set_title("Depth Error Distribution")
        ax.axvline(np.median(depth_errors), color="red", ls="--",
                   label=f"median={np.median(depth_errors):.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "depth_error_hist.png", dpi=150)
        plt.close(fig)

    # 3. Centre error histogram
    if len(centre_errors) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(centre_errors, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("3D Centre Error (m)")
        ax.set_ylabel("Count")
        ax.set_title("3D Centre Error Distribution")
        ax.axvline(np.median(centre_errors), color="red", ls="--",
                   label=f"median={np.median(centre_errors):.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "centre_error_hist.png", dpi=150)
        plt.close(fig)

    # 4. Point count vs depth error
    if len(point_counts) > 0 and len(depth_errors) > 0:
        n = min(len(point_counts), len(depth_errors))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(point_counts[:n], depth_errors[:n], alpha=0.5, s=15)
        ax.set_xlabel("Cleaned LiDAR Point Count")
        ax.set_ylabel("Depth Error (m)")
        ax.set_title("Point Count vs Depth Error")
        fig.tight_layout()
        fig.savefig(fig_dir / "points_vs_error.png", dpi=150)
        plt.close(fig)

    # 5. Method comparison bar chart
    methods = sorted({r["depth_method"] for r in valid})
    if len(methods) > 1:
        method_maes = []
        for m in methods:
            m_de = [r["depth_error"] for r in valid
                    if r["depth_method"] == m and isinstance(r["depth_error"], (int, float))]
            method_maes.append(np.mean(m_de) if m_de else 0.0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(methods, method_maes, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Cleaning Method")
        ax.set_ylabel("Depth MAE (m)")
        ax.set_title("Depth MAE by Frustum Cleaning Method")
        fig.tight_layout()
        fig.savefig(fig_dir / "method_comparison.png", dpi=150)
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequence evaluation: classical frustum 3D localisation on KITTI",
    )
    parser.add_argument("--data-root", type=str, default="data/kitti/object/training")
    parser.add_argument("--start-id", type=str, default="000000")
    parser.add_argument("--end-id", type=str, default="000099")
    parser.add_argument("--output-dir", type=str, default="outputs/sequence_eval")
    parser.add_argument(
        "--detection-source", type=str, default="labels",
        choices=["labels", "yolo"],
        help="'labels' = KITTI GT 2D boxes (fusion-only eval), "
             "'yolo' = external detections (full-stack eval)",
    )
    parser.add_argument("--detections-json", type=str, default=None,
                        help="Path to external YOLO detections JSON")
    parser.add_argument(
        "--depth-method", type=str, nargs="+", default=["all"],
        help="Cleaning methods: raw, iqr, peak, dbscan, all",
    )
    parser.add_argument("--max-distance", type=float, default=80.0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve depth methods
    if "all" in args.depth_method:
        methods = list(CLEANING_METHODS.keys())
    else:
        methods = args.depth_method

    # Load external detections if needed
    ext_dets_data: list[dict] | None = None
    if args.detection_source == "yolo":
        det_path = Path(args.detections_json) if args.detections_json else Path("outputs/detections.json")
        if not det_path.exists():
            print(f"[error] External detections file not found: {det_path}")
            sys.exit(1)
        ext_dets_data = json.loads(det_path.read_text())

    det_source_label = "gt_labels" if args.detection_source == "labels" else "yolo_external"

    start = int(args.start_id)
    end = int(args.end_id)

    all_rows: list[dict] = []
    frames_processed = 0

    for fid_int in range(start, end + 1):
        frame_id = f"{fid_int:06d}"
        image_path = data_root / "image_2" / f"{frame_id}.png"
        velodyne_path = data_root / "velodyne" / f"{frame_id}.bin"
        calib_path = data_root / "calib" / f"{frame_id}.txt"
        label_path = data_root / "label_2" / f"{frame_id}.txt"

        # Check required files
        if not image_path.exists() or not velodyne_path.exists() or not calib_path.exists():
            continue

        # Load sensor data
        points = load_velodyne_bin(velodyne_path)
        points = filter_forward_points(points)
        points = filter_by_range(points, args.max_distance)
        calib = load_calibration(calib_path)
        uv, depth, valid_mask = project_lidar_to_image(points, calib)

        # Get GT labels (always needed for ground truth comparison)
        gt_objects = parse_kitti_labels(label_path) if label_path.exists() else []

        # Get detection boxes
        if args.detection_source == "labels":
            det_objects = gt_objects
        else:
            det_objects = load_external_detections(
                Path(args.detections_json) if args.detections_json else Path("outputs/detections.json"),
                frame_id,
            )

        if not det_objects:
            continue

        frames_processed += 1

        for obj_idx, obj in enumerate(det_objects):
            # Find matching GT for this detection (by IoU-like bbox overlap)
            gt_loc = None
            if gt_objects:
                gt_loc = _find_matching_gt(obj["bbox"], gt_objects)

            for method in methods:
                row = evaluate_object(
                    points, uv, valid_mask, obj,
                    frame_id, obj_idx, det_source_label, method, gt_loc,
                )
                all_rows.append(row)

        if frames_processed % 10 == 0:
            print(f"  processed {frames_processed} frames ({frame_id})...")

    if not all_rows:
        print("[warn] No frames or objects found in the specified range.")
        print(f"  Searched: {data_root}")
        print(f"  Range: {args.start_id} .. {args.end_id}")
        return

    # ── Save results.csv ──────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Save failure_cases.csv ────────────────────────────────────────
    failures = [r for r in all_rows if r["status"] == "fail"]
    fail_path = out_dir / "failure_cases.csv"
    with open(fail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(failures)

    # ── Save summary.json ─────────────────────────────────────────────
    summary = compute_summary(all_rows)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Generate figures ──────────────────────────────────────────────
    fig_dir = out_dir / "figures"
    generate_figures(all_rows, fig_dir)

    # ── Console summary ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Sequence Evaluation Complete")
    print("=" * 65)
    print(f"  Detection source     : {det_source_label}")
    print(f"  Depth methods        : {', '.join(methods)}")
    print(f"  Frames processed     : {summary['frames_processed']}")
    print(f"  Objects (unique)     : {summary['objects_processed']}")
    print(f"  Total rows           : {summary['total_rows']}")
    print(f"  Valid evaluations    : {summary['valid_objects']}")
    print(f"  Failed (no frustum)  : {summary['failed_objects']}")
    print(f"  Valid frustum rate   : {summary['valid_frustum_rate']:.1%}")
    if summary["depth_mae"] is not None:
        print(f"  Depth MAE            : {summary['depth_mae']:.3f} m")
        print(f"  Depth RMSE           : {summary['depth_rmse']:.3f} m")
    if summary["centre_mae"] is not None:
        print(f"  Centre MAE           : {summary['centre_mae']:.3f} m")
        print(f"  Centre RMSE          : {summary['centre_rmse']:.3f} m")
    print(f"  Results CSV          : {csv_path}")
    print(f"  Summary JSON         : {summary_path}")
    print(f"  Failure cases        : {fail_path}")
    print(f"  Figures              : {fig_dir}")
    print("=" * 65)
    print()


def _find_matching_gt(det_bbox: list[float], gt_objects: list[dict]) -> list[float] | None:
    """Find the GT object with highest 2D bbox IoU overlap and return its 3D location."""
    best_iou = 0.0
    best_loc = None
    dx1, dy1, dx2, dy2 = det_bbox
    for gt in gt_objects:
        gx1, gy1, gx2, gy2 = gt["bbox"]
        # 2D IoU
        ix1 = max(dx1, gx1)
        iy1 = max(dy1, gy1)
        ix2 = min(dx2, gx2)
        iy2 = min(dy2, gy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_d = (dx2 - dx1) * (dy2 - dy1)
        area_g = (gx2 - gx1) * (gy2 - gy1)
        union = area_d + area_g - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_loc = gt["location"]
    return best_loc if best_iou > 0.1 else None


if __name__ == "__main__":
    main()
