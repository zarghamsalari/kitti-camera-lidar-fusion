"""Streamlit dashboard: Classical Camera-LiDAR Fusion for Robotic Perception."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is importable when running from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="Classical Camera-LiDAR Fusion for Robotic Perception",
    layout="wide",
)

st.title("Classical Camera-LiDAR Fusion for Robotic Perception")
st.caption(
    "A robotic perception fusion dashboard. YOLO provides 2D visual detections, "
    "LiDAR provides spatial evidence, and frustum filtering connects visual "
    "detections to 3D localisation -- estimating approximate object depth and "
    "3D centre from real sensor data."
)

# ── Dataset paths (preferred -> fallback) ─────────────────────────────
_CANDIDATES = [
    Path("data/kitti/object/training"),
    Path("data/kitti3d/training"),
]
DATA_ROOT: Path | None = None
for _p in _CANDIDATES:
    if _p.exists():
        DATA_ROOT = _p
        break

if DATA_ROOT is None:
    st.error(
        "KITTI data not found. Place at least one sample under "
        "`data/kitti/object/training/` (preferred) or `data/kitti3d/training/` "
        "with subdirectories: `image_2/`, `velodyne/`, `calib/`, `label_2/`."
    )
    st.stop()

IMAGE_DIR = DATA_ROOT / "image_2"
VELODYNE_DIR = DATA_ROOT / "velodyne"
CALIB_DIR = DATA_ROOT / "calib"
LABEL_DIR = DATA_ROOT / "label_2"

# ── Frame selector ───────────────────────────────────────────────────
image_files = sorted(IMAGE_DIR.glob("*.png"))
if not image_files:
    st.warning(f"No images found in {IMAGE_DIR}")
    st.stop()

frame_ids = [f.stem for f in image_files]
n_frames = len(frame_ids)

# Initialise session-state frame index
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

st.sidebar.subheader("Frame Selection")

# Prev / Next buttons
btn_cols = st.sidebar.columns([1, 1])
if btn_cols[0].button("Previous", disabled=st.session_state.frame_idx <= 0):
    st.session_state.frame_idx -= 1
if btn_cols[1].button("Next", disabled=st.session_state.frame_idx >= n_frames - 1):
    st.session_state.frame_idx += 1

# Frame slider (synced with buttons)
if n_frames > 1:
    st.session_state.frame_idx = st.sidebar.slider(
        "Frame index",
        0,
        n_frames - 1,
        st.session_state.frame_idx,
    )

selected_id = frame_ids[st.session_state.frame_idx]
st.sidebar.caption(f"Frame **{selected_id}**  ({st.session_state.frame_idx + 1}/{n_frames})")

image_path = IMAGE_DIR / f"{selected_id}.png"
velodyne_path = VELODYNE_DIR / f"{selected_id}.bin"
calib_path = CALIB_DIR / f"{selected_id}.txt"
label_path = LABEL_DIR / f"{selected_id}.txt"

for p, name in [
    (image_path, "Image"),
    (velodyne_path, "Velodyne"),
    (calib_path, "Calib"),
]:
    if not p.exists():
        st.error(f"{name} file missing: {p}")
        st.stop()

# ── Sidebar controls ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Parameters")
max_distance = st.sidebar.slider("Max LiDAR range (m)", 10.0, 120.0, 80.0, 5.0)

# ── Detection source ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Source")

has_labels = label_path.exists()

_EXT_DET_CANDIDATES = [
    Path("detections.json"),
    Path("detections.csv"),
    Path("outputs/detections.json"),
    Path("data/detections.json"),
    Path("data/detections.csv"),
]
ext_det_path: Path | None = None
for _dp in _EXT_DET_CANDIDATES:
    if _dp.exists():
        ext_det_path = _dp
        break

source_options = []
if has_labels:
    source_options.append("KITTI ground-truth labels")
if ext_det_path is not None:
    source_options.append("External YOLO detections")
source_options.append("YOLOv8 live inference (requires weights)")

detection_source = st.sidebar.radio("Detection source", source_options, index=0)

uploaded_det = st.sidebar.file_uploader(
    "Or upload detections file",
    type=["json", "csv"],
    help=(
        "JSON or CSV with columns: frame_id, class_name, confidence, "
        "xmin, ymin, xmax, ymax (or bbox list)"
    ),
)
if uploaded_det is not None:
    detection_source = "External YOLO detections"

yolo_conf = 0.25
if detection_source == "YOLOv8 live inference (requires weights)":
    yolo_conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.25, 0.05)

# ── Load and process ────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd

from lidar_fusion.calibration import load_calibration, project_lidar_to_image
from lidar_fusion.frustum import FrustumResult, compute_3d_extent, frustum_filter
from lidar_fusion.pointcloud import (
    filter_by_range,
    filter_forward_points,
    load_velodyne_bin,
    make_bev_plot,
)
from lidar_fusion.projection import draw_lidar_on_image
from lidar_fusion.visualisation import draw_boxes_with_lidar, make_3d_scene, make_frustum_bev

image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

points = load_velodyne_bin(velodyne_path)
points = filter_forward_points(points)
points = filter_by_range(points, max_distance)

calib = load_calibration(calib_path)
uv, depth, valid_mask = project_lidar_to_image(points, calib)


# ── Helper: load external detections ─────────────────────────────────

def _load_external_detections(
    source, frame_id: str,
) -> list[dict]:
    """Load detections from a JSON or CSV file/upload, filtered by frame_id."""
    records: list[dict] = []

    if hasattr(source, "read"):
        raw = source.read()
        source.seek(0)
        name = getattr(source, "name", "upload.json")
        if name.endswith(".json"):
            records = json.loads(raw)
        else:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            reader = csv.DictReader(text.splitlines())
            records = list(reader)
    else:
        path = Path(source)
        if path.suffix == ".json":
            records = json.loads(path.read_text())
        else:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                records = list(reader)

    filtered = []
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
        filtered.append({
            "class_name": str(r.get("class_name", "unknown")),
            "confidence": float(r.get("confidence", 0.0)),
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return filtered


# ── Detections ───────────────────────────────────────────────────────

KITTI_CLASSES_USED: list[str] = []
results: list[FrustumResult] = []
det_confidences: list[float] = []
min_pts = 3
source_label = "KITTI GT"

if detection_source == "YOLOv8 live inference (requires weights)":
    from lidar_fusion.yolo_detect import detect_objects

    raw_dets = detect_objects(image_path, conf=yolo_conf)
    for det in raw_dets:
        res = frustum_filter(points, uv, valid_mask, det)
        if res.num_points >= min_pts:
            results.append(res)
            KITTI_CLASSES_USED.append(
                {0: "person", 2: "car"}.get(int(det[5]), f"cls{int(det[5])}")
            )
            det_confidences.append(float(det[4]))
    source_label = "YOLOv8 Live"

elif detection_source == "External YOLO detections":
    det_source = uploaded_det if uploaded_det is not None else ext_det_path
    if det_source is not None:
        ext_dets = _load_external_detections(det_source, selected_id)
        if not ext_dets:
            st.sidebar.warning(
                f"No detections for frame {selected_id} in the external file."
            )
        for d in ext_dets:
            bbox_array = np.array(
                [d["xmin"], d["ymin"], d["xmax"], d["ymax"],
                 d["confidence"], 0.0],
                dtype=np.float64,
            )
            res = frustum_filter(points, uv, valid_mask, bbox_array)
            if res.num_points >= min_pts:
                results.append(res)
                KITTI_CLASSES_USED.append(d["class_name"])
                det_confidences.append(d["confidence"])
    source_label = "External YOLO"

elif has_labels:
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] == "DontCare":
            continue
        x1, y1 = float(parts[4]), float(parts[5])
        x2, y2 = float(parts[6]), float(parts[7])
        bbox_array = np.array([x1, y1, x2, y2, 1.0, 0.0], dtype=np.float64)
        res = frustum_filter(points, uv, valid_mask, bbox_array)
        if res.num_points >= min_pts:
            results.append(res)
            KITTI_CLASSES_USED.append(parts[0])
            det_confidences.append(1.0)
    source_label = "KITTI GT"

num_detections = len(results)

# Precompute 3D extents for each result
extents = [compute_3d_extent(r.points_lidar) for r in results]

# ── Top-level metric row ────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Frame ID", selected_id)
m2.metric("Projected LiDAR Points", f"{len(uv):,}")
m3.metric("Detections with Depth", num_detections)
m4.metric("Max LiDAR Range", f"{max_distance:.0f} m")
m5.metric("Detection Source", source_label)

# ── Tabs ─────────────────────────────────────────────────────────────
tab_cam, tab_proj, tab_frust, tab_3d, tab_bev, tab_report = st.tabs([
    "Camera View",
    "LiDAR Projection",
    "Detection + Frustum Evidence",
    "3D Point Cloud Scene",
    "Bird's-Eye View",
    "Inspection Report",
])

# ── Camera View ──────────────────────────────────────────────────────
with tab_cam:
    st.image(image_rgb, caption=f"Frame {selected_id} -- Left colour camera",
             use_container_width=True)

# ── LiDAR Projection ────────────────────────────────────────────────
with tab_proj:
    proj_img = draw_lidar_on_image(image.copy(), uv, depth)
    proj_rgb = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
    st.image(proj_rgb,
             caption="LiDAR points projected onto camera image (colour = depth)",
             use_container_width=True)

# ── Detection + Frustum Evidence ─────────────────────────────────────
with tab_frust:
    if results:
        box_img = draw_boxes_with_lidar(image, uv, depth, results, valid_mask, points)
        box_rgb = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
        st.image(box_rgb,
                 caption="2D visual detections with frustum-filtered LiDAR spatial evidence",
                 use_container_width=True)
    else:
        st.info("No detections with sufficient LiDAR spatial evidence in this frame.")

# ── 3D Point Cloud Scene ────────────────────────────────────────────
with tab_3d:
    fig_3d = make_3d_scene(points, results, class_names=KITTI_CLASSES_USED)
    st.plotly_chart(fig_3d, use_container_width=True)
    if results:
        st.caption(
            "Interactive 3D view: background LiDAR cloud (grey), "
            "frustum-filtered object points (colour), 3D centre markers "
            "(diamond), and axis-aligned 3D extent cuboids."
        )

# ── Bird's-Eye View ─────────────────────────────────────────────────
with tab_bev:
    if results:
        bev_fig = make_frustum_bev(points, results)
    else:
        bev_fig = make_bev_plot(points)
    st.pyplot(bev_fig)

# ── Inspection Report ────────────────────────────────────────────────
with tab_report:
    if results:
        rows = []
        for i, r in enumerate(results):
            cls_name = KITTI_CLASSES_USED[i] if i < len(KITTI_CLASSES_USED) else "unknown"
            conf = det_confidences[i] if i < len(det_confidences) else 0.0
            ext = extents[i]
            rows.append({
                "Object": i,
                "Class": cls_name,
                "Confidence": f"{conf:.2f}" if conf < 1.0 else "GT",
                "BBox (L, T, R, B)": (
                    f"({r.bbox[0]:.0f}, {r.bbox[1]:.0f}, "
                    f"{r.bbox[2]:.0f}, {r.bbox[3]:.0f})"
                ),
                "Median Depth (m)": f"{r.median_depth:.2f}",
                "LiDAR Points": r.num_points,
                "3D Centre (x, y, z)": (
                    f"({r.center_3d[0]:.2f}, {r.center_3d[1]:.2f}, "
                    f"{r.center_3d[2]:.2f})"
                ),
                "X Extent (m)": f"{ext['x_extent']:.2f}",
                "Y Extent (m)": f"{ext['y_extent']:.2f}",
                "Z Extent (m)": f"{ext['z_extent']:.2f}",
                "3D Box Vol (m3)": f"{ext['volume']:.2f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No objects with LiDAR frustum evidence to report.")

    # ── Export current frame report ─────────────────────────────────
    st.markdown("---")
    if st.button("Export current frame report"):
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "frame_id": selected_id,
            "detection_source": source_label,
            "max_lidar_range_m": max_distance,
            "projected_lidar_points": int(len(uv)),
            "objects": [],
        }
        for i, r in enumerate(results):
            cls_name = KITTI_CLASSES_USED[i] if i < len(KITTI_CLASSES_USED) else "unknown"
            conf = det_confidences[i] if i < len(det_confidences) else 0.0
            ext = extents[i]
            report["objects"].append({
                "index": i,
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox_2d": {
                    "left": round(float(r.bbox[0]), 1),
                    "top": round(float(r.bbox[1]), 1),
                    "right": round(float(r.bbox[2]), 1),
                    "bottom": round(float(r.bbox[3]), 1),
                },
                "median_depth_m": round(r.median_depth, 3),
                "mean_depth_m": round(r.mean_depth, 3),
                "lidar_points_in_frustum": r.num_points,
                "estimated_3d_centre": [round(float(v), 3) for v in r.center_3d],
                "extent_3d": {
                    "x_m": round(ext["x_extent"], 3),
                    "y_m": round(ext["y_extent"], 3),
                    "z_m": round(ext["z_extent"], 3),
                },
                "volume_m3": round(ext["volume"], 3),
            })
        report["context"] = {
            "camera": "RGB / video camera (KITTI image_2)",
            "lidar": "3D LiDAR / depth / spatial sensor (KITTI velodyne)",
            "calibration": "Multi-sensor calibration (KITTI calib)",
            "pipeline": "2D detection + LiDAR frustum -> 3D localisation",
        }
        out_path = out_dir / f"{selected_id}_fusion_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        st.success(f"Report saved to `{out_path}`")

    # ── Digital Inspection Robotics Interpretation ───────────────────
    st.markdown("---")
    st.subheader("Digital Inspection Robotics Interpretation")
    st.markdown(
        "| KITTI Component | Robot Equivalent |\n"
        "|---|---|\n"
        "| Left colour camera (`image_2`) | Inspection robot RGB / video camera |\n"
        "| Velodyne LiDAR (`velodyne`) | Robot 3D LiDAR / depth / spatial sensor |\n"
        "| Calibration files (`calib`) | Multi-sensor robot calibration |\n"
        "| 2D detection + LiDAR frustum filtering "
        "| First step toward 3D object / anomaly localisation |"
    )

    # ── Limitations ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Current Scope and Limitations")
    st.markdown(
        "- This is a classical perception and sensor-fusion baseline implementation.\n"
        "- It does not include robot navigation, manipulation, SLAM, or industrial validation.\n"
        "- Frustum filtering may include background LiDAR points "
        "that do not belong to the target object.\n"
        "- Depth and 3D-centre estimates are approximate and should be interpreted as "
        "engineering evidence, not certified inspection results.\n"
        "- No deep 3D detection networks are used; "
        "3D localisation relies entirely on classical geometry."
    )
