"""Streamlit demo for KITTI camera-LiDAR fusion."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="KITTI Camera-LiDAR Fusion", layout="wide")
st.title("KITTI Camera-LiDAR Fusion Demo")

# ── Dataset paths ────────────────────────────────────────────────────
DATA_ROOT = Path("data/kitti3d/training")
IMAGE_DIR = DATA_ROOT / "image_2"
VELODYNE_DIR = DATA_ROOT / "velodyne"
CALIB_DIR = DATA_ROOT / "calib"

if not DATA_ROOT.exists():
    st.error(
        "KITTI 3D data not found. Please download and extract to `data/kitti3d/training/` "
        "with subdirectories: `image_2/`, `velodyne/`, `calib/`.\n\n"
        "Download from: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"
    )
    st.stop()

# ── Frame selector ───────────────────────────────────────────────────
image_files = sorted(IMAGE_DIR.glob("*.png"))
if not image_files:
    st.warning("No images found in data/kitti3d/training/image_2/")
    st.stop()

frame_ids = [f.stem for f in image_files]
selected_id = st.sidebar.selectbox("Select frame", frame_ids)

image_path = IMAGE_DIR / f"{selected_id}.png"
velodyne_path = VELODYNE_DIR / f"{selected_id}.bin"
calib_path = CALIB_DIR / f"{selected_id}.txt"

for p, name in [(image_path, "Image"), (velodyne_path, "Velodyne"), (calib_path, "Calib")]:
    if not p.exists():
        st.error(f"{name} file missing: {p}")
        st.stop()

# ── Sidebar controls ────────────────────────────────────────────────
max_distance = st.sidebar.slider("Max LiDAR range (m)", 10.0, 120.0, 80.0, 5.0)
run_yolo = st.sidebar.checkbox("Run YOLO detection", value=True)
yolo_conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.25, 0.05)

# ── Load and process ────────────────────────────────────────────────
import cv2
import numpy as np

from lidar_fusion.calibration import load_calibration, project_lidar_to_image
from lidar_fusion.pointcloud import filter_by_range, filter_forward_points, load_velodyne_bin, make_bev_plot
from lidar_fusion.projection import draw_lidar_on_image

image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

points = load_velodyne_bin(velodyne_path)
points = filter_forward_points(points)
points = filter_by_range(points, max_distance)

calib = load_calibration(calib_path)
uv, depth, valid_mask = project_lidar_to_image(points, calib)

# ── Tab 1: Camera image ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Camera", "LiDAR Projection", "Frustum Filtering", "BEV"])

with tab1:
    st.image(image_rgb, caption=f"Frame {selected_id}", use_container_width=True)

# ── Tab 2: LiDAR projection ─────────────────────────────────────────
with tab2:
    proj_img = draw_lidar_on_image(image, uv, depth)
    proj_rgb = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
    st.image(proj_rgb, caption="LiDAR projected onto camera", use_container_width=True)
    st.metric("Projected points", len(uv))

# ── Tab 3: Frustum filtering ────────────────────────────────────────
with tab3:
    if run_yolo:
        from lidar_fusion.frustum import frustum_filter
        from lidar_fusion.visualisation import draw_boxes_with_lidar
        from lidar_fusion.yolo_detect import detect_objects

        detections = detect_objects(image_path, conf=yolo_conf)
        st.info(f"YOLO detected {len(detections)} objects")

        results = []
        min_pts = 3
        for det in detections:
            res = frustum_filter(points, uv, valid_mask, det)
            if res.num_points >= min_pts:
                results.append(res)

        box_img = draw_boxes_with_lidar(image, uv, depth, results, valid_mask, points)
        box_rgb = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
        st.image(box_rgb, caption="Detections + filtered LiDAR", use_container_width=True)

        if results:
            st.subheader("Depth estimates")
            for i, r in enumerate(results):
                cls_name = {0: "person", 2: "car"}.get(int(r.bbox[5]), f"cls{int(r.bbox[5])}")
                st.write(
                    f"**Box {i}** ({cls_name}): "
                    f"depth={r.median_depth:.1f}m, "
                    f"points={r.num_points}, "
                    f"centre=({r.center_3d[0]:.1f}, {r.center_3d[1]:.1f}, {r.center_3d[2]:.1f})"
                )
    else:
        st.info("Enable 'Run YOLO detection' in the sidebar to see frustum filtering.")

# ── Tab 4: Bird's-eye view ──────────────────────────────────────────
with tab4:
    if run_yolo and "results" in dir() and results:
        from lidar_fusion.visualisation import make_frustum_bev
        bev_fig = make_frustum_bev(points, results)
    else:
        bev_fig = make_bev_plot(points)
    st.pyplot(bev_fig)
