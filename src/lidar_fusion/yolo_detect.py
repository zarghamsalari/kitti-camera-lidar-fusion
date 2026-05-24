"""Run YOLOv8 2D detection on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detect_objects(
    image_path: str | Path,
    weights: str = "yolov8m.pt",
    conf: float = 0.25,
    iou: float = 0.5,
    classes: list[int] | None = None,
    device: str = "",
) -> np.ndarray:
    """Run YOLOv8 inference and return detections as Nx6 array.

    Each row: [x1, y1, x2, y2, confidence, class_id].
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        classes=classes,
        device=device or None,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6))

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().reshape(-1, 1)

    return np.hstack([xyxy, confs, cls])
