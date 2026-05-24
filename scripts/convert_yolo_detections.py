"""Convert external YOLO detection outputs into the standard detections.json
format used by the camera-LiDAR fusion dashboard.

Accepts CSV or JSON input from any YOLO-based detection pipeline (e.g. a
separate KITTI tracking project).  Writes a normalised JSON file that the
Streamlit app and fusion scripts can consume directly.

Input formats supported
-----------------------
**CSV** -- must contain a header row.  Recognised column names (case-insensitive,
underscores and hyphens interchangeable):

    frame_id, class_name, confidence, xmin, ymin, xmax, ymax

or the alternative bbox-list style used by some YOLO export tools:

    frame_id, class_name, confidence, x1, y1, x2, y2

**JSON** -- a list of objects.  Each object must have ``frame_id``,
``class_name``, ``confidence``, and bounding-box coordinates as either
separate keys (``xmin/ymin/xmax/ymax`` or ``x1/y1/x2/y2``) or a single
``bbox`` key containing ``[xmin, ymin, xmax, ymax]``.

Output format
-------------
::

    [
      {
        "frame_id": "000000",
        "class_name": "person",
        "confidence": 0.91,
        "bbox": [xmin, ymin, xmax, ymax]
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# ── Normalisation helpers ────────────────────────────────────────────

def _normalise_key(key: str) -> str:
    """Lower-case and replace hyphens with underscores."""
    return key.strip().lower().replace("-", "_")


def _extract_bbox(record: dict) -> list[float]:
    """Extract [xmin, ymin, xmax, ymax] from a record, trying several layouts."""
    # 1) "bbox" key as a list
    if "bbox" in record:
        bbox = record["bbox"]
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        return [float(v) for v in bbox[:4]]

    # 2) xmin / ymin / xmax / ymax
    for keys in [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x1", "y1", "x2", "y2"),
    ]:
        if all(k in record for k in keys):
            return [float(record[k]) for k in keys]

    raise KeyError(
        f"Cannot find bounding-box coordinates in record keys: {list(record.keys())}"
    )


def _normalise_record(raw: dict) -> dict:
    """Convert an arbitrary detection record into the standard format."""
    # Normalise all keys first
    norm = {_normalise_key(k): v for k, v in raw.items()}

    frame_id = str(norm.get("frame_id", norm.get("frame", "000000")))
    # Zero-pad to 6 digits if purely numeric
    if frame_id.isdigit():
        frame_id = frame_id.zfill(6)

    class_name = str(norm.get("class_name", norm.get("class", "unknown")))
    confidence = float(norm.get("confidence", norm.get("conf", 0.0)))
    bbox = _extract_bbox(norm)

    return {
        "frame_id": frame_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "bbox": [round(v, 2) for v in bbox],
    }


# ── Loaders ──────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(_normalise_record(row))
    return records


def _load_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # Some tools wrap detections under a key like "detections"
        for key in ("detections", "results", "predictions"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                "JSON file is a dict but does not contain a recognised "
                "detections list key (detections / results / predictions)."
            )
    return [_normalise_record(r) for r in raw]


def load_detections(path: Path) -> list[dict]:
    """Load detections from a CSV or JSON file."""
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .csv or .json)")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO detection outputs into the standard "
                    "detections.json format for the fusion dashboard.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input detections file (.csv or .json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="outputs/detections.json",
        help="Output path (default: outputs/detections.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    detections = load_detections(input_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    # Summary
    frame_ids = sorted({d["frame_id"] for d in detections})
    classes = sorted({d["class_name"] for d in detections})
    print()
    print(f"  Converted {len(detections)} detections from {input_path}")
    print(f"  Frames   : {len(frame_ids)} ({frame_ids[0]} .. {frame_ids[-1]})"
          if frame_ids else "  Frames   : 0")
    print(f"  Classes  : {', '.join(classes)}" if classes else "  Classes  : none")
    print(f"  Output   : {out_path}")
    print()


if __name__ == "__main__":
    main()
