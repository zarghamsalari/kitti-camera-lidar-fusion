"""CLI script: project LiDAR points onto a KITTI camera image."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lidar_fusion.projection import run_projection


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR-to-image projection demo")
    parser.add_argument("--image", type=str, help="Path to KITTI image")
    parser.add_argument("--velodyne", type=str, help="Path to Velodyne .bin")
    parser.add_argument("--calib", type=str, help="Path to calibration .txt")
    parser.add_argument("--out", type=str, help="Output image path")
    parser.add_argument("--config", type=str, help="YAML config (overrides above)")
    args = parser.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        ds = cfg["dataset"]
        root = Path(ds["root"])
        out_dir = Path(cfg["output"]["projection_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        max_dist = cfg["lidar"]["max_distance"]

        for sid in ds["sample_ids"]:
            image = root / ds["image_dir"] / f"{sid}.png"
            velodyne = root / ds["velodyne_dir"] / f"{sid}.bin"
            calib = root / ds["calib_dir"] / f"{sid}.txt"
            out = out_dir / f"{sid}_projection.png"
            if not image.exists():
                print(f"[skip] {image} not found")
                continue
            run_projection(image, velodyne, calib, out, max_distance=max_dist)
            print(f"[done] {out}")
    else:
        if not all([args.image, args.velodyne, args.calib, args.out]):
            parser.error("Provide --image, --velodyne, --calib, --out or --config")
        run_projection(args.image, args.velodyne, args.calib, args.out)
        print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
