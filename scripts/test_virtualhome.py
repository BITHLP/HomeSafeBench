#!/usr/bin/env python3
"""Connect to a running VirtualHome instance and save one RGB image."""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reset a VirtualHome scene and save one RGB image."
    )
    parser.add_argument("--port", default="18188", help="Unity HTTP port (default: 18188)")
    parser.add_argument("--scene", type=int, default=1, help="Scene ID to reset (default: 1)")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "virtualhome_test.png",
        help="Output PNG path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vh_root = Path(os.environ.get("VH_ROOT", REPO_ROOT / "third_party" / "virtualhome"))
    simulation_dir = vh_root / "virtualhome" / "simulation"
    if not simulation_dir.is_dir():
        raise SystemExit(
            f"VirtualHome source was not found at {vh_root}. "
            "Clone it to third_party/virtualhome or set VH_ROOT."
        )

    sys.path.insert(0, str(simulation_dir))
    from unity_simulator.comm_unity import UnityCommunication

    comm = UnityCommunication(port=str(args.port), timeout_wait=30)
    if not comm.check_connection():
        raise SystemExit("Unity did not accept the connection.")
    if not comm.reset(args.scene):
        raise SystemExit(f"Failed to reset scene {args.scene}.")

    success, images = comm.camera_image([0], image_width=640, image_height=480)
    if not success or not images:
        raise SystemExit("VirtualHome did not return a camera image.")

    import cv2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), images[0]):
        raise SystemExit(f"Failed to write {args.output}.")
    height, width = images[0].shape[:2]
    try:
        output_label = args.output.relative_to(REPO_ROOT)
    except ValueError:
        output_label = args.output
    print(f"Saved RGB image to {output_label} ({width}x{height})")


if __name__ == "__main__":
    main()
