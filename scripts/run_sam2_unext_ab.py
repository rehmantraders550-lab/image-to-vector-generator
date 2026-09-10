#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from poster_vector_rebuilder.segmentation_ab import run_segmentation_ab


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline segmentation against SAM2-UNeXT without modifying production masks")
    parser.add_argument("image", help="Input/reference image")
    parser.add_argument("-o", "--output", required=True, help="Output directory for A/B diagnostics")
    parser.add_argument("--sam2-unext-repo", required=True, help="Local clone of WZH0120/SAM2-UNeXT")
    parser.add_argument("--checkpoint", required=True, help="Trained SAM2-UNeXT checkpoint")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda, cuda:0 or cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="SAM2-UNeXT probability threshold")
    args = parser.parse_args()

    report = run_segmentation_ab(
        args.image,
        args.output,
        sam2_unext_repo=args.sam2_unext_repo,
        checkpoint=args.checkpoint,
        device=args.device,
        model_threshold=args.threshold,
    )
    print(Path(report["outputs"]["report"]))


if __name__ == "__main__":
    main()
