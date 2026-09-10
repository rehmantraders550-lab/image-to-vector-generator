from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from .generalized_preflight import generic_foreground_risk
from .sam2_unext_adapter import binary_mask_from_probability, boundary_map, sam2_unext_probability


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _baseline_mask(rgb: np.ndarray, threshold: float = 0.44) -> tuple[np.ndarray, np.ndarray]:
    risk = generic_foreground_risk(rgb)
    raw = (risk >= threshold).astype(np.uint8) * 255
    h, w = raw.shape
    detail_k = max(3, int(round(min(h, w) * 0.005)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detail_k, detail_k))
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return risk, mask


def _component_stats(mask: np.ndarray) -> dict:
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int32)
    return {
        "components": int(max(0, count - 1)),
        "tiny_components_lt_25px": int(np.count_nonzero(areas < 25)),
        "median_component_area": float(np.median(areas)) if areas.size else 0.0,
        "foreground_ratio": round(float(binary.mean()), 6),
    }


def _mask_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    aa = a > 0
    bb = b > 0
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    iou = inter / union if union else 1.0
    changed = float(np.mean(aa != bb))

    ba = boundary_map(a) > 0
    bbnd = boundary_map(b) > 0
    boundary_union = int(np.count_nonzero(ba | bbnd))
    boundary_inter = int(np.count_nonzero(ba & bbnd))
    boundary_iou = boundary_inter / boundary_union if boundary_union else 1.0
    return {
        "mask_iou": round(float(iou), 6),
        "changed_pixel_ratio": round(changed, 6),
        "boundary_iou": round(float(boundary_iou), 6),
    }


def run_segmentation_ab(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    sam2_unext_repo: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
    model_threshold: float = 0.5,
) -> dict:
    """Run baseline vs SAM2-UNeXT segmentation without altering production masks."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    baseline_risk, baseline_mask = _baseline_mask(rgb)
    model_probability = sam2_unext_probability(
        rgb,
        repo_path=sam2_unext_repo,
        checkpoint=checkpoint,
        device=device,
    )
    model_mask = binary_mask_from_probability(model_probability, threshold=model_threshold)

    _save_gray(output_dir / "baseline_risk.png", baseline_risk)
    _save_gray(output_dir / "baseline_mask.png", baseline_mask)
    _save_gray(output_dir / "sam2_unext_probability.png", model_probability)
    _save_gray(output_dir / "sam2_unext_mask.png", model_mask)

    # Side-by-side diagnostic only. No production file is overwritten.
    overlay = rgb.astype(np.float32)
    baseline_edge = boundary_map(baseline_mask) > 0
    model_edge = boundary_map(model_mask) > 0
    overlay[baseline_edge] = np.array([255, 80, 0], dtype=np.float32)
    overlay[model_edge] = np.array([0, 255, 255], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_dir / "boundary_overlay.png")

    report = {
        "schema": "poster-vector-rebuilder.sam2-unext-ab.v1",
        "source": str(image_path),
        "baseline": _component_stats(baseline_mask),
        "sam2_unext": _component_stats(model_mask),
        "comparison": _mask_comparison(baseline_mask, model_mask),
        "model_threshold": model_threshold,
        "production_baseline_modified": False,
        "outputs": {
            "baseline_risk": str(output_dir / "baseline_risk.png"),
            "baseline_mask": str(output_dir / "baseline_mask.png"),
            "sam2_unext_probability": str(output_dir / "sam2_unext_probability.png"),
            "sam2_unext_mask": str(output_dir / "sam2_unext_mask.png"),
            "boundary_overlay": str(output_dir / "boundary_overlay.png"),
            "report": str(output_dir / "ab_report.json"),
        },
    }
    (output_dir / "ab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
