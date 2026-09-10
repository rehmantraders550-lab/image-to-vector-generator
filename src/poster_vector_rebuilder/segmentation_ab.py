from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


def _binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _component_stats(mask: np.ndarray) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return {"components": 0, "small_components_le_16px": 0, "largest_component_px": 0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "components": int(len(areas)),
        "small_components_le_16px": int(np.count_nonzero(areas <= 16)),
        "largest_component_px": int(areas.max(initial=0)),
    }


def compare_masks(
    baseline_mask_path: str | Path,
    candidate_mask_path: str | Path,
    output_report: str | Path,
) -> dict:
    """Compare candidate segmentation with the current baseline without claiming GT accuracy."""

    baseline = _binary(baseline_mask_path)
    candidate = _binary(candidate_mask_path)
    if baseline.shape != candidate.shape:
        raise ValueError("Baseline and SAM2-UNeXT masks must have identical dimensions")

    intersection = np.logical_and(baseline, candidate)
    union = np.logical_or(baseline, candidate)
    changed = np.logical_xor(baseline, candidate)
    added = np.logical_and(candidate, ~baseline)
    removed = np.logical_and(baseline, ~candidate)

    # A narrow morphological edge band is useful for measuring where the two
    # systems disagree near boundaries, but it is not a ground-truth metric.
    kernel = np.ones((3, 3), dtype=np.uint8)
    base_u8 = baseline.astype(np.uint8)
    cand_u8 = candidate.astype(np.uint8)
    base_edge = cv2.morphologyEx(base_u8, cv2.MORPH_GRADIENT, kernel) > 0
    cand_edge = cv2.morphologyEx(cand_u8, cv2.MORPH_GRADIENT, kernel) > 0
    edge_union = np.logical_or(base_edge, cand_edge)
    edge_disagreement = np.logical_and(changed, cv2.dilate(edge_union.astype(np.uint8), kernel) > 0)

    union_count = int(union.sum())
    report = {
        "schema": "poster-vector-rebuilder.segmentation-ab.v1",
        "shape": [int(x) for x in baseline.shape],
        "baseline_foreground_ratio": round(float(baseline.mean()), 6),
        "candidate_foreground_ratio": round(float(candidate.mean()), 6),
        "change_ratio": round(float(changed.mean()), 6),
        "added_pixel_ratio": round(float(added.mean()), 6),
        "removed_pixel_ratio": round(float(removed.mean()), 6),
        "iou_vs_baseline": round(float(intersection.sum() / union_count), 6) if union_count else 1.0,
        "boundary_disagreement_ratio": round(float(edge_disagreement.mean()), 6),
        "baseline_components": _component_stats(baseline),
        "candidate_components": _component_stats(candidate),
        "interpretation": {
            "accuracy_claim": False,
            "reason": "No reviewed ground-truth mask was supplied; metrics quantify change from the existing baseline only.",
        },
    }
    out = Path(output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
