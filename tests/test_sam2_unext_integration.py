from pathlib import Path

import numpy as np
from PIL import Image

from poster_vector_rebuilder.precision_segmentation_sam2_unext import (
    SAM2UNeXTConfig,
    infer_precision_foreground,
)
from poster_vector_rebuilder.segmentation_ab import compare_masks


def test_disabled_backend_is_zero_cost(tmp_path: Path):
    result = infer_precision_foreground(
        tmp_path / "missing.png",
        tmp_path / "mask.png",
        config=SAM2UNeXTConfig(enabled=False),
    )
    assert result["status"] == "disabled"
    assert result["output_mask"] is None
    assert not (tmp_path / "mask.png").exists()


def test_ab_metrics_are_diagnostic_not_accuracy_claim(tmp_path: Path):
    baseline = np.zeros((16, 16), dtype=np.uint8)
    baseline[2:10, 2:10] = 255
    candidate = baseline.copy()
    candidate[8:14, 8:14] = 255

    baseline_path = tmp_path / "baseline.png"
    candidate_path = tmp_path / "candidate.png"
    report_path = tmp_path / "ab.json"
    Image.fromarray(baseline, mode="L").save(baseline_path)
    Image.fromarray(candidate, mode="L").save(candidate_path)

    report = compare_masks(baseline_path, candidate_path, report_path)
    assert report_path.is_file()
    assert report["interpretation"]["accuracy_claim"] is False
    assert report["change_ratio"] > 0
    assert 0 < report["iou_vs_baseline"] < 1
    assert report["candidate_components"]["components"] >= 1
