from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from poster_vector_rebuilder.precision_segmentation_sam2_unext import (
    SAM2UNeXTConfig,
    SAM2UNeXTUnavailable,
    compare_masks,
    infer_precision_foreground,
)


def test_disabled_backend_does_not_import_or_write(tmp_path: Path):
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "white").save(image)
    output = tmp_path / "mask.png"

    result = infer_precision_foreground(
        image,
        output,
        config=SAM2UNeXTConfig(enabled=False),
    )

    assert result["status"] == "disabled"
    assert result["output_mask"] is None
    assert not output.exists()


def test_enabled_backend_requires_checkpoint(tmp_path: Path):
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "white").save(image)

    with pytest.raises(SAM2UNeXTUnavailable):
        infer_precision_foreground(
            image,
            tmp_path / "mask.png",
            config=SAM2UNeXTConfig(enabled=True, checkpoint_path=None),
        )


def test_compare_masks_reports_delta_without_accuracy_claim(tmp_path: Path):
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[:, :5] = 255
    b[:, 2:7] = 255
    pa = tmp_path / "a.png"
    pb = tmp_path / "b.png"
    Image.fromarray(a).save(pa)
    Image.fromarray(b).save(pb)

    result = compare_masks(pa, pb)

    assert result["accuracy_claim"] is False
    assert result["change_ratio"] == pytest.approx(0.4)
    assert result["added_pixel_ratio"] == pytest.approx(0.2)
    assert result["removed_pixel_ratio"] == pytest.approx(0.2)
