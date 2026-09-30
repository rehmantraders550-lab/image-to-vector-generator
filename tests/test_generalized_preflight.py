from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from poster_vector_rebuilder.generalized_preflight import (
    generic_foreground_risk,
    normalize_any_artwork,
    run_blocks_1_to_4,
)
from poster_vector_rebuilder.intake_classify import classify_artwork


def _fixture(path: Path, base=(210, 70, 45)) -> Path:
    h, w = 240, 360
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(base[0] + 20 * x / w, 0, 255)
    rgb[..., 1] = np.clip(base[1] + 35 * y / h, 0, 255)
    rgb[..., 2] = np.clip(base[2] + 15 * x / w, 0, 255)
    cv2.rectangle(rgb, (58, 48), (150, 128), (245, 235, 40), -1)
    cv2.circle(rgb, (260, 125), 42, (30, 45, 210), -1)
    cv2.line(rgb, (15, 205), (345, 170), (245, 245, 245), 6)
    Image.fromarray(rgb).save(path)
    return path


def test_color_agnostic_risk_responds_to_edges_on_non_blue_artwork(tmp_path):
    image = np.array(Image.open(_fixture(tmp_path / "art.png")))
    risk = generic_foreground_risk(image)
    assert risk.shape == image.shape[:2]
    assert 0.0 <= float(risk.min()) <= float(risk.max()) <= 1.0
    assert float(risk[48:130, 58:152].mean()) > float(risk[5:35, 5:35].mean())


def test_classifier_is_generic_and_emits_routes(tmp_path):
    path = _fixture(tmp_path / "art.png", base=(60, 155, 75))
    report = classify_artwork(path)
    assert report["primary_class"] in {"smooth_composite", "hard_graphic_composite", "mixed_or_photographic"}
    assert set(report["routes"]) == {
        "background_gradient_fit",
        "panel_detection",
        "hard_graphic_vectorization",
        "photographic_fallback_possible",
    }


def test_normalize_any_artwork_accepts_native_full_frame_raster(tmp_path):
    path = _fixture(tmp_path / "art.png")
    job = tmp_path / "job"
    result = normalize_any_artwork(path, job)
    assert (job / result["normalized_path"]).exists()
    assert (job / "metadata" / "geometry.json").exists()


def test_blocks_1_to_4_produce_manifest_and_masks(tmp_path):
    path = _fixture(tmp_path / "art.png")
    job = tmp_path / "job"
    result = run_blocks_1_to_4(path, job, max_panels=3)
    assert result["status"] == "complete"
    assert all(v["status"] == "complete" for v in result["blocks"].values())
    assert (job / "masks" / "foreground_mask.png").exists()
    assert (job / "masks" / "background_known.png").exists()
    manifest = job / "metadata" / "blocks_1_to_4.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["schema"] == "poster-vector-rebuilder.blocks-1-4.v2"
