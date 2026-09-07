from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

from .normalize import normalize_reference, preserve_source
from .job_lock import exclusive_job
from .intake_classify import classify_artwork
from .panel_detect import run_phase24b


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _full_frame_normalize(input_path: str | Path, job_dir: Path, reason: str) -> dict:
    preserved, source_meta = preserve_source(input_path, job_dir)
    with Image.open(preserved) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    work = job_dir / "work"
    meta = job_dir / "metadata"
    debug = job_dir / "debug"
    work.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    debug.mkdir(parents=True, exist_ok=True)
    normalized = work / "normalized_reference.png"
    Image.fromarray(rgb).save(normalized)
    geometry = {
        "version": 1,
        "source": source_meta,
        "quad_detection": {"method": "full_frame_fallback", "score": None, "points_tl_tr_br_bl": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]},
        "source_quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        "perspective_matrix": np.eye(3).tolist(),
        "rectified_width": w,
        "rectified_height": h,
        "rotation": "keep",
        "rotation_matrix": np.eye(3).tolist(),
        "normalized_width": w,
        "normalized_height": h,
        "normalized_path": "work/normalized_reference.png",
        "fallback_reason": reason,
        "accuracy_note": "No photographed sheet boundary was required or reliably detected; the complete raster canvas is treated as the artwork coordinate system.",
    }
    (meta / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    return geometry


@exclusive_job('job_dir')
def normalize_any_artwork(input_path: str | Path, job_dir: str | Path) -> dict:
    job = Path(job_dir)
    try:
        result = normalize_reference(input_path, job)
        # A detected quad that discards too much of a native raster is unsafe.
        src = result.get("source", {})
        sw, sh = int(src.get("width", result["normalized_width"])), int(src.get("height", result["normalized_height"]))
        kept = (result["normalized_width"] * result["normalized_height"]) / max(sw * sh, 1)
        if kept < 0.58:
            return _full_frame_normalize(input_path, job, f"detected quad retained only {kept:.3f} of raster area")
        return result
    except Exception as exc:
        return _full_frame_normalize(input_path, job, f"automatic page detection unavailable: {type(exc).__name__}: {exc}")


def generic_foreground_risk(rgb: np.ndarray) -> np.ndarray:
    """Colour-agnostic foreground/detail risk for arbitrary artwork.

    The field identifies pixels unsafe for smooth-background measurement using
    multiscale residual, edge energy and local texture. No target hue is encoded.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    sigma1 = max(3.0, min(h, w) * 0.010)
    sigma2 = max(8.0, min(h, w) * 0.030)
    smooth1 = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma1)
    smooth2 = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma2)
    fine = np.linalg.norm(lab - smooth1, axis=2)
    coarse = np.linalg.norm(smooth1 - smooth2, axis=2)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)

    k = max(7, int(round(min(h, w) * 0.012)) | 1)
    mean = cv2.boxFilter(gray, -1, (k, k), normalize=True)
    mean2 = cv2.boxFilter(gray * gray, -1, (k, k), normalize=True)
    texture = np.sqrt(np.maximum(mean2 - mean * mean, 0))

    def robust(v: np.ndarray, lo: float = 55, hi: float = 98) -> np.ndarray:
        a, b = float(np.percentile(v, lo)), float(np.percentile(v, hi))
        if b <= a + 1e-6:
            return np.zeros_like(v, dtype=np.float32)
        return np.clip((v - a) / (b - a), 0, 1).astype(np.float32)

    rf, rc, re, rt = robust(fine), robust(coarse), robust(edge, 65, 99), robust(texture, 60, 98)
    risk = np.maximum.reduce([0.94 * re, 0.86 * rf, 0.72 * rt, 0.58 * rc])
    risk = np.clip(risk + 0.18 * np.minimum(re, rf) + 0.10 * np.minimum(rt, rf), 0, 1)
    return cv2.GaussianBlur(risk.astype(np.float32), (0, 0), sigmaX=max(0.8, min(h, w) * 0.0012))


@exclusive_job('job_dir')
def separate_foreground_background(image_path: str | Path, job_dir: str | Path) -> dict:
    job = Path(job_dir)
    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    risk = generic_foreground_risk(rgb)
    raw = (risk >= 0.44).astype(np.uint8) * 255

    detail_k = max(3, int(round(min(h, w) * 0.005)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detail_k, detail_k))
    fg = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=1)
    fg = cv2.dilate(fg, kernel, iterations=1)

    dist = cv2.distanceTransform(255 - fg, cv2.DIST_L2, 3)
    margin = max(7.0, min(h, w) * 0.010)
    confidence = np.minimum(1.0 - risk, np.clip(dist / margin, 0, 1))
    known = (confidence >= 0.70).astype(np.uint8) * 255
    uncertain = ((known == 0) & (fg == 0)).astype(np.uint8) * 255

    masks = job / "masks"
    debug = job / "debug"
    meta = job / "metadata"
    for p in (masks, debug, meta):
        p.mkdir(parents=True, exist_ok=True)
    _save_gray(masks / "foreground_risk.png", risk)
    _save_gray(masks / "foreground_mask.png", fg)
    _save_gray(masks / "background_confidence.png", confidence)
    _save_gray(masks / "background_known.png", known)
    _save_gray(masks / "uncertain.png", uncertain)

    overlay = rgb.astype(np.float32)
    alpha = 0.35
    for mask, color in ((known > 0, np.array([0, 255, 0], np.float32)), (fg > 0, np.array([255, 0, 0], np.float32)), (uncertain > 0, np.array([255, 210, 0], np.float32))):
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(debug / "generalized_segmentation_overlay.png")

    report = {
        "schema": "poster-vector-rebuilder.generic-segmentation.v1",
        "backends_used": ["opencv-color-agnostic-risk"],
        "ratios": {
            "foreground": round(float((fg > 0).mean()), 6),
            "background_known": round(float((known > 0).mean()), 6),
            "uncertain": round(float((uncertain > 0).mean()), 6),
        },
        "thresholds": {"foreground_risk": 0.44, "known_background_confidence": 0.70, "safety_margin_px": round(float(margin), 3)},
        "outputs": {
            "foreground_risk": "masks/foreground_risk.png",
            "foreground_mask": "masks/foreground_mask.png",
            "background_confidence": "masks/background_confidence.png",
            "background_known": "masks/background_known.png",
            "uncertain": "masks/uncertain.png",
            "overlay": "debug/generalized_segmentation_overlay.png",
        },
        "rules": {"brand_color_assumptions": False, "visible_background_only_is_authoritative": True},
    }
    (meta / "generalized_segmentation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@exclusive_job('job_dir')
def run_blocks_1_to_4(input_path: str | Path, job_dir: str | Path, *, max_panels: int = 4) -> dict:
    """Run the generalized preparation blocks needed before vector reconstruction.

    1. Raster intake + coordinate normalization
    2. Generic artwork classification/routing
    3. Foreground/background confidence separation
    4. Region/panel boundary detection on authoritative background pixels
    """
    job = Path(job_dir)
    job.mkdir(parents=True, exist_ok=True)
    geometry = normalize_any_artwork(input_path, job)
    normalized = job / geometry["normalized_path"]
    classification = classify_artwork(normalized, job / "metadata" / "artwork_classification.json")
    segmentation = separate_foreground_background(normalized, job)

    panel_error = None
    try:
        panels = run_phase24b(job, image_path=normalized, background_known_path=job / "masks" / "background_known.png", max_panels=max_panels)
    except Exception as exc:
        panel_error = f"{type(exc).__name__}: {exc}"
        panels = {
            "schema": "poster-vector-rebuilder.phase24b.skipped.v1",
            "boundary_count": 0,
            "panel_hypothesis_count": 0,
            "model_selection": None,
            "status": "not_applicable_or_insufficient_support",
            "error": panel_error,
        }

    result = {
        "schema": "poster-vector-rebuilder.blocks-1-4.v1",
        "status": "complete",
        "blocks": {
            "1_raster_intake_normalization": {"status": "complete", "method": geometry["quad_detection"]["method"]},
            "2_artwork_classification": {"status": "complete", "primary_class": classification["primary_class"], "routes": classification["routes"]},
            "3_foreground_background_separation": {"status": "complete", "ratios": segmentation["ratios"]},
            "4_region_panel_detection": {"status": "complete", "boundary_count": panels.get("boundary_count", 0), "panel_hypothesis_count": panels.get("panel_hypothesis_count", 0), "nonfatal_note": panel_error},
        },
        "outputs": {
            "normalized_reference": str(normalized),
            "geometry": str(job / "metadata" / "geometry.json"),
            "classification": str(job / "metadata" / "artwork_classification.json"),
            "segmentation": str(job / "metadata" / "generalized_segmentation.json"),
            "panel_report": panels.get("outputs", {}).get("report") if isinstance(panels.get("outputs"), dict) else None,
        },
    }
    manifest = job / "metadata" / "blocks_1_to_4.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["outputs"]["manifest"] = str(manifest)
    return result
