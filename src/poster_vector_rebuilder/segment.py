from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np
from PIL import Image
from .job_lock import exclusive_job


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _save_gray(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _robust_scale(values: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)


def deterministic_foreground_risk(rgb: np.ndarray) -> np.ndarray:
    """Return a conservative 0..1 foreground-risk field.

    This is not semantic segmentation. It is the always-available fallback used
    to identify pixels that should not be trusted as clean background samples.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hue, sat, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    sigma = max(5.0, min(h, w) * 0.018)
    smooth = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = np.linalg.norm(lab - smooth, axis=2)
    residual = _robust_scale(residual, 55, 98)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient = _robust_scale(gradient, 65, 99)
    gradient = cv2.GaussianBlur(
        gradient, (0, 0), sigmaX=max(1.5, min(h, w) * 0.002)
    )

    k = max(7, int(round(min(h, w) * 0.012)) | 1)
    mean = cv2.boxFilter(gray, -1, (k, k), normalize=True)
    mean2 = cv2.boxFilter(gray * gray, -1, (k, k), normalize=True)
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    texture = _robust_scale(local_std, 60, 98)

    # Current target posters use a blue/cyan background. This remains only one
    # cue and is never used as the sole foreground classifier.
    hue_distance = np.abs(hue - 105.0)
    hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)
    non_blue = np.clip((hue_distance - 18.0) / 30.0, 0, 1) * np.clip(sat / 80.0, 0, 1)

    neutral_bright = np.clip((value - 165.0) / 70.0, 0, 1) * np.clip((58.0 - sat) / 40.0, 0, 1)
    very_dark = np.clip((82.0 - value) / 48.0, 0, 1)

    risk = np.maximum.reduce(
        [
            0.92 * gradient,
            0.72 * texture,
            0.88 * residual,
            0.92 * non_blue,
            0.95 * neutral_bright,
            0.88 * very_dark,
        ]
    )
    risk = np.clip(
        risk
        + 0.18 * np.minimum(gradient, residual)
        + 0.12 * np.minimum(texture, non_blue),
        0,
        1,
    )
    risk = cv2.GaussianBlur(risk, (0, 0), sigmaX=max(0.8, min(h, w) * 0.0012))
    return np.clip(risk, 0, 1).astype(np.float32)


def _precision_foreground_mask(risk: np.ndarray, threshold: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    h, w = risk.shape
    raw = (risk >= threshold).astype(np.uint8) * 255

    detail_kernel = max(3, int(round(min(h, w) * 0.006)) | 1)
    detail_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detail_kernel, detail_kernel))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, detail_shape, iterations=1)
    raw = cv2.dilate(raw, detail_shape, iterations=1)

    # Precision mode intentionally bridges nearby risky fragments and fills
    # substantive clusters. This sacrifices some usable background pixels to
    # avoid accepting contamination through transparent or complex objects.
    bridge_kernel = max(9, int(round(min(h, w) * 0.018)) | 1)
    bridge_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_kernel, bridge_kernel))
    merged = cv2.dilate(raw, bridge_shape, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, bridge_shape, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    min_cluster_area = h * w * 0.002
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        if area < min_cluster_area or cw <= 30 or ch <= 20:
            continue
        if (x < 5 and cw < 40) or (x + cw > w - 5 and cw < 40):
            continue
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)

    shrink_kernel = max(3, int(round(min(h, w) * 0.006)) | 1)
    shrink_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_kernel, shrink_kernel))
    filled = cv2.erode(filled, shrink_shape, iterations=1)
    precision = cv2.bitwise_or(raw, filled)
    return raw, precision


def _birefnet_probability(rgb: np.ndarray, model_source: str, device: str | None = None) -> np.ndarray:
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("BiRefNet backend requires torch, torchvision and transformers") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageSegmentation.from_pretrained(model_source, trust_remote_code=True)
    model.to(device).eval()
    size = 1024
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    if hasattr(output, "logits"):
        pred = output.logits
    elif isinstance(output, (list, tuple)):
        pred = output[-1]
    else:
        pred = output
    pred = torch.sigmoid(pred).float().cpu().numpy()
    pred = np.squeeze(pred)
    pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(pred, 0, 1).astype(np.float32)


def _sam2_foreground_union(
    rgb: np.ndarray,
    deterministic_risk: np.ndarray,
    model_source: str | None = None,
    model_cfg: str | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_hf
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("SAM2 backend requires the facebookresearch/sam2 package and torch") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_source:
        model = build_sam2_hf(model_source, device=device)
    elif model_cfg and checkpoint:
        model = build_sam2(model_cfg, checkpoint, device=device)
    else:
        raise ValueError("SAM2 requires --sam2-model or both --sam2-config and --sam2-checkpoint")

    generator = SAM2AutomaticMaskGenerator(model)
    annotations = generator.generate(rgb)
    union = np.zeros(deterministic_risk.shape, dtype=np.float32)
    components: list[dict[str, Any]] = []

    for idx, ann in enumerate(annotations):
        seg = np.asarray(ann["segmentation"], dtype=bool)
        if not seg.any():
            continue
        area_fraction = float(seg.mean())
        samples = deterministic_risk[seg]
        mean_risk = float(samples.mean())
        p90_risk = float(np.percentile(samples, 90))
        touches = int(seg[0, :].any()) + int(seg[-1, :].any()) + int(seg[:, 0].any()) + int(seg[:, -1].any())
        score = 0.58 * mean_risk + 0.42 * p90_risk

        likely_background = area_fraction > 0.50 and touches >= 2
        accepted = (not likely_background) and score >= 0.38
        if accepted:
            confidence = float(ann.get("predicted_iou", ann.get("stability_score", 0.9)))
            union[seg] = np.maximum(union[seg], max(0.75, min(1.0, confidence)))

        components.append(
            {
                "id": idx,
                "area_fraction": round(area_fraction, 6),
                "mean_deterministic_risk": round(mean_risk, 6),
                "p90_deterministic_risk": round(p90_risk, 6),
                "touching_borders": touches,
                "selection_score": round(score, 6),
                "accepted_foreground": accepted,
            }
        )
    return union, components


def _load_manual_mask(path: str | Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


@exclusive_job('job_dir')
def segment_reference(
    job_dir: str | Path,
    image_path: str | Path | None = None,
    mode: str = "precision",
    birefnet_model: str | None = None,
    sam2_model: str | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    device: str | None = None,
    manual_foreground_mask: str | Path | None = None,
) -> dict:
    job_dir = Path(job_dir)
    image_path = Path(image_path) if image_path else job_dir / "work" / "normalized_reference.png"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if mode not in {"precision", "detail"}:
        raise ValueError("mode must be 'precision' or 'detail'")

    rgb = _load_rgb(image_path)
    h, w = rgb.shape[:2]
    masks_dir = job_dir / "masks"
    meta_dir = job_dir / "metadata"
    debug_dir = job_dir / "debug"
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    deterministic = deterministic_foreground_risk(rgb)
    combined_risk = deterministic.copy()
    backends = ["opencv-risk"]
    optional_outputs: dict[str, Any] = {}

    if birefnet_model:
        prob = _birefnet_probability(rgb, birefnet_model, device=device)
        combined_risk = np.maximum(combined_risk, 0.95 * prob)
        _save_gray(masks_dir / "birefnet_foreground_probability.png", prob)
        backends.append("birefnet")

    if sam2_model or (sam2_config and sam2_checkpoint):
        sam_prob, components = _sam2_foreground_union(
            rgb,
            deterministic,
            model_source=sam2_model,
            model_cfg=sam2_config,
            checkpoint=sam2_checkpoint,
            device=device,
        )
        combined_risk = np.maximum(combined_risk, sam_prob)
        _save_gray(masks_dir / "sam2_foreground_probability.png", sam_prob)
        (meta_dir / "sam2_components.json").write_text(json.dumps(components, indent=2), encoding="utf-8")
        backends.append("sam2")
        optional_outputs["sam2_components"] = "metadata/sam2_components.json"

    if manual_foreground_mask:
        manual = _load_manual_mask(manual_foreground_mask, w, h)
        combined_risk = np.maximum(combined_risk, manual)
        _save_gray(masks_dir / "manual_foreground_mask.png", manual)
        backends.append("manual-mask")

    raw_mask, precision_mask = _precision_foreground_mask(combined_risk)
    foreground = precision_mask if mode == "precision" else raw_mask

    distance = cv2.distanceTransform(255 - foreground, cv2.DIST_L2, 3)
    safety_margin = max(8.0, min(h, w) * 0.012)
    margin_confidence = np.clip(distance / safety_margin, 0, 1)
    confidence = np.minimum(1.0 - combined_risk, margin_confidence).astype(np.float32)
    known = (confidence >= 0.72).astype(np.uint8) * 255
    uncertain = ((known == 0) & (foreground == 0)).astype(np.uint8) * 255

    _save_gray(masks_dir / "foreground_risk.png", combined_risk)
    _save_gray(masks_dir / "foreground_detail.png", raw_mask)
    _save_gray(masks_dir / "foreground_conservative.png", precision_mask)
    _save_gray(masks_dir / "foreground_mask.png", foreground)
    _save_gray(masks_dir / "background_confidence.png", confidence)
    _save_gray(masks_dir / "background_known.png", known)
    _save_gray(masks_dir / "uncertain.png", uncertain)

    overlay = rgb.astype(np.float32)
    alpha = 0.38
    for mask, color in (
        (known > 0, np.array([0, 255, 0], dtype=np.float32)),
        (foreground > 0, np.array([255, 0, 0], dtype=np.float32)),
        (uncertain > 0, np.array([255, 210, 0], dtype=np.float32)),
    ):
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    overlay_path = debug_dir / "segmentation_overlay.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)

    result = {
        "version": 1,
        "image": str(image_path),
        "width": w,
        "height": h,
        "mode": mode,
        "backends_used": backends,
        "ratios": {
            "foreground": round(float((foreground > 0).mean()), 6),
            "background_known": round(float((known > 0).mean()), 6),
            "uncertain": round(float((uncertain > 0).mean()), 6),
        },
        "thresholds": {
            "foreground_risk": 0.42,
            "known_background_confidence": 0.72,
            "safety_margin_px": round(float(safety_margin), 3),
        },
        "outputs": {
            "foreground_risk": "masks/foreground_risk.png",
            "foreground_detail": "masks/foreground_detail.png",
            "foreground_conservative": "masks/foreground_conservative.png",
            "foreground_mask": "masks/foreground_mask.png",
            "background_confidence": "masks/background_confidence.png",
            "background_known": "masks/background_known.png",
            "uncertain": "masks/uncertain.png",
            "overlay": "debug/segmentation_overlay.png",
            **optional_outputs,
        },
        "accuracy_note": (
            "background_known is intentionally conservative. Green pixels in the overlay are approved for source measurement; "
            "red pixels are excluded foreground/risk regions; yellow pixels are uncertain and must not be used as authoritative background samples."
        ),
    }
    (meta_dir / "segmentation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
