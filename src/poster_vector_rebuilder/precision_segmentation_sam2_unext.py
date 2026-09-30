from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import json
import sys

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SAM2UNeXTConfig:
    enabled: bool = False
    backend_root: str | None = None
    checkpoint_path: str | None = None
    threshold: float = 0.5
    input_resolution: int = 1024
    device: str = "auto"


class SAM2UNeXTUnavailable(RuntimeError):
    pass


def _resolve_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _load_backend(cfg: SAM2UNeXTConfig):
    if cfg.backend_root:
        root = str(Path(cfg.backend_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        torch = importlib.import_module("torch")
        module = importlib.import_module("SAM2UNeXT")
        model_cls = getattr(module, "SAM2UNeXT")
    except Exception as exc:
        raise SAM2UNeXTUnavailable(
            "SAM2-UNeXT backend is not importable. Configure backend_root or install the upstream backend."
        ) from exc

    if not cfg.checkpoint_path:
        raise SAM2UNeXTUnavailable("A trained SAM2-UNeXT checkpoint is required for inference.")

    checkpoint = Path(cfg.checkpoint_path)
    if not checkpoint.is_file():
        raise SAM2UNeXTUnavailable(f"SAM2-UNeXT checkpoint not found: {checkpoint}")

    device = _resolve_device(torch, cfg.device)
    model = model_cls().to(device)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, torch, device


def _preprocess(image: Image.Image, torch_module, size: int, device: str):
    # Matches the upstream TestDataset transform: Resize -> ToTensor -> ImageNet Normalize.
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch_module.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def infer_precision_foreground(
    image_path: str | Path,
    output_mask_path: str | Path,
    *,
    config: SAM2UNeXTConfig,
    report_path: str | Path | None = None,
) -> dict:
    """Run SAM2-UNeXT without modifying the generator's baseline foreground mask."""
    if not config.enabled:
        return {"status": "disabled", "backend": "precision-segmentation-sam2-unext", "output_mask": None}
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("SAM2-UNeXT threshold must be between 0 and 1.")

    model, torch, device = _load_backend(config)
    with Image.open(image_path) as im:
        source = im.convert("RGB")
        source_size = source.size
        tensor = _preprocess(source, torch, config.input_resolution, device)

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)
        probability = torch.nn.functional.interpolate(
            probability,
            size=(source_size[1], source_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        mask = (probability[0, 0] >= config.threshold).to(torch.uint8).cpu().numpy() * 255

    output = Path(output_mask_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output)

    report = {
        "schema": "poster-vector-rebuilder.sam2-unext-segmentation.v1",
        "status": "complete",
        "backend": "precision-segmentation-sam2-unext",
        "device": device,
        "threshold": config.threshold,
        "input_resolution": config.input_resolution,
        "foreground_ratio": round(float((mask > 0).mean()), 6),
        "output_mask": str(output),
    }
    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_masks(baseline_mask_path: str | Path, candidate_mask_path: str | Path) -> dict:
    """Ground-truth-free A/B delta metrics; does not claim segmentation accuracy."""
    baseline = np.asarray(Image.open(baseline_mask_path).convert("L")) >= 128
    candidate = np.asarray(Image.open(candidate_mask_path).convert("L")) >= 128
    if baseline.shape != candidate.shape:
        raise ValueError("Baseline and SAM2-UNeXT masks must have identical dimensions.")
    intersection = np.logical_and(baseline, candidate).sum()
    union = np.logical_or(baseline, candidate).sum()
    return {
        "baseline_foreground_ratio": round(float(baseline.mean()), 6),
        "candidate_foreground_ratio": round(float(candidate.mean()), 6),
        "change_ratio": round(float(np.logical_xor(baseline, candidate).mean()), 6),
        "iou_vs_baseline": round(float(intersection / union), 6) if union else 1.0,
        "added_pixel_ratio": round(float(np.logical_and(candidate, ~baseline).mean()), 6),
        "removed_pixel_ratio": round(float(np.logical_and(baseline, ~candidate).mean()), 6),
        "accuracy_claim": False,
    }
