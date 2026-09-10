from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import json
import sys

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SAM2UNeXTConfig:
    """Runtime configuration for the optional SAM2-UNeXT backend.

    The backend remains completely dormant unless ``enabled`` is True.
    ``backend_root`` should point at a checkout of WZH0120/SAM2-UNeXT and
    ``model_checkpoint`` must be a trained SAM2-UNeXT state dict.
    """

    enabled: bool = False
    backend_root: str | None = None
    model_checkpoint: str | None = None
    sam2_checkpoint: str | None = None
    dinov2_path: str | None = None
    threshold: float = 0.5
    input_resolution: int = 1024
    device: str = "auto"


class SAM2UNeXTUnavailable(RuntimeError):
    pass


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _load_backend(config: SAM2UNeXTConfig):
    if not config.backend_root:
        raise SAM2UNeXTUnavailable("SAM2-UNeXT backend_root is required when the backend is enabled")
    if not config.model_checkpoint:
        raise SAM2UNeXTUnavailable("A trained SAM2-UNeXT model checkpoint is required when the backend is enabled")

    root = Path(config.backend_root).expanduser().resolve()
    module_file = root / "SAM2UNeXT.py"
    if not module_file.is_file():
        raise SAM2UNeXTUnavailable(f"SAM2UNeXT.py not found under backend_root: {root}")

    try:
        import torch
    except Exception as exc:
        raise SAM2UNeXTUnavailable("PyTorch is not available for SAM2-UNeXT inference") from exc

    root_text = str(root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    try:
        module = importlib.import_module("SAM2UNeXT")
        model_class = module.SAM2UNeXT
    except Exception as exc:
        raise SAM2UNeXTUnavailable(f"Unable to import SAM2-UNeXT from {root}") from exc
    finally:
        if inserted and sys.path and sys.path[0] == root_text:
            sys.path.pop(0)

    device = _resolve_device(torch, config.device)
    model = model_class(
        checkpoint_path=config.sam2_checkpoint,
        dinov2_path=config.dinov2_path,
    )
    state = torch.load(str(Path(config.model_checkpoint).expanduser()), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, torch, device


def _preprocess(image: Image.Image, torch_module: Any, size: int, device: str):
    # Mirrors the upstream TestDataset pipeline: Resize -> ToTensor ->
    # ImageNet normalization.
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
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
    """Generate a binary foreground mask without mutating baseline artifacts."""

    if not config.enabled:
        return {
            "status": "disabled",
            "backend": "precision-segmentation-sam2-unext",
            "output_mask": None,
        }
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("SAM2-UNeXT threshold must be strictly between 0 and 1")
    if config.input_resolution <= 0:
        raise ValueError("SAM2-UNeXT input_resolution must be positive")

    model, torch, device = _load_backend(config)
    with Image.open(image_path) as source_image:
        source = source_image.convert("RGB")
        source_width, source_height = source.size
        tensor = _preprocess(source, torch, config.input_resolution, device)

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)
        probability = torch.nn.functional.interpolate(
            probability,
            size=(source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
        probability = probability[0, 0].detach().cpu().numpy()

    # Preserve the model's probability field separately so threshold changes can
    # be evaluated without rerunning the expensive encoders.
    out = Path(output_mask_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    probability_path = out.with_name(out.stem + "_probability.png")
    Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(probability_path)

    mask = (probability >= config.threshold).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(out)

    report = {
        "schema": "poster-vector-rebuilder.sam2-unext-segmentation.v1",
        "status": "complete",
        "backend": "precision-segmentation-sam2-unext",
        "device": device,
        "threshold": float(config.threshold),
        "input_resolution": int(config.input_resolution),
        "foreground_ratio": round(float((mask > 0).mean()), 6),
        "outputs": {
            "foreground_mask": str(out),
            "probability_map": str(probability_path),
        },
    }
    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
