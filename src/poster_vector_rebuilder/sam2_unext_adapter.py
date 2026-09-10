from __future__ import annotations

from pathlib import Path
import importlib
import sys
from typing import Any

import cv2
import numpy as np


def _resolve_device(torch_module: Any, device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _load_external_model_class(repo_path: str | Path):
    repo = Path(repo_path).expanduser().resolve()
    module_file = repo / "SAM2UNeXT.py"
    if not module_file.exists():
        raise FileNotFoundError(f"SAM2-UNeXT source not found: {module_file}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    module = importlib.import_module("SAM2UNeXT")
    model_class = getattr(module, "SAM2UNeXT", None)
    if model_class is None:
        raise RuntimeError("SAM2UNeXT.py does not expose SAM2UNeXT")
    return model_class


def sam2_unext_probability(
    rgb: np.ndarray,
    *,
    repo_path: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
) -> np.ndarray:
    """Run the upstream SAM2-UNeXT model and return a source-size 0..1 map.

    This adapter intentionally keeps SAM2-UNeXT isolated from the baseline
    pipeline. It follows the upstream test-time preprocessing: resize the RGB
    image to 1024x1024, ImageNet-normalize, run one forward pass, bilinear
    upsample logits to source size, sigmoid, then min/max normalize.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("SAM2-UNeXT requires PyTorch") from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 RGB array")

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model_class = _load_external_model_class(repo_path)
    selected_device = _resolve_device(torch, device)

    model = model_class().to(selected_device)
    state = torch.load(str(checkpoint), map_location=selected_device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    image = (image - mean) / std
    image = F.interpolate(image.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False)
    image = image.to(selected_device)

    h, w = rgb.shape[:2]
    with torch.inference_mode():
        logits = model(image)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        probability = torch.sigmoid(logits).float().cpu().numpy().squeeze()

    lo = float(probability.min())
    hi = float(probability.max())
    if hi > lo + 1e-8:
        probability = (probability - lo) / (hi - lo)
    else:
        probability = np.zeros((h, w), dtype=np.float32)

    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def binary_mask_from_probability(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if probability.ndim != 2:
        raise ValueError("probability must be a 2D array")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within 0..1")
    return (probability >= threshold).astype(np.uint8) * 255


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Return a 1px morphological boundary for deterministic A/B metrics."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return (binary - eroded).astype(np.uint8)
