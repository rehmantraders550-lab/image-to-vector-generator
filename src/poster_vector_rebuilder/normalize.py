from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import hashlib
import json
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
from .job_lock import exclusive_job

Rotation = Literal["keep", "90cw", "90ccw", "180"]


@dataclass(frozen=True)
class QuadDetection:
    points: np.ndarray
    method: str
    score: float


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_metadata(path: Path) -> dict:
    with Image.open(path) as im:
        exif = im.getexif()
        return {
            "filename": path.name,
            "format": im.format,
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
            "dpi": list(im.info.get("dpi", ())) or None,
            "exif_orientation": exif.get(274),
            "icc_profile_present": bool(im.info.get("icc_profile")),
            "sha256": sha256_file(path),
        }


@exclusive_job('job_dir')
def preserve_source(input_path: str | Path, job_dir: str | Path) -> tuple[Path, dict]:
    input_path = Path(input_path)
    job_dir = Path(job_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    source_dir = job_dir / "input"
    meta_dir = job_dir / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower() or ".bin"
    preserved = source_dir / f"source_original{suffix}"
    shutil.copyfile(input_path, preserved)

    metadata = _source_metadata(preserved)
    (meta_dir / "source_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return preserved, metadata


def _load_exif_transposed_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _quad_score(quad: np.ndarray, image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = area / float(w * h)
    if area_ratio < 0.15 or area_ratio > 1.02:
        return -1e9

    q = order_quad(quad)
    widths = [np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])]
    heights = [np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])]
    if min(widths + heights) < min(w, h) * 0.12:
        return -1e9

    width_balance = min(widths) / max(widths)
    height_balance = min(heights) / max(heights)
    rect = cv2.minAreaRect(q.astype(np.float32))
    box_area = max(rect[1][0] * rect[1][1], 1.0)
    rectangularity = min(area / box_area, 1.0)
    centroid = q.mean(axis=0)
    centre = np.array([w / 2, h / 2], dtype=np.float32)
    centre_dist = np.linalg.norm((centroid - centre) / np.array([w, h]))
    centre_score = max(0.0, 1.0 - centre_dist * 2.5)

    return (
        area_ratio * 6.0
        + rectangularity * 1.5
        + width_balance * 0.5
        + height_balance * 0.5
        + centre_score * 0.5
    )


def _contour_quads(mask: np.ndarray, image_shape: tuple[int, int], scale_back: float, method: str):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[QuadDetection] = []
    h, w = image_shape
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < h * w * 0.08:
            continue
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        candidates = []
        for eps_ratio in (0.012, 0.018, 0.025, 0.035, 0.05, 0.07):
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append(approx.reshape(4, 2))
        if not candidates:
            candidates.append(cv2.boxPoints(cv2.minAreaRect(hull)))

        for quad_small in candidates:
            score = _quad_score(quad_small, (h, w))
            if score <= 0:
                continue
            quad_full = order_quad(quad_small.astype(np.float32) * scale_back)
            results.append(QuadDetection(quad_full, method, score))
    return results


def detect_poster_quad(rgb: np.ndarray, max_detection_side: int = 1600) -> QuadDetection:
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, max_detection_side / float(max(h0, w0)))
    if scale < 1.0:
        small = cv2.resize(rgb, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb.copy()
    h, w = small.shape[:2]
    scale_back = 1.0 / scale

    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    otsu_t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_threshold = int(max(28, min(145, otsu_t * 0.72)))
    sat_mask = np.where(sat >= sat_threshold, 255, 0).astype(np.uint8)
    k = max(5, int(round(min(h, w) * 0.018)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    edge = cv2.Canny(blur, int(max(0, 0.55 * med)), int(min(255, 1.35 * med)))
    ek = max(3, int(round(min(h, w) * 0.008)) | 1)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek)),
        iterations=2,
    )
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

    candidates = []
    candidates.extend(_contour_quads(sat_mask, (h, w), scale_back, "saturation"))
    candidates.extend(_contour_quads(edge, (h, w), scale_back, "edges"))
    combined = cv2.bitwise_or(sat_mask, edge)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidates.extend(_contour_quads(combined, (h, w), scale_back, "combined"))

    if not candidates:
        raise RuntimeError("Could not detect a plausible poster/page quadrilateral")
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def warp_quad(rgb: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, dict]:
    q = order_quad(quad)
    tl, tr, br, bl = q
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = max(2, int(round((width_top + width_bottom) / 2)))
    out_h = max(2, int(round((height_left + height_right) / 2)))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        rgb,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, {
        "source_quad": q.round(3).tolist(),
        "perspective_matrix": matrix.tolist(),
        "rectified_width": out_w,
        "rectified_height": out_h,
    }


def rotate_image(rgb: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    if rotation == "keep":
        return rgb, np.eye(3, dtype=np.float64)
    if rotation == "90cw":
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), np.array(
            [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
    if rotation == "90ccw":
        return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array(
            [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=np.float64
        )
    if rotation == "180":
        return cv2.rotate(rgb, cv2.ROTATE_180), np.array(
            [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64
        )
    raise ValueError(f"Unsupported rotation: {rotation}")


@exclusive_job('job_dir')
def normalize_reference(
    input_path: str | Path,
    job_dir: str | Path,
    rotation: Rotation = "keep",
    corners: np.ndarray | list[list[float]] | None = None,
) -> dict:
    job_dir = Path(job_dir)
    preserved, source_meta = preserve_source(input_path, job_dir)
    rgb = _load_exif_transposed_rgb(preserved)

    if corners is None:
        detection = detect_poster_quad(rgb)
    else:
        supplied = order_quad(np.asarray(corners, dtype=np.float32))
        detection = QuadDetection(supplied, "manual_override", _quad_score(supplied, rgb.shape[:2]))

    rectified, geom = warp_quad(rgb, detection.points)
    normalized, rotation_matrix = rotate_image(rectified, rotation)

    work_dir = job_dir / "work"
    debug_dir = job_dir / "debug"
    meta_dir = job_dir / "metadata"
    work_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = work_dir / "normalized_reference.png"
    Image.fromarray(normalized).save(normalized_path)

    debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    q = order_quad(detection.points).astype(int)
    cv2.polylines(debug, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 5, cv2.LINE_AA)
    for idx, (x, y) in enumerate(q):
        cv2.circle(debug, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(debug, str(idx), (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    debug_path = debug_dir / "detected_quad.png"
    cv2.imwrite(str(debug_path), debug)

    geometry = {
        "version": 1,
        "source": source_meta,
        "exif_transposed_dimensions": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
        "quad_detection": {
            "method": detection.method,
            "score": round(float(detection.score), 6),
            "points_tl_tr_br_bl": detection.points.round(3).tolist(),
        },
        **geom,
        "rotation": rotation,
        "rotation_matrix": rotation_matrix.tolist(),
        "normalized_width": int(normalized.shape[1]),
        "normalized_height": int(normalized.shape[0]),
        "normalized_path": str(normalized_path.relative_to(job_dir)),
        "debug_overlay_path": str(debug_path.relative_to(job_dir)),
        "accuracy_note": "Automatic quadrilateral detection is provisional and must be visually verified before downstream pixel fitting. The --corners/manual override is the deterministic fallback for precision work.",
    }
    (meta_dir / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    return geometry
