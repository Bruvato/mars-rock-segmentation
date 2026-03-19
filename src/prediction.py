from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common import (
    CameraModel,
    RockDetection,
    clamp,
    color_for_id,
    ellipse_kernel,
    ensure_uint8_grayscale,
    normalize_uint8,
    robust_normalize_map,
)
from data import image_to_tensor
from deps import cv2, ndi, ensure_torch, np, remove_small_holes, remove_small_objects, torch, watershed
from training import choose_device, load_checkpoint_model, model_logits


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    models_dir = Path("models")
    candidates = sorted(path for path in models_dir.glob("*.pt") if path.is_file())
    if len(candidates) == 1:
        fallback_path = candidates[0]
        print(f"checkpoint {checkpoint_path} not found; using {fallback_path} instead")
        return fallback_path

    if candidates:
        available = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Available checkpoints: {available}. "
            "Pass one explicitly with --checkpoint."
        )

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. No .pt checkpoints were found under {models_dir}."
    )


def load_image(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    alpha_mask: Optional[np.ndarray] = None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha_mask = image[:, :, 3] > 0
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image, alpha_mask


def build_valid_ground_mask(image_bgr: np.ndarray, alpha_mask: Optional[np.ndarray]) -> np.ndarray:
    gray = ensure_uint8_grayscale(image_bgr)
    valid = gray > 10
    if alpha_mask is not None:
        valid &= alpha_mask
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, ellipse_kernel(9)) > 0
    valid = remove_small_holes(valid, area_threshold=1024)
    valid = remove_small_objects(valid, min_size=4096)
    valid = ndi.binary_fill_holes(valid)
    return valid.astype(bool)


def estimate_horizon_row(gray: np.ndarray, valid_mask: np.ndarray) -> Optional[float]:
    height, width = gray.shape[:2]
    top_limit = max(32, int(height * 0.65))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    cropped_edges = edges[:top_limit].copy()
    cropped_edges[~valid_mask[:top_limit]] = 0
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(40, width // 12),
        minLineLength=max(40, int(width * 0.25)),
        maxLineGap=18,
    )
    candidates: List[Tuple[float, float]] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = math.hypot(x2 - x1, y2 - y1)
            slope = abs((y2 - y1) / max(abs(x2 - x1), 1))
            if length >= width * 0.22 and slope <= 0.55:
                y_mid = (y1 + y2) / 2.0
                weight = length * (1.0 - slope / 0.55)
                candidates.append((y_mid, weight))
    if candidates:
        total_weight = sum(weight for _, weight in candidates)
        return sum(y * weight for y, weight in candidates) / max(total_weight, 1e-6)
    return None


def build_camera_model(
    image_shape: Sequence[int],
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    camera_height_m: Optional[float],
    vfov_deg: Optional[float],
    pitch_deg: Optional[float],
) -> CameraModel:
    height, width = int(image_shape[0]), int(image_shape[1])
    resolved_height = 2.0 if camera_height_m is None else float(camera_height_m)
    resolved_vfov = 45.0 if vfov_deg is None else float(vfov_deg)
    resolved_pitch = 35.0
    horizon_row: Optional[float] = None
    auto_estimated = False
    pitch_source = "fallback"
    if pitch_deg is not None:
        resolved_pitch = float(pitch_deg)
        pitch_source = "cli"
    else:
        gray = ensure_uint8_grayscale(image_bgr)
        horizon_row = estimate_horizon_row(gray, valid_mask)
        if horizon_row is not None:
            fy = height / (2.0 * math.tan(math.radians(resolved_vfov) / 2.0))
            inferred_pitch = math.degrees(math.atan((height / 2.0 - horizon_row) / max(fy, 1e-6)))
            if horizon_row < height * 0.42 and 14.0 <= inferred_pitch <= 60.0:
                resolved_pitch = inferred_pitch
                auto_estimated = True
                pitch_source = "horizon"
            else:
                horizon_row = None
    return CameraModel(
        width=width,
        height=height,
        camera_height_m=resolved_height,
        vfov_deg=resolved_vfov,
        pitch_deg=resolved_pitch,
        auto_estimated=auto_estimated,
        horizon_row=horizon_row,
        pitch_source=pitch_source,
        vfov_source="cli" if vfov_deg is not None else "fallback",
        camera_height_source="cli" if camera_height_m is not None else "fallback",
    )


def project_pixels_to_ground(camera: CameraModel, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    x = (points[:, 0] - camera.cx) / camera.fx
    y = -(points[:, 1] - camera.cy) / camera.fy
    cos_pitch = math.cos(math.radians(camera.pitch_deg))
    sin_pitch = math.sin(math.radians(camera.pitch_deg))
    world_y = cos_pitch * y - sin_pitch
    world_z = sin_pitch * y + cos_pitch
    valid = world_y < -1e-5
    intersections = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        scale = -camera.camera_height_m / world_y[valid]
        intersections[valid, 0] = x[valid] * scale
        intersections[valid, 1] = world_z[valid] * scale
    return intersections, valid


def estimate_local_ground_scale(camera: CameraModel, x: float, y: float) -> Tuple[float, float]:
    sample_points = np.array([[x, y], [x + 1.0, y], [x, y + 1.0]], dtype=np.float32)
    projected, valid = project_pixels_to_ground(camera, sample_points)
    if not (valid[0] and valid[1]):
        return 0.0, float("inf")
    horizontal_cm_per_px = float(np.linalg.norm(projected[1] - projected[0]) * 100.0)
    distance_m = float(np.linalg.norm(projected[0]))
    if not np.isfinite(horizontal_cm_per_px) or horizontal_cm_per_px <= 0.0:
        return 0.0, float("inf")
    return horizontal_cm_per_px, distance_m


def estimate_visible_size_cm(camera: CameraModel, contour: np.ndarray) -> Tuple[float, float]:
    all_points = contour[:, 0, :].astype(np.float32)
    area_px = max(float(cv2.contourArea(contour)), 1.0)
    x, y, w, h = contour_bbox(contour)
    if w <= 0 or h <= 0:
        return 0.0, float("inf")
    bottom_band = max(2.0, 0.18 * h)
    bottom_points = all_points[all_points[:, 1] >= np.max(all_points[:, 1]) - bottom_band]
    if bottom_points.shape[0] == 0:
        bottom_points = all_points
    sample_x = float(np.median(bottom_points[:, 0]))
    sample_y = float(np.median(bottom_points[:, 1]))
    cm_per_px, distance_m = estimate_local_ground_scale(camera, sample_x, sample_y)
    if cm_per_px <= 0.0 or distance_m == float("inf") or cm_per_px > 12.0:
        return 0.0, float("inf")
    rect = cv2.minAreaRect(contour)
    side_a, side_b = [float(value) for value in rect[1]]
    major_px = max(side_a, side_b, float(max(w, h)))
    characteristic_px = min(major_px, 2.2 * math.sqrt(area_px))
    characteristic_px = max(characteristic_px, 1.0 * math.sqrt(area_px))
    burial_factor = 1.12 if y + h >= camera.height - 2 else 1.06
    return characteristic_px * cm_per_px * burial_factor, distance_m


def estimate_height_from_contour(camera: CameraModel, contour: np.ndarray) -> Tuple[float, float, float, str]:
    visible_span_cm, distance_m = estimate_visible_size_cm(camera, contour)
    if visible_span_cm <= 0.0 or not np.isfinite(distance_m):
        return 0.0, 0.0, float("inf"), "unavailable"

    area_px = max(float(cv2.contourArea(contour)), 1.0)
    x, y, w, h = contour_bbox(contour)
    rect = cv2.minAreaRect(contour)
    side_a, side_b = [max(float(value), 1.0) for value in rect[1]]
    major_px = max(side_a, side_b, float(max(w, h)), 1.0)
    minor_px = max(min(side_a, side_b), float(min(w, h)), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = clamp(area_px / hull_area, 0.0, 1.0)
    rect_fill = clamp(area_px / max(major_px * minor_px, 1.0), 0.0, 1.0)
    compactness = clamp(minor_px / major_px, 0.0, 1.0)
    edge_penalty = 0.92 if y + h >= camera.height - 2 else 1.0
    height_ratio = clamp(0.50 * compactness + 0.25 * solidity + 0.20 * rect_fill + 0.07, 0.32, 0.95)
    estimated_height_cm = visible_span_cm * height_ratio * edge_penalty
    return visible_span_cm, estimated_height_cm, distance_m, "ground_scale_shape_heuristic"


def estimate_detection_heights(
    detections: Sequence[RockDetection],
    camera: CameraModel,
    min_height_cm: float,
) -> List[RockDetection]:
    estimated: List[RockDetection] = []
    for detection in detections:
        visible_span_cm, height_cm, distance_m, method = estimate_height_from_contour(camera, detection.contour)
        distance_value = None if not np.isfinite(distance_m) else float(distance_m)
        detection.estimated_visible_span_cm = float(visible_span_cm) if visible_span_cm > 0.0 else None
        detection.estimated_height_cm = float(height_cm) if height_cm > 0.0 else None
        detection.estimated_distance_m = distance_value
        detection.size_estimate_method = method if height_cm > 0.0 else None
        detection.passes_height_filter = bool(
            detection.passes_threshold
            and detection.estimated_height_cm is not None
            and detection.estimated_height_cm >= float(min_height_cm)
        )
        estimated.append(detection)
    return estimated


def compute_tile_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    step = max(tile_size - overlap, tile_size // 4)
    positions = list(range(0, max(length - tile_size, 0) + 1, step))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def pad_image_to_tile(image_bgr: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return image_bgr
    return cv2.copyMakeBorder(image_bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)


def predict_probability_map(
    model: "torch.nn.Module",
    image_bgr: np.ndarray,
    device: "torch.device",
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    ensure_torch()
    padded = pad_image_to_tile(image_bgr, tile_size)
    height, width = padded.shape[:2]
    y_positions = compute_tile_positions(height, tile_size, tile_overlap)
    x_positions = compute_tile_positions(width, tile_size, tile_overlap)
    probabilities = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                tile_bgr = padded[y : y + tile_size, x : x + tile_size]
                tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = image_to_tensor(tile_rgb).unsqueeze(0).to(device)
                logits = model_logits(model, image_tensor)
                tile_prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
                probabilities[y : y + tile_size, x : x + tile_size] += tile_prob
                counts[y : y + tile_size, x : x + tile_size] += 1.0
    counts = np.maximum(counts, 1.0)
    merged = probabilities / counts
    return merged[: image_bgr.shape[0], : image_bgr.shape[1]]


def compute_row_weight_map(height: int, horizon_row: Optional[float]) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None]
    start_row = 0.12 * height if horizon_row is None else horizon_row + 0.04 * height
    start_row = clamp(start_row, 0.08 * height, 0.32 * height)
    row_position = np.clip((rows - start_row) / max(height - start_row - 1.0, 1.0), 0.0, 1.0)
    weights = 0.18 + 0.82 * np.sqrt(row_position)
    return weights.astype(np.float32)


def build_scene_score_map(
    image_bgr: np.ndarray,
    probability_map: np.ndarray,
    valid_mask: np.ndarray,
    horizon_row: Optional[float],
) -> np.ndarray:
    gray = ensure_uint8_grayscale(image_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), sigmaX=3.0)
    tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, ellipse_kernel(15))
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, ellipse_kernel(17))
    gradient_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    local_contrast = cv2.absdiff(clahe, cv2.GaussianBlur(clahe, (0, 0), sigmaX=6.0))

    probability_score = robust_normalize_map(probability_map, valid_mask, lower_percentile=55.0, upper_percentile=99.8)
    contrast_score = robust_normalize_map(tophat, valid_mask, lower_percentile=58.0, upper_percentile=99.5)
    shadow_score = robust_normalize_map(blackhat, valid_mask, lower_percentile=62.0, upper_percentile=99.5)
    edge_score = robust_normalize_map(gradient, valid_mask, lower_percentile=66.0, upper_percentile=99.5)
    texture_score = robust_normalize_map(local_contrast, valid_mask, lower_percentile=58.0, upper_percentile=99.3)
    row_weight = 0.82 + 0.18 * compute_row_weight_map(image_bgr.shape[0], horizon_row)

    score = (
        0.42 * probability_score
        + 0.20 * contrast_score
        + 0.16 * shadow_score
        + 0.12 * edge_score
        + 0.10 * texture_score
    )
    score *= row_weight
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=0.9)
    score[~valid_mask] = 0.0
    return np.clip(score, 0.0, 1.0)


def postprocess_probability_map(
    probability_map: np.ndarray,
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    horizon_row: Optional[float],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    score_map = build_scene_score_map(image_bgr, probability_map, valid_mask, horizon_row)
    valid_scores = score_map[valid_mask]
    if valid_scores.size == 0:
        return np.zeros_like(valid_mask, dtype=bool), score_map

    score_cutoff = max(float(np.percentile(valid_scores, 98.2)), 0.24)
    support_cutoff = max(float(np.percentile(valid_scores, 96.5)), 0.18)

    binary = (score_map >= score_cutoff) & valid_mask
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, ellipse_kernel(7)) > 0
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, ellipse_kernel(5)) > 0
    grown_support = cv2.dilate(binary.astype(np.uint8), ellipse_kernel(5), iterations=1) > 0
    binary |= grown_support & (score_map >= support_cutoff) & valid_mask
    min_component_size = max(18, int(image_bgr.shape[0] * image_bgr.shape[1] * 0.000012))
    binary = remove_small_objects(binary, min_size=min_component_size)
    binary = remove_small_holes(binary, area_threshold=128)
    binary &= valid_mask

    component_labels, component_count = ndi.label(binary)
    filtered = np.zeros_like(binary, dtype=bool)
    for label in range(1, component_count + 1):
        component = component_labels == label
        area = int(component.sum())
        mean_score = float(np.mean(score_map[component]))
        peak_score = float(np.max(score_map[component]))
        if area < min_component_size:
            continue
        if peak_score < max(score_cutoff * 0.95, 0.24):
            continue
        if mean_score < max(support_cutoff * 0.92, 0.16):
            continue
        filtered |= component

    filtered = cv2.morphologyEx(filtered.astype(np.uint8), cv2.MORPH_CLOSE, ellipse_kernel(5)) > 0
    filtered = remove_small_holes(filtered, area_threshold=96)
    filtered &= valid_mask
    return filtered.astype(bool), score_map


def split_instances(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.zeros(binary_mask.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary_mask)
    distance = ndi.gaussian_filter(distance, sigma=1.0)
    maxima = distance == ndi.maximum_filter(distance, size=19)
    maxima &= binary_mask & (distance > 3.0)
    markers, marker_count = ndi.label(maxima)
    if marker_count == 0:
        markers, _ = ndi.label(binary_mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for label in range(1, int(labels.max()) + 1):
        component = labels == label
        if component.sum() < 24:
            continue
        output[component] = next_label
        next_label += 1
    return output


def contour_bbox(contour: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)


def is_probable_hardware(contour: np.ndarray, image_shape: Sequence[int]) -> bool:
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = contour_bbox(contour)
    touches_bottom = y + h >= height - 1
    touches_side = x <= 0 or x + w >= width - 1
    area = float(cv2.contourArea(contour))
    rect_fill = area / max(float(w * h), 1.0)
    aspect = max(w / max(h, 1.0), h / max(w, 1.0))
    return touches_bottom and touches_side and rect_fill > 0.22 and aspect > 1.8 and area > 400.0


def extract_detections(
    labels: np.ndarray,
    probability_map: np.ndarray,
    score_map: np.ndarray,
    threshold: float,
) -> List[RockDetection]:
    detections: List[RockDetection] = []
    height = labels.shape[0]
    for label in range(1, int(labels.max()) + 1):
        component = labels == label
        if not np.any(component):
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
        x, y, w, h = contour_bbox(contour)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = area / hull_area
        extent = area / max(float(w * h), 1.0)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        mean_probability = float(np.mean(probability_map[component]))
        mean_score = float(np.mean(score_map[component]))
        peak_score = float(np.max(score_map[component]))
        row_ratio = float(centroid_y / max(height - 1.0, 1.0))
        min_area = float(np.interp(row_ratio, [0.0, 0.22, 0.45, 0.7, 1.0], [110.0, 84.0, 60.0, 34.0, 18.0]))
        if area < min_area:
            continue
        if min(w, h) < (6 if row_ratio < 0.35 else 4):
            continue
        if solidity < 0.38 or extent < 0.12 or circularity < 0.015:
            continue
        if is_probable_hardware(contour, labels.shape):
            continue
        confidence = (
            0.44 * mean_probability
            + 0.20 * peak_score
            + 0.16 * mean_score
            + 0.10 * clamp(solidity, 0.0, 1.0)
            + 0.06 * clamp(extent, 0.0, 1.0)
            + 0.04 * clamp(circularity / 0.45, 0.0, 1.0)
        )
        confidence = clamp(confidence, 0.0, 0.99)
        passes_threshold = bool(
            peak_score >= max(float(threshold) * 0.55, 0.26)
            and mean_score >= 0.16
            and confidence >= 0.30
        )
        detections.append(
            RockDetection(
                rock_id=len(detections) + 1,
                contour=contour,
                bbox=(x, y, w, h),
                centroid_px=(float(centroid_x), float(centroid_y)),
                confidence=float(confidence),
                passes_threshold=passes_threshold,
                area_px=float(area),
                mean_probability=mean_probability,
                mean_score=mean_score,
                peak_score=peak_score,
            )
        )
    return detections


def choose_label_rect(
    image_shape: Sequence[int],
    bbox: Tuple[int, int, int, int],
    text_size: Tuple[int, int],
    occupied: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = bbox
    text_width, text_height = text_size
    padding = 6
    box_width = text_width + padding * 2
    box_height = text_height + padding * 2
    candidates = [
        (x, y - box_height - 6),
        (x + w - box_width, y - box_height - 6),
        (x, y + h + 6),
        (x + w - box_width, y + h + 6),
        (x + w + 8, y),
        (x - box_width - 8, y),
    ]

    def intersects(left: int, top: int, right: int, bottom: int, other: Tuple[int, int, int, int]) -> bool:
        o_left, o_top, o_right, o_bottom = other
        return not (right <= o_left or left >= o_right or bottom <= o_top or top >= o_bottom)

    for left, top in candidates:
        left = int(clamp(left, 0, max(width - box_width, 0)))
        top = int(clamp(top, 0, max(height - box_height, 0)))
        rect = (left, top, left + box_width, top + box_height)
        if not any(intersects(*rect, other) for other in occupied):
            occupied.append(rect)
            return rect

    left = int(clamp(x, 0, max(width - box_width, 0)))
    top = int(clamp(y, 0, max(height - box_height, 0)))
    rect = (left, top, left + box_width, top + box_height)
    occupied.append(rect)
    return rect


def annotate_image(image_bgr: np.ndarray, detections: Sequence[RockDetection], show_labels: bool = False) -> np.ndarray:
    annotated = image_bgr.copy()
    overlay = image_bgr.copy()
    label_boxes: List[Tuple[int, int, int, int]] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    selected = sorted(
        (item for item in detections if item.passes_threshold),
        key=lambda item: (item.confidence, item.area_px),
        reverse=True,
    )
    unified_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for detection in selected:
        cv2.drawContours(unified_mask, [detection.contour], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    rock_color = np.array([40, 180, 255], dtype=np.uint8)
    overlay[unified_mask > 0] = rock_color
    annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0.0)
    for detection in selected:
        cv2.drawContours(annotated, [detection.contour], -1, tuple(int(value) for value in rock_color), 1, lineType=cv2.LINE_AA)
        if not show_labels:
            continue
        label = f"Rock {detection.rock_id} ({detection.confidence:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.42, 1)
        left, top, right, bottom = choose_label_rect(annotated.shape, detection.bbox, (text_width, text_height), label_boxes)
        cv2.putText(
            annotated,
            label,
            (left + 6, bottom - baseline - 4),
            font,
            0.42,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (left + 6, bottom - baseline - 4),
            font,
            0.42,
            tuple(int(value) for value in rock_color),
            1,
            cv2.LINE_AA,
        )
    return annotated


def annotate_height_filtered_image(
    image_bgr: np.ndarray,
    detections: Sequence[RockDetection],
    min_height_cm: float,
) -> np.ndarray:
    annotated = image_bgr.copy()
    label_boxes: List[Tuple[int, int, int, int]] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    selected = sorted(
        (item for item in detections if item.passes_height_filter),
        key=lambda item: (
            0.0 if item.estimated_height_cm is None else item.estimated_height_cm,
            item.confidence,
        ),
        reverse=True,
    )

    for detection in selected:
        color = color_for_id(detection.rock_id)
        cv2.drawContours(annotated, [detection.contour], -1, color, 2, lineType=cv2.LINE_AA)
        estimated_height_cm = 0.0 if detection.estimated_height_cm is None else detection.estimated_height_cm
        label = f"Rock {detection.rock_id}: {estimated_height_cm:.1f} cm"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.5, 1)
        left, top, right, bottom = choose_label_rect(annotated.shape, detection.bbox, (text_width, text_height), label_boxes)
        cv2.rectangle(annotated, (left, top), (right, bottom), (245, 245, 245), cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (left + 6, bottom - baseline - 4),
            font,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    if not selected:
        label = f"No rocks >= {float(min_height_cm):.1f} cm"
        cv2.putText(annotated, label, (18, 28), font, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (18, 28), font, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
    return annotated


def simplify_contour(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.01 * perimeter)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return simplified if simplified.shape[0] >= 3 else contour


def serialize_detections(
    input_path: Path,
    model_metadata: Dict[str, object],
    detections: Sequence[RockDetection],
    image_shape: Sequence[int],
    threshold_probability: float,
    output_mask_path: Path,
    output_height_image_path: Path,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    camera: CameraModel,
    min_height_cm: float,
) -> Dict[str, object]:
    height, width = int(image_shape[0]), int(image_shape[1])
    rocks: List[Dict[str, object]] = []
    for detection in detections:
        simplified = simplify_contour(detection.contour)
        points = []
        for point in simplified[:, 0, :]:
            x = int(clamp(float(point[0]), 0, width - 1))
            y = int(clamp(float(point[1]), 0, height - 1))
            points.append([x, y])
        rocks.append(
            {
                "id": detection.rock_id,
                "bbox": list(detection.bbox),
                "centroid": [round(detection.centroid_px[0], 2), round(detection.centroid_px[1], 2)],
                "confidence": round(detection.confidence, 3),
                "area_px": round(detection.area_px, 2),
                "mean_probability": round(detection.mean_probability, 4),
                "mean_score": round(detection.mean_score, 4),
                "peak_score": round(detection.peak_score, 4),
                "passes_threshold": detection.passes_threshold,
                "estimated_visible_span_cm": (
                    None
                    if detection.estimated_visible_span_cm is None
                    else round(detection.estimated_visible_span_cm, 2)
                ),
                "estimated_height_cm": (
                    None if detection.estimated_height_cm is None else round(detection.estimated_height_cm, 2)
                ),
                "estimated_distance_m": (
                    None if detection.estimated_distance_m is None else round(detection.estimated_distance_m, 3)
                ),
                "size_estimate_method": detection.size_estimate_method,
                "passes_height_filter": detection.passes_height_filter,
                "contour": points,
            }
        )
    passing = [detection for detection in detections if detection.passes_threshold]
    height_selected = [detection for detection in detections if detection.passes_height_filter]
    foreground_pixels = int(np.count_nonzero(binary_mask))
    valid_pixels = int(np.count_nonzero(valid_mask))
    return {
        "image": str(input_path),
        "mask_image": str(output_mask_path),
        "height_filtered_image": str(output_height_image_path),
        "threshold_probability": round(float(threshold_probability), 3),
        "height_filter_cm": round(float(min_height_cm), 2),
        "model": model_metadata,
        "camera": camera.to_dict(),
        "segmentation_summary": {
            "instances": len(detections),
            "selected_instances": len(passing),
            "height_filtered_instances": len(height_selected),
            "foreground_pixels": foreground_pixels,
            "valid_pixels": valid_pixels,
            "foreground_ratio": round(foreground_pixels / max(valid_pixels, 1), 6),
        },
        "rocks": rocks,
    }


def save_debug_outputs(
    debug_dir: Path,
    probability_map: np.ndarray,
    score_map: np.ndarray,
    binary_mask: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "probability_map.png"), normalize_uint8(probability_map))
    cv2.imwrite(str(debug_dir / "score_map.png"), normalize_uint8(score_map))
    cv2.imwrite(str(debug_dir / "binary_mask.png"), binary_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "labels.png"), normalize_uint8(labels.astype(np.float32)))
    cv2.imwrite(str(debug_dir / "valid_mask.png"), valid_mask.astype(np.uint8) * 255)


def write_mask_image(output_mask_path: Path, binary_mask: np.ndarray) -> None:
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask_path), binary_mask.astype(np.uint8) * 255)


def format_height_tag(min_height_cm: float) -> str:
    rounded = round(float(min_height_cm), 1)
    if math.isclose(rounded, round(rounded), abs_tol=1e-6):
        return f"{int(round(rounded))}cm"
    return f"{rounded:.1f}".replace(".", "p").rstrip("0").rstrip("p") + "cm"


def default_output_paths(input_path: Path, min_height_cm: float = 10.0) -> Tuple[Path, Path, Path, Path]:
    stem = input_path.stem
    output_dir = Path("outputs")
    height_tag = format_height_tag(min_height_cm)
    return (
        output_dir / f"{stem}_annotated.png",
        output_dir / f"{stem}.json",
        output_dir / f"{stem}_mask.png",
        output_dir / f"{stem}_rocks_over_{height_tag}.png",
    )


def predict_image(args: argparse.Namespace) -> None:
    ensure_torch()
    input_path = Path(args.input)
    output_image_path = Path(args.output_image)
    output_json_path = Path(args.output_json)
    output_mask_path = Path(args.output_mask)
    output_height_image_path = Path(args.output_height_image)
    device = choose_device(args.device)
    checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint))
    model, model_metadata = load_checkpoint_model(checkpoint_path, device)
    image_bgr, alpha_mask = load_image(input_path)
    valid_mask = build_valid_ground_mask(image_bgr, alpha_mask)
    camera = build_camera_model(
        image_shape=image_bgr.shape,
        image_bgr=image_bgr,
        valid_mask=valid_mask,
        camera_height_m=args.camera_height_m,
        vfov_deg=args.vfov_deg,
        pitch_deg=args.pitch_deg,
    )
    probability_map = predict_probability_map(
        model,
        image_bgr=image_bgr,
        device=device,
        tile_size=int(args.tile_size),
        tile_overlap=int(args.tile_overlap),
    )
    binary_mask, score_map = postprocess_probability_map(
        probability_map,
        image_bgr=image_bgr,
        valid_mask=valid_mask,
        horizon_row=camera.horizon_row,
        threshold=float(args.threshold),
    )
    labels = split_instances(binary_mask)
    detections = extract_detections(labels, probability_map, score_map, threshold=float(args.threshold))
    detections = estimate_detection_heights(detections, camera=camera, min_height_cm=float(args.min_height_cm))
    annotated = annotate_image(image_bgr, detections, show_labels=bool(args.show_labels))
    height_filtered_annotated = annotate_height_filtered_image(
        image_bgr,
        detections,
        min_height_cm=float(args.min_height_cm),
    )
    payload = serialize_detections(
        input_path=input_path,
        model_metadata=model_metadata,
        detections=detections,
        image_shape=image_bgr.shape,
        threshold_probability=float(args.threshold),
        output_mask_path=output_mask_path,
        output_height_image_path=output_height_image_path,
        binary_mask=binary_mask,
        valid_mask=valid_mask,
        camera=camera,
        min_height_cm=float(args.min_height_cm),
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_height_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), annotated)
    cv2.imwrite(str(output_height_image_path), height_filtered_annotated)
    write_mask_image(output_mask_path, binary_mask)
    output_json_path.write_text(json.dumps(payload, indent=2))
    if args.debug_dir is not None:
        save_debug_outputs(Path(args.debug_dir), probability_map, score_map, binary_mask, labels, valid_mask)
