from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .deps import cv2, np

Point = Tuple[float, float]

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MASK_POSITIVE_BGR = np.array([0, 255, 255], dtype=np.uint8)
ALL_VARIANTS = ("raw", "hf", "vf", "rcw", "rccw", "warp", "sup", "sdown", "tleft", "tup")
DEFAULT_VARIANTS = ("raw", "warp", "sup", "sdown", "tleft", "tup")
DEFAULT_S5MARS_REPO_ID = "Voxel51/S5Mars"
S5MARS_MASK_TARGETS = {
    1: "sky",
    2: "ridge",
    3: "soil",
    4: "sand",
    5: "bedrock",
    6: "rock",
    7: "rover",
    8: "trace",
    9: "hole",
}
S5MARS_CLASS_NAME_TO_ID = {name: class_id for class_id, name in S5MARS_MASK_TARGETS.items()}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ellipse_kernel(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    values = image[np.isfinite(image)].astype(np.float32)
    if values.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.uint8(np.round(scaled * 255.0))


def robust_normalize_map(
    values: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    lower_percentile: float = 50.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(data)
    if valid_mask is not None:
        finite &= valid_mask.astype(bool)
    valid_values = data[finite]
    if valid_values.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(valid_values, lower_percentile))
    hi = float(np.percentile(valid_values, upper_percentile))
    if not np.isfinite(lo):
        lo = float(valid_values.min())
    if not np.isfinite(hi):
        hi = float(valid_values.max())
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    normalized = (data - lo) / (hi - lo)
    normalized[~np.isfinite(normalized)] = 0.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def ensure_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        return gray
    return normalize_uint8(gray)


def color_for_id(rock_id: int) -> Tuple[int, int, int]:
    hue = (rock_id * 53) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class CameraModel:
    width: int
    height: int
    camera_height_m: float
    vfov_deg: float
    pitch_deg: float
    auto_estimated: bool
    horizon_row: Optional[float]
    pitch_source: str
    vfov_source: str
    camera_height_source: str

    @property
    def cx(self) -> float:
        return (self.width - 1.0) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1.0) / 2.0

    @property
    def fy(self) -> float:
        import math

        return self.height / (2.0 * math.tan(math.radians(self.vfov_deg) / 2.0))

    @property
    def fx(self) -> float:
        import math

        aspect = self.width / max(self.height, 1.0)
        hfov = 2.0 * math.atan(math.tan(math.radians(self.vfov_deg) / 2.0) * aspect)
        return self.width / (2.0 * math.tan(hfov / 2.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "camera_height_m": round(self.camera_height_m, 4),
            "vfov_deg": round(self.vfov_deg, 4),
            "pitch_deg": round(self.pitch_deg, 4),
            "auto_estimated": self.auto_estimated,
            "horizon_row": None if self.horizon_row is None else round(float(self.horizon_row), 2),
            "pitch_source": self.pitch_source,
            "vfov_source": self.vfov_source,
            "camera_height_source": self.camera_height_source,
        }


@dataclass
class RockDetection:
    rock_id: int
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid_px: Point
    confidence: float
    passes_threshold: bool
    area_px: float
    mean_probability: float
    mean_score: float
    peak_score: float
    estimated_visible_span_cm: Optional[float] = None
    estimated_height_cm: Optional[float] = None
    estimated_distance_m: Optional[float] = None
    size_estimate_method: Optional[str] = None
    passes_height_filter: bool = False


@dataclass
class TrainMetrics:
    loss: float
    dice: float
    iou: float
