from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from common import (
    ALL_VARIANTS,
    DEFAULT_S5MARS_REPO_ID,
    IMAGE_MEAN,
    IMAGE_STD,
    MASK_POSITIVE_BGR,
    S5MARS_CLASS_NAME_TO_ID,
    S5MARS_MASK_TARGETS,
)
from deps import Dataset, cv2, ensure_fiftyone, ensure_torch, np, torch

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def iter_image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def infer_variant(stem: str) -> Optional[str]:
    token = stem.rsplit("_", 1)[-1].lower()
    return token if token in ALL_VARIANTS else None


def resolve_target_class_id(target_class: str) -> int:
    token = str(target_class).strip().lower()
    if token.isdigit():
        class_id = int(token)
        if class_id in S5MARS_MASK_TARGETS:
            return class_id
    if token in S5MARS_CLASS_NAME_TO_ID:
        return S5MARS_CLASS_NAME_TO_ID[token]
    valid_names = ", ".join(sorted(S5MARS_CLASS_NAME_TO_ID))
    raise ValueError(f"Unknown S5Mars class {target_class!r}. Use one of: {valid_names}, or a class id 1-9.")


def _maybe_grayscale_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        return mask[:, :, 0]
    if mask.ndim == 3 and mask.shape[2] >= 3:
        rgb_like = mask[:, :, :3]
        if np.array_equal(rgb_like[:, :, 0], rgb_like[:, :, 1]) and np.array_equal(rgb_like[:, :, 0], rgb_like[:, :, 2]):
            return rgb_like[:, :, 0]
    return None


def label_image_to_binary(label_image: np.ndarray, target_class_id: Optional[int] = None) -> np.ndarray:
    grayscale = _maybe_grayscale_mask(label_image)
    if grayscale is not None:
        unique_values = set(int(value) for value in np.unique(grayscale).tolist())
        if target_class_id is not None and not unique_values.issubset({0, 1, 255}):
            return grayscale == target_class_id
        return grayscale > 0

    if label_image.ndim != 3 or label_image.shape[2] < 3:
        raise ValueError(f"Unsupported label image shape: {label_image.shape}")

    bgr = label_image[:, :, :3]
    return (
        (bgr[:, :, 0] == int(MASK_POSITIVE_BGR[0]))
        & (bgr[:, :, 1] >= 200)
        & (bgr[:, :, 2] >= 200)
    )


def segmentation_mask_to_binary(mask: np.ndarray, target_class_id: int) -> np.ndarray:
    grayscale = _maybe_grayscale_mask(np.asarray(mask))
    if grayscale is None:
        raise ValueError("Expected a grayscale segmentation mask from S5Mars.")

    unique_values = set(int(value) for value in np.unique(grayscale).tolist())
    if unique_values.issubset({0, 1, 255}):
        binary = grayscale > 0
    else:
        binary = grayscale == target_class_id
    return (binary.astype(np.uint8)) * 255


def image_to_tensor(image_rgb: np.ndarray) -> "torch.Tensor":
    ensure_torch()
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).float()


def resize_pair(image_rgb: np.ndarray, mask: np.ndarray, image_size: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if image_size is None or image_size <= 0:
        return image_rgb, mask
    target_size = (int(image_size), int(image_size))
    if image_rgb.shape[:2] == target_size[::-1]:
        return image_rgb, mask
    resized_image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return resized_image, resized_mask


def pad_pair_to_minimum(image_rgb: np.ndarray, mask: np.ndarray, minimum_size: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    pad_bottom = max(0, int(minimum_size) - height)
    pad_right = max(0, int(minimum_size) - width)
    if pad_bottom == 0 and pad_right == 0:
        return image_rgb, mask

    padded_image = cv2.copyMakeBorder(
        image_rgb,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_REFLECT_101,
    )
    padded_mask = cv2.copyMakeBorder(
        mask.astype(np.uint8),
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    ).astype(np.float32)
    return padded_image, padded_mask


def crop_pair(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    prefer_positive: bool,
    positive_crop_probability: float,
) -> Tuple[np.ndarray, np.ndarray]:
    crop_size = int(crop_size)
    image_rgb, mask = pad_pair_to_minimum(image_rgb, mask, crop_size)
    height, width = image_rgb.shape[:2]
    max_left = max(width - crop_size, 0)
    max_top = max(height - crop_size, 0)

    left = 0
    top = 0
    if max_left > 0:
        left = random.randint(0, max_left)
    if max_top > 0:
        top = random.randint(0, max_top)

    if prefer_positive and random.random() < positive_crop_probability:
        positive_locations = np.argwhere(mask > 0.5)
        if positive_locations.size > 0:
            point_y, point_x = positive_locations[random.randrange(len(positive_locations))]
            jitter = max(crop_size // 6, 8)
            point_x = int(point_x + random.randint(-jitter, jitter))
            point_y = int(point_y + random.randint(-jitter, jitter))
            left = int(np.clip(point_x - crop_size // 2, 0, max_left))
            top = int(np.clip(point_y - crop_size // 2, 0, max_top))

    cropped_image = image_rgb[top : top + crop_size, left : left + crop_size]
    cropped_mask = mask[top : top + crop_size, left : left + crop_size]
    return np.ascontiguousarray(cropped_image), np.ascontiguousarray(cropped_mask)


def augment_pair(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.45:
        image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.7:
        height, width = image_rgb.shape[:2]
        center = (width / 2.0, height / 2.0)
        angle = random.uniform(-7.0, 7.0)
        scale = random.uniform(0.94, 1.06)
        tx = random.uniform(-0.04, 0.04) * width
        ty = random.uniform(-0.04, 0.04) * height
        transform = cv2.getRotationMatrix2D(center, angle, scale)
        transform[:, 2] += [tx, ty]
        image_rgb = cv2.warpAffine(
            image_rgb,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    image = image_rgb.astype(np.float32)
    if random.random() < 0.8:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10.0, 10.0)
        image = image * alpha + beta
    if random.random() < 0.35:
        hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.85, 1.15)
        hsv[:, :, 2] *= random.uniform(0.9, 1.1)
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if random.random() < 0.25:
        sigma = random.uniform(0.3, 1.0)
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    if random.random() < 0.25:
        noise = np.random.normal(0.0, random.uniform(2.0, 6.0), image.shape).astype(np.float32)
        image = image + noise
    image_rgb = np.clip(image, 0, 255).astype(np.uint8)
    return image_rgb, mask


def collect_image_label_pairs(root: Path, split: str, variants: Sequence[str]) -> List[Tuple[Path, Path]]:
    image_dir = root / "img" / split
    label_dir = root / "label" / split
    label_paths = {path.stem: path for path in iter_image_files(label_dir)}
    allowed = {token.strip() for token in variants if token.strip()}
    pairs: List[Tuple[Path, Path]] = []
    for image_path in iter_image_files(image_dir):
        variant = infer_variant(image_path.stem)
        if variant is not None and allowed and variant not in allowed:
            continue
        label_path = label_paths.get(image_path.stem)
        if label_path is not None:
            pairs.append((image_path, label_path))
    return pairs


class MarsRockDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        variants: Sequence[str],
        augment: bool,
        image_size: Optional[int] = 512,
        limit: Optional[int] = None,
        target_class: str = "rock",
        pairs: Optional[Sequence[Tuple[Path, Path]]] = None,
        crop_size: Optional[int] = None,
        positive_crop_probability: float = 0.75,
        patches_per_image: int = 1,
    ) -> None:
        ensure_torch()
        self.root = root
        self.split = split
        self.augment = augment
        self.image_size = None if image_size in (None, 0) else int(image_size)
        self.crop_size = None if crop_size in (None, 0) else int(crop_size)
        self.positive_crop_probability = float(positive_crop_probability)
        self.patches_per_image = max(1, int(patches_per_image))
        self.target_class_id = resolve_target_class_id(target_class)
        resolved_pairs = list(pairs) if pairs is not None else collect_image_label_pairs(root, split, variants=variants)
        if limit is not None:
            resolved_pairs = resolved_pairs[:limit]
        if not resolved_pairs:
            image_dir = root / "img" / split
            label_dir = root / "label" / split
            raise FileNotFoundError(f"No paired samples found under {image_dir} and {label_dir}")
        self.pairs = resolved_pairs

    def __len__(self) -> int:
        if self.augment:
            return len(self.pairs) * self.patches_per_image
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        image_path, label_path = self.pairs[index % len(self.pairs)]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None or label_image is None:
            raise FileNotFoundError(f"Failed to load pair: {image_path}, {label_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = label_image_to_binary(label_image, target_class_id=self.target_class_id).astype(np.float32)
        if self.augment:
            if self.crop_size is not None:
                image_rgb, mask = crop_pair(
                    image_rgb,
                    mask,
                    crop_size=self.crop_size,
                    prefer_positive=True,
                    positive_crop_probability=self.positive_crop_probability,
                )
            image_rgb, mask = augment_pair(image_rgb, mask)
        elif self.crop_size is not None:
            image_rgb, mask = crop_pair(
                image_rgb,
                mask,
                crop_size=self.crop_size,
                prefer_positive=False,
                positive_crop_probability=0.0,
            )
        image_rgb, mask = resize_pair(image_rgb, mask, self.image_size)
        image_tensor = image_to_tensor(image_rgb)
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()
        return image_tensor, mask_tensor


def _resolve_sample_split(tags: Optional[Sequence[str]]) -> str:
    normalized = {tag.strip().lower() for tag in tags or []}
    if "val" in normalized or "validation" in normalized:
        return "val"
    if "test" in normalized:
        return "test"
    return "train"


def _sample_matches_split(tags: Optional[Sequence[str]], split: str) -> bool:
    return _resolve_sample_split(tags) == split


def _cache_mask_from_array(mask: np.ndarray, cache_root: Path, sample_id: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    mask_path = cache_root / f"{sample_id}.png"
    cv2.imwrite(str(mask_path), np.asarray(mask))
    return mask_path


def load_hub_dataset(
    repo_id: str,
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> object:
    load_from_hub = ensure_fiftyone()
    resolved_name = dataset_name or f"s5mars_{uuid4().hex[:8]}"
    import huggingface_hub

    original_snapshot_download = huggingface_hub.snapshot_download

    def snapshot_download_single_worker(*args, **kwargs):
        kwargs.setdefault("max_workers", 1)
        return original_snapshot_download(*args, **kwargs)

    huggingface_hub.snapshot_download = snapshot_download_single_worker
    try:
        return load_from_hub(
            repo_id,
            name=resolved_name,
            overwrite=True,
            persistent=False,
            max_samples=max_samples,
            token=token,
        )
    finally:
        huggingface_hub.snapshot_download = original_snapshot_download


def load_hub_segmentation_pairs(
    repo_id: str,
    split: Optional[str] = None,
    label_field: str = "ground_truth",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[object, List[Tuple[Path, Path]]]:
    resolved_name = dataset_name or f"s5mars_{split}_{uuid4().hex[:8]}"
    dataset = load_hub_dataset(
        repo_id=repo_id,
        max_samples=max_samples,
        token=token,
        dataset_name=resolved_name,
    )

    cache_root = Path(".cache") / "hub_masks" / resolved_name
    pairs: List[Tuple[Path, Path]] = []
    for sample in dataset:
        if split is not None and not _sample_matches_split(getattr(sample, "tags", None), split):
            continue
        try:
            segmentation = sample[label_field]
        except KeyError as exc:
            dataset.delete()
            raise KeyError(f"Field {label_field!r} was not found in the loaded FiftyOne dataset.") from exc
        if segmentation is None or not segmentation.has_mask:
            continue
        image_path = Path(sample.filepath)
        if segmentation.mask_path:
            mask_path = Path(segmentation.mask_path)
        else:
            mask = segmentation.get_mask()
            if mask is None:
                continue
            mask_path = _cache_mask_from_array(mask, cache_root, str(sample.id))
        pairs.append((image_path, mask_path))
    return dataset, pairs


def _copy_or_convert_image(source_path: Path, destination_path: Path) -> None:
    if source_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        shutil.copy2(source_path, destination_path)
        return

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read source image: {source_path}")
    cv2.imwrite(str(destination_path), image)


def prepare_s5mars_dataset(
    output_root: Path,
    repo_id: str = DEFAULT_S5MARS_REPO_ID,
    target_class: str = "rock",
    label_field: str = "ground_truth",
    dataset_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    token: Optional[str] = None,
) -> Dict[str, object]:
    load_from_hub = ensure_fiftyone()
    target_class_id = resolve_target_class_id(target_class)
    target_class_name = S5MARS_MASK_TARGETS[target_class_id]
    output_root = output_root.resolve()

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists and is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "img" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "label" / split).mkdir(parents=True, exist_ok=True)

    resolved_dataset_name = dataset_name or f"s5mars_{target_class_name}_tmp"
    dataset = load_from_hub(
        repo_id,
        max_samples=max_samples,
        name=resolved_dataset_name,
        overwrite=True,
        persistent=False,
        token=token,
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped_without_mask = 0
    written_samples = 0

    try:
        for sample in dataset:
            try:
                segmentation = sample[label_field]
            except KeyError as exc:
                raise KeyError(
                    f"Field {label_field!r} was not found in the loaded FiftyOne dataset."
                ) from exc

            if segmentation is None or not segmentation.has_mask:
                skipped_without_mask += 1
                continue

            mask = segmentation.get_mask()
            if mask is None:
                skipped_without_mask += 1
                continue

            split = _resolve_sample_split(getattr(sample, "tags", None))
            source_path = Path(sample.filepath)
            source_suffix = source_path.suffix.lower() if source_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else ".png"
            sample_suffix = str(sample.id)[-8:]
            destination_stem = f"{source_path.stem}_{sample_suffix}"
            destination_image = output_root / "img" / split / f"{destination_stem}{source_suffix}"
            destination_mask = output_root / "label" / split / f"{destination_stem}.png"

            _copy_or_convert_image(source_path, destination_image)
            binary_mask = segmentation_mask_to_binary(mask, target_class_id)
            cv2.imwrite(str(destination_mask), binary_mask)

            split_counts[split] += 1
            written_samples += 1
    finally:
        try:
            dataset.delete()
        except Exception:
            pass

    metadata = {
        "repo_id": repo_id,
        "target_class": {"id": target_class_id, "name": target_class_name},
        "label_field": label_field,
        "output_root": str(output_root),
        "written_samples": written_samples,
        "skipped_without_mask": skipped_without_mask,
        "splits": split_counts,
        "max_samples": max_samples,
    }
    (output_root / "dataset_info.json").write_text(json.dumps(metadata, indent=2))
    return metadata
