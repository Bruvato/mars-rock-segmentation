from __future__ import annotations

import os
from pathlib import Path

try:
    import cv2
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_holes, remove_small_objects
    from skimage.segmentation import watershed
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Image dependencies are not installed. Install them with `uv sync`."
    ) from exc

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass

    deeplabv3_mobilenet_v3_large = None
    lraspp_mobilenet_v3_large = None


def ensure_torch() -> None:
    if (
        torch is None
        or F is None
        or DataLoader is None
        or lraspp_mobilenet_v3_large is None
        or deeplabv3_mobilenet_v3_large is None
    ):
        raise RuntimeError(
            "PyTorch dependencies are not installed. Install them with `uv sync` or `uv add torch torchvision`."
        )


def ensure_fiftyone():
    cache_dir = Path.cwd() / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    try:
        from fiftyone.utils.huggingface import load_from_hub
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FiftyOne and Hugging Face dependencies are not installed. Install them with `uv sync`."
        ) from exc

    return load_from_hub
