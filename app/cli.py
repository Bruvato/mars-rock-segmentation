from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .common import DEFAULT_S5MARS_REPO_ID, DEFAULT_VARIANTS
from .data import prepare_s5mars_dataset
from .prediction import default_output_paths, predict_image
from .training import train_model


def _dotenv_candidates() -> Sequence[Path]:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    return (
        Path.cwd() / ".env",
        project_root / ".env",
        package_dir / ".env",
    )


def _parse_dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return value
    if value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _read_token_from_dotenv(dotenv_path: Path) -> Optional[str]:
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "hf_token"}:
            continue
        value = _parse_dotenv_value(raw_value)
        if value:
            return value
    return None


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        os.environ.setdefault("HF_TOKEN", explicit_token)
        return explicit_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
    if env_token:
        os.environ.setdefault("HF_TOKEN", env_token)
        return env_token

    seen_paths = set()
    for dotenv_path in _dotenv_candidates():
        resolved_path = dotenv_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        token = _read_token_from_dotenv(dotenv_path)
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    return None


def add_predict_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("predict", help="Run Mars rock segmentation on an input image.")
    parser.add_argument("--input", "-input", required=True, type=Path, help="Path to the input image.")
    parser.add_argument("--output-image", "-output-image", type=Path, default=None, help="Annotated image output path.")
    parser.add_argument("--output-json", "-output-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--output-mask", type=Path, default=None, help="Binary segmentation mask output path.")
    parser.add_argument(
        "--output-height-image",
        type=Path,
        default=None,
        help="Filtered annotation image for rocks heuristically estimated above --min-height-cm.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/mars_rock_lraspp.pt"),
        help="Model checkpoint path created by the train command.",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Rock probability threshold.")
    parser.add_argument("--tile-size", type=int, default=512, help="Inference tile size.")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Inference tile overlap in pixels.")
    parser.add_argument("--show-labels", action="store_true", help="Draw per-instance confidence labels on the overlay.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--min-height-cm",
        type=float,
        default=10.0,
        help="Minimum heuristic rock height to include in the separate height-annotated output.",
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="Optional camera height in meters used for single-image height heuristics.",
    )
    parser.add_argument(
        "--vfov-deg",
        type=float,
        default=None,
        help="Optional vertical field of view in degrees used for single-image height heuristics.",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=None,
        help="Optional camera pitch in degrees. When omitted, the predictor tries to infer it from the horizon.",
    )
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug output directory.")


def add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train the binary rock segmenter on MarsData, a prepared S5Mars export, or S5Mars directly via FiftyOne.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("MarsData"), help="Dataset root with img/ and label/ directories.")
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo to load directly through FiftyOne, for example Voxel51/S5Mars.",
    )
    parser.add_argument(
        "--hf-label-field",
        type=str,
        default="ground_truth",
        help="FiftyOne segmentation field to read when training directly from a Hugging Face dataset.",
    )
    parser.add_argument(
        "--hf-max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of Hugging Face samples to load before splitting by tags.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or hf_token from a local .env file if present.",
    )
    parser.add_argument("--hf-dataset-name", type=str, default=None, help="Optional temporary FiftyOne dataset name.")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("models/mars_rock_lraspp.pt"),
        help="Where to save the best checkpoint.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="deeplabv3_mobilenet_v3_large",
        choices=("deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large"),
        help="Segmentation architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated legacy MarsData variants to use. Non-variant S5Mars exports are loaded automatically.",
    )
    parser.add_argument("--target-class", type=str, default="rock", help="S5Mars class name or id to treat as foreground.")
    parser.add_argument("--image-size", type=int, default=512, help="Resize training images and masks to this square size. Use 0 to keep original resolution.")
    parser.add_argument("--crop-size", type=int, default=512, help="Optional training crop size before resize. Use 0 to disable crop sampling.")
    parser.add_argument("--patches-per-image", type=int, default=3, help="How many random training patches to sample from each image per epoch.")
    parser.add_argument(
        "--positive-crop-probability",
        type=float,
        default=0.85,
        help="Probability of centering a training crop around positive rock pixels when available.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm. Use 0 to disable.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional training sample cap.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional validation sample cap.")


def add_prepare_s5mars_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "prepare-s5mars",
        help="Download Voxel51/S5Mars through FiftyOne and convert it into the local img/label training layout.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_S5MARS_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--output-root", type=Path, default=Path("data/s5mars_rock"), help="Where to write the prepared local dataset.")
    parser.add_argument("--target-class", type=str, default="rock", help="S5Mars class name or id to extract as the positive class.")
    parser.add_argument("--label-field", type=str, default="ground_truth", help="FiftyOne segmentation field to export.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional temporary FiftyOne dataset name.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap while prototyping.")
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or hf_token from a local .env file if present.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared dataset at the output path.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    known_commands = {"train", "predict", "prepare-s5mars"}
    if effective_argv and effective_argv[0] not in known_commands:
        effective_argv = ["predict", *effective_argv]

    parser = argparse.ArgumentParser(
        description=(
            "Mars rock segmentation CLI. "
            "The Hugging Face asset used here, Voxel51/S5Mars, is a dataset that can be converted into the local binary training layout."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_predict_parser(subparsers)
    add_train_parser(subparsers)
    add_prepare_s5mars_parser(subparsers)
    args = parser.parse_args(effective_argv)

    if (
        args.command == "predict"
        and (
            args.output_image is None
            or args.output_json is None
            or args.output_mask is None
            or args.output_height_image is None
        )
    ):
        default_image, default_json, default_mask, default_height_image = default_output_paths(
            args.input,
            min_height_cm=float(args.min_height_cm),
        )
        if args.output_image is None:
            args.output_image = default_image
        if args.output_json is None:
            args.output_json = default_json
        if args.output_mask is None:
            args.output_mask = default_mask
        if args.output_height_image is None:
            args.output_height_image = default_height_image

    if hasattr(args, "hf_token"):
        args.hf_token = resolve_hf_token(args.hf_token)

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train_model(args)
        return
    if args.command == "prepare-s5mars":
        metadata = prepare_s5mars_dataset(
            output_root=args.output_root,
            repo_id=args.repo_id,
            target_class=args.target_class,
            label_field=args.label_field,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
            token=args.hf_token,
        )
        print(json.dumps(metadata, indent=2))
        return
    predict_image(args)
