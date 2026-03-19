from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import DEFAULT_VARIANTS, IMAGE_MEAN, IMAGE_STD, TrainMetrics
from data import MarsRockDataset, collect_image_label_pairs, load_hub_dataset
from deps import DataLoader, F, cv2, deeplabv3_mobilenet_v3_large, ensure_torch, np, torch, lraspp_mobilenet_v3_large

DEFAULT_MODEL_NAME = "deeplabv3_mobilenet_v3_large"


def choose_group_count(num_channels: int, preferred_groups: int = 8) -> int:
    for group_count in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


def replace_batchnorm_with_groupnorm(module: "torch.nn.Module", preferred_groups: int = 8) -> "torch.nn.Module":
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            group_count = choose_group_count(child.num_features, preferred_groups=preferred_groups)
            setattr(module, name, torch.nn.GroupNorm(group_count, child.num_features))
        else:
            replace_batchnorm_with_groupnorm(child, preferred_groups=preferred_groups)
    return module


def choose_device(device_arg: str) -> "torch.device":
    ensure_torch()
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_model(model_name: str = DEFAULT_MODEL_NAME) -> "torch.nn.Module":
    ensure_torch()
    kwargs = {"num_classes": 1}
    if model_name == "deeplabv3_mobilenet_v3_large":
        try:
            model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, **kwargs)
        except TypeError:
            model = deeplabv3_mobilenet_v3_large(pretrained=False, **kwargs)
        return replace_batchnorm_with_groupnorm(model)
    if model_name == "lraspp_mobilenet_v3_large":
        try:
            model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, **kwargs)
        except TypeError:
            model = lraspp_mobilenet_v3_large(pretrained=False, **kwargs)
        return replace_batchnorm_with_groupnorm(model)
    raise ValueError(f"Unsupported model architecture: {model_name}")


def model_logits(model: "torch.nn.Module", images: "torch.Tensor") -> "torch.Tensor":
    output = model(images)
    if isinstance(output, dict):
        return output["out"]
    return output


def dice_loss_from_logits(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (probabilities * targets).sum(dim=1) + 1.0
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1) + 1.0
    return 1.0 - (numerator / denominator).mean()


def balanced_bce_loss_from_logits(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    positive = targets.sum().detach()
    total = torch.tensor(float(targets.numel()), dtype=targets.dtype, device=targets.device)
    negative = torch.clamp(total - positive, min=1.0)
    pos_weight = torch.clamp(negative / torch.clamp(positive, min=1.0), min=1.0, max=12.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_loss_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> "torch.Tensor":
    probabilities = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    return (alpha_factor * modulating * ce).mean()


def tversky_loss_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.3,
    beta: float = 0.7,
) -> "torch.Tensor":
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)
    true_positive = (probabilities * targets).sum(dim=1)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=1)
    score = (true_positive + 1.0) / (true_positive + alpha * false_positive + beta * false_negative + 1.0)
    return 1.0 - score.mean()


def segmentation_loss_from_logits(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    return (
        0.35 * balanced_bce_loss_from_logits(logits, targets)
        + 0.30 * focal_loss_from_logits(logits, targets)
        + 0.35 * tversky_loss_from_logits(logits, targets)
    )


def segmentation_metrics_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.45,
) -> TrainMetrics:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    intersection = (predictions * targets).sum().item()
    pred_sum = predictions.sum().item()
    target_sum = targets.sum().item()
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)
    iou = (intersection + 1.0) / (union + 1.0)
    loss = segmentation_loss_from_logits(logits, targets).item()
    return TrainMetrics(loss=float(loss), dice=float(dice), iou=float(iou))


def run_epoch(
    model: "torch.nn.Module",
    loader: "DataLoader",
    device: "torch.device",
    optimizer: Optional["torch.optim.Optimizer"],
    scheduler: Optional[object] = None,
    grad_clip_norm: float = 1.0,
) -> TrainMetrics:
    ensure_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model_logits(model, images)
            loss = segmentation_loss_from_logits(logits, masks)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            metrics = segmentation_metrics_from_logits(logits.detach(), masks.detach())
            total_loss += loss.item()
            total_dice += metrics.dice
            total_iou += metrics.iou
            total_batches += 1
    if total_batches == 0:
        return TrainMetrics(loss=0.0, dice=0.0, iou=0.0)
    return TrainMetrics(
        loss=total_loss / total_batches,
        dice=total_dice / total_batches,
        iou=total_iou / total_batches,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    val_metrics: TrainMetrics,
    config: Dict[str, object],
) -> None:
    ensure_torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": config.get("model_name", DEFAULT_MODEL_NAME),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics.__dict__,
            "config": config,
            "normalization": {"mean": IMAGE_MEAN.tolist(), "std": IMAGE_STD.tolist()},
        },
        checkpoint_path,
    )


def train_model(args: argparse.Namespace) -> None:
    ensure_torch()
    set_seed(args.seed)
    variants = tuple(token.strip() for token in args.variants.split(",") if token.strip()) or DEFAULT_VARIANTS
    device = choose_device(args.device)
    dataset_root = Path(args.dataset_root)
    hub_datasets_to_cleanup: List[object] = []

    if args.hf_repo_id:
        print(
            f"loading Hugging Face dataset {args.hf_repo_id!r} through FiftyOne "
            f"with max_samples={args.hf_max_samples}"
        )
        train_dataset_handle = load_hub_dataset(
            repo_id=args.hf_repo_id,
            max_samples=args.hf_max_samples,
            token=args.hf_token,
            dataset_name=args.hf_dataset_name,
        )
        hub_datasets_to_cleanup.append(train_dataset_handle)
        train_pairs = []
        val_pairs = []
        for sample in train_dataset_handle:
            split_name = "train"
            tags = {tag.strip().lower() for tag in getattr(sample, "tags", None) or []}
            if "val" in tags or "validation" in tags:
                split_name = "val"
            elif "test" in tags:
                split_name = "test"
            try:
                segmentation = sample[args.hf_label_field]
            except KeyError:
                continue
            if segmentation is None or not segmentation.has_mask:
                continue
            image_path = Path(sample.filepath)
            if segmentation.mask_path:
                mask_path = Path(segmentation.mask_path)
            else:
                mask = segmentation.get_mask()
                if mask is None:
                    continue
                cache_root = Path(".cache") / "hub_masks" / (args.hf_dataset_name or "s5mars_direct")
                cache_root.mkdir(parents=True, exist_ok=True)
                mask_path = cache_root / f"{sample.id}.png"
                cv2.imwrite(str(mask_path), np.asarray(mask))
            if split_name == "train":
                train_pairs.append((image_path, mask_path))
            elif split_name == "val":
                val_pairs.append((image_path, mask_path))

        if args.train_limit is not None:
            train_pairs = train_pairs[: args.train_limit]
        if args.val_limit is not None:
            val_pairs = val_pairs[: args.val_limit]
    else:
        train_pairs = collect_image_label_pairs(dataset_root, "train", variants=variants)
        if args.train_limit is not None:
            train_pairs = train_pairs[: args.train_limit]

        val_pairs = collect_image_label_pairs(dataset_root, "val", variants=variants)
        if args.val_limit is not None:
            val_pairs = val_pairs[: args.val_limit]

    if not train_pairs:
        if args.hf_repo_id:
            raise FileNotFoundError("No training-tagged samples were available from the loaded Hugging Face dataset.")
        raise FileNotFoundError(f"No training samples found under {dataset_root / 'img' / 'train'}")

    if not val_pairs:
        if len(train_pairs) < 2:
            raise FileNotFoundError("Need at least two training samples to derive a validation split.")
        derived_val_count = max(1, min(len(train_pairs) // 10, 200))
        val_pairs = train_pairs[-derived_val_count:]
        train_pairs = train_pairs[:-derived_val_count]
        source_label = args.hf_repo_id if args.hf_repo_id else str(dataset_root)
        print(
            f"no explicit val split found in {source_label}; "
            f"using {len(val_pairs)} held-out training samples for validation"
        )

    try:
        train_dataset = MarsRockDataset(
            dataset_root,
            "train",
            variants=variants,
            augment=True,
            image_size=args.image_size,
            target_class=args.target_class,
            pairs=train_pairs,
            crop_size=args.crop_size,
            positive_crop_probability=args.positive_crop_probability,
            patches_per_image=args.patches_per_image,
        )
        val_dataset = MarsRockDataset(
            dataset_root,
            "val",
            variants=variants,
            augment=False,
            image_size=args.image_size,
            target_class=args.target_class,
            pairs=val_pairs,
            crop_size=None,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=len(train_dataset) > args.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model(args.model_arch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=max(args.epochs, 1),
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=50.0,
        )
        checkpoint_path = Path(args.output_checkpoint)
        config = {
            "model_name": args.model_arch,
            "dataset_root": str(dataset_root),
            "hf_repo_id": args.hf_repo_id,
            "hf_label_field": args.hf_label_field,
            "hf_max_samples": args.hf_max_samples,
            "variants": list(variants),
            "target_class": args.target_class,
            "image_size": args.image_size,
            "crop_size": args.crop_size,
            "patches_per_image": args.patches_per_image,
            "positive_crop_probability": args.positive_crop_probability,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        }

        source_label = args.hf_repo_id if args.hf_repo_id else str(dataset_root)
        print(
            f"training on {len(train_dataset)} train samples and {len(val_dataset)} val samples "
            f"from {source_label} using model={args.model_arch!r} target_class={args.target_class!r} "
            f"image_size={args.image_size} crop_size={args.crop_size}"
        )

        best_iou = -1.0
        best_loss = float("inf")
        history: List[Dict[str, object]] = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                scheduler=scheduler,
                grad_clip_norm=args.grad_clip_norm,
            )
            val_metrics = run_epoch(model, val_loader, device, optimizer=None, scheduler=None, grad_clip_norm=0.0)
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics.__dict__,
                "val": val_metrics.__dict__,
                "seconds": round(time.time() - start, 2),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            }
            history.append(epoch_record)
            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_metrics.loss:.4f} train_dice={train_metrics.dice:.4f} train_iou={train_metrics.iou:.4f} "
                f"val_loss={val_metrics.loss:.4f} val_dice={val_metrics.dice:.4f} val_iou={val_metrics.iou:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            if val_metrics.iou > best_iou + 1e-6 or (abs(val_metrics.iou - best_iou) <= 1e-6 and val_metrics.loss < best_loss):
                best_iou = val_metrics.iou
                best_loss = val_metrics.loss
                save_checkpoint(checkpoint_path, model, optimizer, epoch, val_metrics, config)
                print(f"saved best checkpoint to {checkpoint_path}")

        history_path = checkpoint_path.with_suffix(".history.json")
        history_path.write_text(json.dumps(history, indent=2))
        print(f"wrote training history to {history_path}")
    finally:
        for dataset in hub_datasets_to_cleanup:
            try:
                dataset.delete()
            except Exception:
                pass


def load_checkpoint_model(checkpoint_path: Path, device: "torch.device") -> Tuple["torch.nn.Module", Dict[str, object]]:
    ensure_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name") or checkpoint.get("config", {}).get("model_name", DEFAULT_MODEL_NAME)
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "epoch": checkpoint.get("epoch"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "config": checkpoint.get("config", {}),
    }
    return model, metadata
