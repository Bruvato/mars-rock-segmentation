# Mars Rock Segmentation

This project trains and runs a binary rock-segmentation pipeline for Mars rover imagery.

The Hugging Face asset referenced in the prompt, [`Voxel51/S5Mars`](https://huggingface.co/datasets/Voxel51/S5Mars), is a dataset rather than a pretrained model. The codebase includes a `prepare-s5mars` command that downloads it through FiftyOne and converts the `rock` class into the local `img/{split}` and `label/{split}` layout that the trainer already understands.

## Project Layout

```text
.
├── pyproject.toml
├── README.md
├── src/
│   ├── interview.py
│   ├── cli.py
│   ├── prediction.py
│   └── ...
├── inputs/
├── models/
└── outputs/
```

## Quick Start

Install the locked environment and confirm the CLI is available:

```bash
uv sync
uv run interview --help
```

## Run Inference

Run inference with an existing checkpoint:

```bash
uv run interview predict --input inputs/input1.png --checkpoint models/s5mars_long.pt
```

The `predict` command writes four artifacts by default:

- `outputs/{stem}_annotated.png`: the standard segmentation overlay
- `outputs/{stem}_mask.png`: the binary mask
- `outputs/{stem}.json`: per-rock metadata, including heuristic height estimates
- `outputs/{stem}_rocks_over_10cm.png`: a separate outline-and-label image for rocks whose estimated height exceeds `--min-height-cm`

You can improve the single-image height heuristic when camera metadata is known:

```bash
uv run interview predict \
  --input inputs/input1.png \
  --checkpoint models/s5mars_long.pt \
  --camera-height-m 1.8 \
  --vfov-deg 45 \
  --pitch-deg 32 \
  --min-height-cm 10
```

If you prefer the module form, this is equivalent:

```bash
uv run python -m interview predict --input inputs/input1.png --checkpoint models/s5mars_long.pt
```

## Training

Train on the existing local dataset:

```bash
uv run interview train --dataset-root MarsData
```

Prepare `S5Mars` from Hugging Face and train on it:

```bash
uv run interview prepare-s5mars --output-root data/s5mars_rock
uv run interview train --dataset-root data/s5mars_rock --target-class rock --image-size 512
```

## Commands

- `predict`: run the binary rock detector and write an annotated image, binary mask, JSON payload, and a separate `>10 cm` outline image
- `train`: train the segmenter on a local `img/` and `label/` dataset root
