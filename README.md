# Mars Rock Segmentation

This project trains and runs a binary rock-segmentation pipeline for Mars rover imagery.

The Hugging Face asset referenced in the prompt, [`Voxel51/S5Mars`](https://huggingface.co/datasets/Voxel51/S5Mars), is a dataset rather than a pretrained model. The codebase now includes a `prepare-s5mars` command that downloads it through FiftyOne and converts the `rock` class into the local `img/{split}` and `label/{split}` layout that the trainer already understands.

## Quick Start

```bash
uv sync
```

Run inference with the existing checkpoint:

```bash
.venv/bin/python main.py predict --input inputs/input1.png --checkpoint models/s5mars_long.pt
```

The `predict` command writes four artifacts by default:

- `outputs/{stem}_annotated.png`: the standard segmentation overlay
- `outputs/{stem}_mask.png`: the binary mask
- `outputs/{stem}.json`: per-rock metadata, including heuristic height estimates
- `outputs/{stem}_rocks_over_10cm.png`: a separate outline-and-label image for rocks whose estimated height exceeds `--min-height-cm`

You can improve the single-image height heuristic when camera metadata is known:

```bash
.venv/bin/python main.py predict \
  --input inputs/input1.png \
  --checkpoint models/s5mars_long.pt \
  --camera-height-m 1.8 \
  --vfov-deg 45 \
  --pitch-deg 32 \
  --min-height-cm 10
```

Train on the existing local dataset:

```bash
.venv/bin/python main.py train --dataset-root MarsData
```

Prepare `S5Mars` from Hugging Face and train on it:

```bash
.venv/bin/python main.py prepare-s5mars --output-root data/s5mars_rock
.venv/bin/python main.py train --dataset-root data/s5mars_rock --target-class rock --image-size 512
```

## Commands

- `predict`: run the binary rock detector and write an annotated image, binary mask, JSON payload, and a separate `>10 cm` outline image
- `train`: train the LR-ASPP model on a local `img/` and `label/` dataset root
- `prepare-s5mars`: download `Voxel51/S5Mars` with FiftyOne and export a binary `rock` mask dataset for training

## Notes

- The legacy `MarsData` variant filter still works for filenames ending in `raw`, `warp`, `sup`, `sdown`, `tleft`, or `tup`.
- Prepared S5Mars exports are non-variant datasets, so they bypass variant filtering automatically.
- `--target-class` accepts either a class name like `rock` or a numeric S5Mars class id.
