# Experiment Part 1: CIFAR-10 cGAN Ablation

This folder contains **Experiment Part 1**: CIFAR-10 conditional GAN ablation studies (concat vs projection conditioning, BCE vs hinge loss). Implemented in **PyTorch**.

## Contents

- **`test.py`** — Main script: trains cGAN variants, saves samples and (optionally) computes FID/KID. Headless-safe; all figures saved as JPG under `./outputs_cgan_ablation_jpg/`.
- **`relabel_grids.py`** — Adds class labels to saved sample grids (CIFAR-10 class names and sample index).
- **`data/`** — Place CIFAR-10 data here (e.g. `cifar-10-batches-py/`). This directory is gitignored.
- **`outputs_cgan_ablation_jpg/`** — Generated samples and logs from `test.py`.
- **`*.log`** — Run logs.

## Requirements

- Python 3
- PyTorch, torchvision
- Optional: `torchmetrics` (for FID/KID), matplotlib

## How to run

From the **repository root**:

```bash
cd experiment_part1
python test.py
```

Or from inside `experiment_part1`:

```bash
python test.py
```

To add labels to the saved grid images:

```bash
python relabel_grids.py
```

## Outputs

- Sample images per run (concat/proj × BCE/hinge) in `outputs_cgan_ablation_jpg/`.
- FID/KID values printed if `torchmetrics` is available.

For the overall project structure and Part 2–3 (multi-dataset pipeline), see the main [README.md](../README.md) in the repository root.
