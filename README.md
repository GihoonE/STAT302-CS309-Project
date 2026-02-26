# Conditional GAN for Synthetic Data: Evaluation Framework

**STATS302 / COMPSCI309 Group Project**  
**Authors:** Gihun Lee, Sam Akhmedjonov, and Sattor Khamroev

**Abstract.** We provide a modular, reproducible codebase for training and evaluating conditional GANs (cGANs) on image data. **Experiment Part 1** (CIFAR-10) implements cGAN ablation studies (conditioning: concat vs projection; loss: BCE vs hinge) with PyTorch and optional FID/KID. **Experiment Part 2–3** (TensorFlow) implement a multi-dataset pipeline: data preparation, cGAN training with optional label noise, downstream classification with mixed real/synthetic data, and FID/KID evaluation. The repository is organized for clarity and NeurIPS-style code submission.

-

## Code structure (by experiment)

| Part | Folder | Description |
|------|--------|-------------|
| **Run scripts** | `run_experiment/` | **Start here for reproducibility:** `run_part1.py`, `run_part23.py` (run from repo root). |
| **Part 1** | `experiment_part1/` | CIFAR-10 cGAN ablation (PyTorch): concat vs projection, BCE vs hinge; outputs in `outputs/part1/`. |
| **Part 2–3** | `experiment_part23/` | Multi-dataset pipeline (TensorFlow): data prep, cGAN training, label-noise experiments, downstream mixing, FID/KID. |
| **CLI** | `tools/` | Entry points for Part 2–3: `run_pipeline.py`, `test_data_load.py`. |
| **Outputs** | `outputs/`, `logs/` | All experiment outputs: `outputs/part1` (Part 1), `outputs/part23` (Part 2–3), `outputs/grids` (grid figures). |

---

## Directory structure

```
STAT302-CS309-Project/
├── run_experiment/           # Run scripts (reproduce from repo root)
│   ├── run_part1.py          # Part 1: CIFAR-10 ablation (seed=17)
│   ├── run_part23.py         # Part 2–3: pipeline CLI (seed=42)
│   └── README.md
├── experiment_part1/         # Experiment Part 1: CIFAR-10 cGAN ablation (PyTorch)
│   ├── test.py               # Main script: ablation (concat/proj, BCE/hinge)
│   ├── relabel_grids.py      # Add class labels to sample grids
│   ├── data/                 # CIFAR-10 data (gitignored; download separately)
│   └── *.log
├── experiment_part23/        # Experiment Part 2–3: multi-dataset pipeline (TensorFlow)
│   ├── __init__.py
│   ├── datasets_config.py    # Dataset keys, slugs, formats
│   ├── data_prep.py          # prepare_ds(): train/val/test loaders
│   ├── cgan.py               # Generator, Discriminator, train_cgan, label noise
│   ├── classifier.py         # EfficientNetB0 + run_ratio_experiments (mixing)
│   ├── fid_kid.py            # FID/KID evaluation (Inception V3)
│   ├── run_pipeline.py       # run_pipeline_one(): full pipeline entrypoint
│   ├── test_multi_datasets.py
│   └── README.md             # Usage details
├── tools/
│   ├── run_pipeline.py       # CLI: single run or label-noise sweep (Part 2–3)
│   ├── test_data_load.py     # Sanity check dataset download and loading
│   ├── grid.py
│   └── grid_epoch.py
├── outputs/                  # All experiment outputs (unified)
│   ├── part1/                # Part 1: CIFAR-10 ablation (samples, loss curves, etc.)
│   ├── part23/               # Part 2–3: per-dataset (fake_samples, CSVs, metrics, sweeps)
│   │   └── {dataset_key}/
│   │       ├── fake_samples/baseline/epoch_*/
│   │       ├── fake_samples/label_noise_p{XX}/epoch_*
│   │       ├── cnn_ratio_results_*.csv, metrics/, sweeps/
│   └── grids/                # Grid figures from tools/grid_epoch.py
├── logs/
├── requirements.txt
└── README.md                 # This file
```

---

## Environment and dependencies

- **Python 3** with TensorFlow 2.x. Optional: `pandas` for CSV/sweep output; `kagglehub` for Kaggle datasets (MNIST uses built-in Keras, no Kaggle).
- Install from project root:

```bash
pip install -r requirements.txt
```

Core dependencies include `tensorflow`, `kagglehub`, `scipy`, `numpy`. For CLI sweep and CSV handling, `pandas` is used by `tools/run_pipeline.py`.

- **Part 1 (experiment_part1):** Uses PyTorch and (optionally) `torchmetrics` for FID/KID. CIFAR-10 data must be placed under `experiment_part1/data/` (e.g. `cifar-10-batches-py/`). See `experiment_part1/README.md` for run commands.
- **Part 2–3 (experiment_part23):** TensorFlow + kagglehub (except MNIST, which uses Keras built-in). Run from repo root so that the `experiment_part23` package is importable.

## Running the experiments

From repo root. All outputs under `outputs/part1/` and `outputs/part23/`.

```bash
# Run all: Part 1 (seed=17) + Part 2–3 on all datasets (seed=42)
python run_experiment/run_part1.py
python run_experiment/run_part23.py --dataset_key mnist
python run_experiment/run_part23.py --dataset_key sports_ball
python run_experiment/run_part23.py --dataset_key animals

# Part 2–3 only: single dataset
python run_experiment/run_part23.py --dataset_key mnist

# Part 2–3: label-noise sweep (e.g. p=0,0.1,0.2,0.3)
python run_experiment/run_part23.py --dataset_key mnist --label_noise_ps "0.0,0.1,0.2,0.3"
```

**Datasets (Part 2–3):** `mnist` (Keras), `sports_ball` / `animals` (Kaggle). Part 1 needs CIFAR-10 in `experiment_part1/data/`.
