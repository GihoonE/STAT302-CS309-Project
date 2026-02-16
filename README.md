# Conditional GAN for Synthetic Data: Evaluation Framework

A modular pipeline for training and evaluating conditional GANs (cGANs) across multiple datasets. It covers data preparation, cGAN training (with optional label noise), downstream classification with mixed real/synthetic data, and quality metrics (FID/KID).

---

## Directory structure

```
STAT302-CS309-Project/
├── colab/                    # Main library and pipeline
│   ├── __init__.py
│   ├── datasets_config.py    # Dataset keys, slugs, formats
│   ├── data_prep.py          # prepare_ds(): train/val/test loaders
│   ├── cgan.py               # Generator, Discriminator, train_cgan, label noise
│   ├── classifier.py        # EfficientNetB0 + run_ratio_experiments (mixing)
│   ├── fid_kid.py            # FID/KID evaluation (Inception V3)
│   ├── run_pipeline.py       # run_pipeline_one(): full pipeline entrypoint
│   ├── test_multi_datasets.py
│   └── README.md             # Colab-focused usage details
├── tools/
│   ├── run_pipeline.py       # CLI: single run or label-noise sweep
│   └── test_data_load.py    # Sanity check dataset download and loading
├── results/                  # Per-dataset outputs (fake_samples, CSVs, metrics)
│   └── {dataset_key}/
│       ├── fake_samples/baseline/epoch_*/
│       ├── fake_samples/label_noise_p{XX}/epoch_*
│       ├── cnn_ratio_results_baseline.csv
│       ├── cnn_ratio_results_label_noise_p{XX}.csv
│       ├── metrics/metrics_label_noise_p{XX}.json
│       └── sweeps/label_noise_sweep.csv
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

---

## Training pipeline overview

The pipeline has two main sides:

- **Model-side:** cGAN architecture and training (conditioning and loss options; see Ablation below).
- **Data-side:** Dataset formats, label noise during cGAN training, and downstream mixing ratios (original vs synthetic).

End-to-end flow:

1. **Data** — `prepare_ds()` loads train/val/test (and optional inference set) per `datasets_config.py`.
2. **cGAN training** — Train two cGANs: baseline (no label noise) and label-noise (configurable `label_noise_p`). Samples are saved every `sample_every` epochs under `fake_samples/`.
3. **Downstream mixing** — For each fake folder (baseline and label-noise), run CNN ratio experiments: train classifier on mixes of original + synthetic at several ratios (e.g. 100% orig, 80/20, 70/30, …, 50/50). Results are written to CSV.
4. **Evaluation** — FID/KID between real test images and fake folders; test accuracy at chosen mixing ratio(s).

---

## (1) Model-side experiments

- **Architecture:** Conditional Generator and Discriminator in `colab/cgan.py`.
  - **Conditioning:** Label conditioning is implemented as **concatenation**: embedding → flatten → concat with noise (G) or spatial map concat with image (D).
  - **Loss:** **BCE** (Binary Cross-Entropy) with label smoothing (0.9) on real logits.
- **Training:** Adam (lr=2e-4, β₁=0.5), 64×64 RGB images in [-1,1]. Optional **label noise** (`label_noise_p`): with probability `p`, the label is replaced by a random class during cGAN training.

### Ablation: concat vs projection, BCE vs hinge

The current codebase uses **concat** conditioning and **BCE** only. To reproduce or extend ablations:

- **Concat vs projection**
  - **Current (concat):** In `cgan.py`, Generator uses `layers.Concatenate()([z_in, y_emb])`; Discriminator uses `layers.Concatenate()([x_in, y_emb])` with `y_emb` reshaped to spatial map.
  - **Projection variant:** Replace concat in G by projecting `y_emb` to a small dimension and adding (e.g. linear layer then `+`) to an intermediate feature map; in D, project label embedding and add to the flattened feature vector before the final Dense(1). Implement in `build_generator()` and `build_discriminator()` and optionally make the mode selectable via an argument.
- **BCE vs hinge**
  - **Current (BCE):** `d_loss_fn` and `g_loss_fn` in `cgan.py` use `tf.keras.losses.BinaryCrossEntropy(from_logits=True)`.
  - **Hinge variant:** Replace D loss with `real_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits))`, `fake_loss = -tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))`; G loss with `-tf.reduce_mean(fake_logits)`. Add a flag (e.g. `loss_type="bce"|"hinge"`) and branch in `train_cgan()` (or in loss helpers) so you can run the same pipeline with either loss.

Example (conceptual) for a single run with hinge and projection would require calling your modified `build_*` and `train_cgan(..., loss_type="hinge")`; the rest of the pipeline (data, sampling, CNN, FID/KID) stays the same.

---

## (2) Data-side experiments

- **Dataset formats** (`colab/data_prep.py`): `imagefolder_supervised_train_test`, `imagefolder_train_labeled_test_unlabeled`, `mnist_keras`, `mnist_idx`, `mnist_csv`. Configured in `colab/datasets_config.py` (e.g. `sports_ball`, `animals`, `mnist`).
- **Label-noise experiments:** During cGAN training, labels are corrupted with probability `label_noise_p`. To run a **single** setting use `run_pipeline_one(..., label_noise_p=0.2)`. To run a **sweep** over multiple values (e.g. 0.0, 0.1, 0.2, 0.3), use the CLI (see below).
- **Downstream mixing:** After generating fakes, the pipeline runs **ratio experiments**: train an EfficientNetB0 classifier on mixed data (original + synthetic) at ratios such as 1.0, 0.8, 0.7, 0.6, 0.5 (configurable). Results are saved per fake source (baseline vs label-noise) in CSVs.

---

## (3) Evaluation metrics

- **FID (Fréchet Inception Distance):** Inception V3 features; Fréchet distance between real and fake distributions. Lower is better.
- **KID (Kernel Inception Distance):** Polynomial kernel MMD (mean ± std over subsets). Lower is better.
- **Test accuracy:** For each mixing ratio, the classifier is trained and evaluated on the test set; accuracy is reported and (optionally) one ratio is picked for sweep CSV (e.g. 80% original / 20% synthetic).

FID/KID are computed in `colab/fid_kid.py`; test accuracy comes from `colab/classifier.py` and is aggregated in `tools/run_pipeline.py` for the sweep CSV.

---

## Example commands

**Data loading sanity check (from repo root):**

```bash
python tools/test_data_load.py --dataset_key mnist
python tools/test_data_load.py --dataset_key sports_ball --val_split 0.2
```

**Full pipeline for one dataset (baseline + one label-noise level, with FID/KID and CNN ratio experiments):**

```bash
python tools/run_pipeline.py --dataset_key mnist --out_root results
```

**Single dataset, custom label noise and ratios:**

```bash
python tools/run_pipeline.py --dataset_key sports_ball --out_root results \
  --label_noise_p 0.2 \
  --cnn_ratios "1.0,0.9,0.8,0.7,0.6,0.5" \
  --cgan_epochs 30
```

**Label-noise sweep (multiple p values):**

```bash
python tools/run_pipeline.py --dataset_key mnist --out_root results \
  --label_noise_ps "0.0,0.1,0.2,0.3" \
  --acc_ratio 0.8
```

Output: `results/<dataset_key>/sweeps/label_noise_sweep.csv` with columns such as `label_noise_p`, `acc_baseline_fake`, `acc_label_noise_fake`, `fid_baseline`, `fid_label_noise`, `kid_*`, and paths to CSVs and fake dirs.

**Python API (one full pipeline run):**

```python
from colab.run_pipeline import run_pipeline_one

result = run_pipeline_one(
    "mnist",  # or "sports_ball", "animals"
    out_root="results",
    label_noise_p=0.2,
    cnn_ratios=[1.0, 0.8, 0.7, 0.6, 0.5],
    run_fid_kid=True,
    reuse_if_exists=True,
    verbose=1,
)
# result contains cnn_results_baseline, cnn_results_label_noise, fid_*, kid_*, paths, etc.
```

**Downstream mixing only (if you already have a fake folder):**

```python
from colab.classifier import run_ratio_experiments
from colab.data_prep import prepare_ds
from colab.datasets_config import DATASETS

cfg = DATASETS["mnist"]
train_ds, val_ds, test_ds, class_names, data_dir, _ = prepare_ds(
    cfg.get("slug", ""), dataset_format=cfg["dataset_format"], val_split=0.2
)
results = run_ratio_experiments(
    train_ds, val_ds, test_ds, class_names,
    fake_folder="results/mnist/fake_samples/baseline/epoch_030",
    ratios=[1.0, 0.8, 0.6, 0.5],
    out_csv="results/mnist/cnn_ratio_results_baseline.csv",
)
```

**FID/KID only:**

```python
from colab.fid_kid import eval_folder_vs_real, real_ds_to_images_only

real_images_ds = real_ds_to_images_only(test_ds)
fid, kid_mean, kid_std = eval_folder_vs_real(
    real_images_ds,
    "results/mnist/fake_samples/baseline/epoch_030",
    max_samples=2000,
)
```

---

## Reproducing the ablation study (concat vs projection, BCE vs hinge)

1. **Baseline (current):** Run the pipeline as-is; conditioning is concat and loss is BCE.
2. **Projection:** In `colab/cgan.py`, add a projection-based conditioning path in `build_generator()` and `build_discriminator()` (see Model-side section), and optionally a keyword argument to switch between concat and projection. Run the same pipeline with the projection option.
3. **Hinge:** In `colab/cgan.py`, implement hinge loss in `d_loss_fn`/`g_loss_fn` and a `loss_type` (or global) switch. Call `train_cgan(..., loss_type="hinge")` and run the pipeline again.
4. **Combinations:** Run (concat, BCE), (concat, hinge), (projection, BCE), (projection, hinge); compare FID/KID and downstream accuracy at the same mixing ratios and label-noise settings.

All other steps (data, sampling, CNN ratios, FID/KID) remain unchanged; only the cGAN build and loss in `cgan.py` are modified.

---

## Output files reference

| Path pattern | Description |
|--------------|-------------|
| `results/<key>/fake_samples/baseline/epoch_030/` | Generated images per class (baseline cGAN) |
| `results/<key>/fake_samples/label_noise_p20/epoch_030/` | Generated images (cGAN trained with 20% label noise) |
| `results/<key>/cnn_ratio_results_baseline.csv` | orig_ratio, cgan_ratio, test_accuracy (baseline fake) |
| `results/<key>/cnn_ratio_results_label_noise_p20.csv` | Same for label-noise fake at p=0.2 |
| `results/<key>/metrics/metrics_label_noise_p20.json` | fid_baseline, kid_baseline_*, fid_label_noise, kid_label_noise_* |
| `results/<key>/sweeps/label_noise_sweep.csv` | Sweep over label_noise_p with acc and FID/KID columns |

---

## Datasets (from `colab/datasets_config.py`)

| Key | Name | Format | Notes |
|-----|------|--------|--------|
| `sports_ball` | Sports Ball | imagefolder_supervised_train_test | Kaggle slug: samuelcortinhas/... |
| `animals` | Cats-and-Dogs-Breed | imagefolder_train_labeled_test_unlabeled | train/val/inf structure |
| `mnist` | MNIST | mnist_keras | Keras built-in; no Kaggle |

For more details, Colab-specific usage, and module reference, see `colab/README.md`.
