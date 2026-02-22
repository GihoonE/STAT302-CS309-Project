# Conditional GAN for Synthetic Data Generation: A Comprehensive Evaluation Framework

**Experiment Part 2–3.** This folder contains the multi-dataset pipeline (TensorFlow). For the full repository layout (Part 1 = CIFAR-10 ablation, Part 2–3 = this folder), see the [main README](../README.md). Package name: `experiment_part23`.

## Abstract

We present a modular framework for training and evaluating conditional Generative Adversarial Networks (cGANs) for synthetic image generation across multiple datasets. Our pipeline integrates data preparation, cGAN training, classification performance evaluation, and quantitative assessment using Fréchet Inception Distance (FID) and Kernel Inception Distance (KID). The framework supports automated experiments comparing classification accuracy across different ratios of original and synthetic data (100% original to 50% original + 50% synthetic), enabling systematic analysis of synthetic data utility. We demonstrate the framework on three diverse datasets: Sports Ball classification (15 classes), MNIST digit recognition (10 classes), and Animal classification, providing a standardized evaluation protocol for conditional generative models.

## 1. Introduction

Synthetic data generation using Generative Adversarial Networks (GANs) has shown promise in augmenting training datasets for classification tasks. However, evaluating the quality and utility of generated samples remains challenging. This work provides a comprehensive, reproducible framework that:

- **Automates the full pipeline** from data loading to evaluation across multiple datasets
- **Quantifies synthetic data quality** using FID and KID metrics
- **Measures classification utility** by training classifiers on mixed datasets with varying ratios of original and synthetic samples
- **Supports multiple datasets** through a unified interface

The framework is designed for Google Colab, leveraging GPU acceleration and Kaggle dataset integration for seamless experimentation.

## 2. Method

### 2.1 Architecture

Our framework consists of four main components:

**Data Preparation (`data_prep.py`):** Downloads datasets from Kaggle using `kagglehub` and constructs TensorFlow datasets with train/validation/test splits. Supports automatic handling of different folder structures (e.g., `train/` vs `Train/`).

**cGAN Module (`cgan.py`):** Implements a conditional GAN with:
- **Generator:** Convolutional architecture using transposed convolutions, conditioned on class labels via embedding layers. Outputs 64×64 RGB images normalized to [-1, 1].
- **Discriminator:** Convolutional network with label conditioning through spatial label maps. Uses label smoothing (0.9) and dropout (0.3) for stability.
- **Training:** Adam optimizer (lr=2e-4, β₁=0.5) with binary cross-entropy loss. Supports optional label noise injection for robustness experiments.

**Evaluation (`fid_kid.py`):** Computes FID and KID metrics using Inception V3 features:
- Extracts 2048-dimensional features from preprocessed images (299×299)
- FID: Fréchet distance between real and fake feature distributions
- KID: Unbiased polynomial kernel MMD with 50 random subsets

**Classification (`classifier.py`):** EfficientNetB0-based classifier with:
- Stage 1: Frozen ImageNet-pretrained backbone, trainable classification head (15 epochs)
- Stage 2: Fine-tuning of last 40 layers (10 epochs)
- Data augmentation: Random horizontal flip, rotation (±5%), zoom (±10%)

### 2.2 Experimental Protocol

For each dataset, we run:

1. **cGAN Training:** Train conditional GAN for 30 epochs, saving samples every 5 epochs
2. **Sample Visualization:** Display 3-5 generated samples per class
3. **Classification Experiments:** Train classifiers on mixed datasets with ratios:
   - 100% original
   - 90% original + 10% synthetic
   - 80% original + 20% synthetic
   - 70% original + 30% synthetic
   - 60% original + 40% synthetic
   - 50% original + 50% synthetic
4. **Quality Assessment:** Compute FID/KID between real test set and generated samples
5. **Baseline Comparison (optional):** Generate noisy versions of real images (Gaussian noise on train, JPEG compression on test) and compute FID/KID as a quality reference

## 3. Installation

### Requirements

```python
!pip install -q kagglehub tensorflow scipy matplotlib
```

### Setup

**Option A: Clone repository**
```python
!git clone https://github.com/GihoonE/STAT302-CS309-Project.git
import sys
sys.path.insert(0, "/content/STAT302-CS309-Project")
```

**Option B: Upload `experiment_part23` folder**
```python
import sys
sys.path.insert(0, "/content/...")  # path to parent of experiment_part23 folder
```

## 4. Usage

### 4.1 Automated Pipeline

Run experiments on all three datasets:

```python
from experiment_part23 import run_all_datasets  # if implemented

results = run_all_datasets(
    out_root="/content/results",
    n_show_samples=5,
    cgan_epochs=30,
    cnn_epochs_stage1=15,
    cnn_epochs_stage2=10,
    run_noisy_clone=False,  # Set True for noisy baseline
    verbose=1,
)
```

Run on a single dataset:

```python
from experiment_part23.run_pipeline import run_pipeline_one

result = run_pipeline_one(
    "sports_ball",  # or "mnist", "animals"
    out_root="/content/results",
    n_show_samples=5,
    run_noisy_clone=True,
)
```

### 4.2 Supported Datasets

| Dataset | Kaggle Slug | Classes | Description |
|---------|-------------|---------|-------------|
| Sports Ball | `samuelcortinhas/sports-balls-multiclass-image-classification` | 15 | Multi-class sports ball classification |
| MNIST | `arnavsharma45/mnist-dataset` | 10 | Handwritten digit recognition |
| Animals | `antobenedetti/animals` | Variable | Animal classification |

### 4.3 Output Structure

```
/content/results/
├── {dataset_key}/
│   ├── fake_samples/baseline/
│   │   └── epoch_030/{class_name}/*.png
│   ├── cnn_ratio_results.csv
│   └── noisy/ (if run_noisy_clone=True)
│       ├── train_gaussian_s0.1/
│       └── test_jpeg_q35/
```

**CSV Format (`cnn_ratio_results.csv`):**
```csv
orig_ratio,cgan_ratio,test_accuracy
1.00,0.00,0.9123
0.90,0.10,0.9087
...
```

### 4.4 Individual Components

**Data Loading:**
```python
from experiment_part23.data_prep import prepare_ds

train_ds, val_ds, test_ds, class_names, data_dir = prepare_ds(
    "samuelcortinhas/sports-balls-multiclass-image-classification",
    img_size=(224, 224),
    batch_size=32,
    val_split=0.2,
)
```

**cGAN Training:**
```python
from experiment_part23.cgan import build_gan_ds, build_generator, build_discriminator, train_cgan

gan_ds = build_gan_ds(train_dir, img_size=(64, 64), batch_size=64)
G = build_generator(latent_dim=128, num_classes=15, img_size=(64, 64))
D = build_discriminator(num_classes=15, img_size=(64, 64))
train_cgan(gan_ds, G, D, class_names, latent_dim=128, num_classes=15, epochs=30)
```

**FID/KID Evaluation:**
```python
from experiment_part23.fid_kid import eval_folder_vs_real, real_ds_to_images_only

real_images_ds = real_ds_to_images_only(test_ds)
fid, kid_mean, kid_std = eval_folder_vs_real(
    real_images_ds,
    "/content/results/sports_ball/fake_samples/baseline/epoch_030",
    max_samples=2000,
)
```

**Classification Experiments:**
```python
from experiment_part23.classifier import run_ratio_experiments

results = run_ratio_experiments(
    train_ds, val_ds, test_ds, class_names,
    fake_folder="/content/results/sports_ball/fake_samples/baseline/epoch_030",
    ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
)
```

## 5. Results

The pipeline outputs:

- **Generated samples:** Visual inspection of cGAN outputs (3-5 samples per class)
- **Classification accuracy:** Test accuracy for each original/synthetic ratio
- **FID score:** Lower is better (real vs fake comparison)
- **KID score:** Mean ± std across 50 random subsets (lower is better)

Example output:
```
[4] CNN performance (100% orig -> 50% orig + 50% cGAN):
    100% orig + 0% cGAN -> test_acc = 0.9123
    90% orig + 10% cGAN -> test_acc = 0.9087
    ...

[5] FID/KID (real test vs cGAN fake):
    FID: 310.52
    KID: 0.2156 ± 0.0031
```

## 6. Module Reference

| Module | Key Functions | Description |
|--------|---------------|-------------|
| `data_prep.py` | `prepare_ds()` | Kaggle dataset download and TensorFlow dataset construction |
| `cgan.py` | `build_generator()`, `build_discriminator()`, `train_cgan()`, `make_noisy_clone()` | cGAN architecture, training, and noisy image generation |
| `fid_kid.py` | `eval_folder_vs_real()`, `compute_fid_kid_from_datasets()` | FID/KID computation using Inception V3 |
| `classifier.py` | `build_effnet_classifier()`, `run_ratio_experiments()` | EfficientNetB0 classifier and ratio experiments |
| `datasets_config.py` | `DATASETS`, `KAGGLE_URLS` | Dataset configuration and Kaggle slugs |
| `run_pipeline.py` | `run_pipeline_one()`, `run_all_datasets()` | Automated end-to-end pipeline |

## 7. Configuration

Dataset-specific settings can be modified in `datasets_config.py`:

```python
DATASETS = {
    "sports_ball": {
        "slug": "samuelcortinhas/sports-balls-multiclass-image-classification",
        "train_subdir": "train",
        "test_subdir": "test",
    },
    # ...
}
```

If a dataset uses different folder names (e.g., `Train/` vs `train/`), the pipeline automatically attempts both variants.

## 8. Hardware Requirements

- **GPU:** Recommended (T4, V100, or A100 on Google Colab)
- **RAM:** Minimum 12GB (for EfficientNetB0 and Inception V3)
- **Storage:** ~2GB per dataset (including generated samples)

## 9. Citation

If you use this framework in your research, please cite:

```bibtex
@software{cgan_eval_framework,
  title = {Conditional GAN Evaluation Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/GihoonE/STAT302-CS309-Project}
}
```

## 10. License

[Specify your license]

---

**Note:** This framework is designed for research reproducibility. For production deployments, consider additional validation, error handling, and optimization.
