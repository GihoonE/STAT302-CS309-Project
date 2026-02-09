"""
FID/KID evaluation for Colab.
InceptionV3 features; compare real vs fake (or noisy) image folders/datasets.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import linalg


def make_inception_extractor():
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(299, 299, 3),
    )
    base.trainable = False
    return base


def preprocess_for_inception(x):
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, (299, 299))
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x


def collect_activations(ds, extractor, max_samples=5000):
    feats = []
    seen = 0
    for batch in ds:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = preprocess_for_inception(images)
        f = extractor(images, training=False).numpy()
        feats.append(f)
        seen += f.shape[0]
        if seen >= max_samples:
            break
    feats = np.concatenate(feats, axis=0)[:max_samples]
    return feats


def compute_stats(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1.0):
    d = x.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    return (gamma * x @ y.T + coef0) ** degree


def mmd2_unbiased_polynomial(x, y, degree=3, gamma=None, coef0=1.0):
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("Need at least 2 samples per set for unbiased KID.")

    Kxx = polynomial_kernel(x, x, degree, gamma, coef0)
    Kyy = polynomial_kernel(y, y, degree, gamma, coef0)
    Kxy = polynomial_kernel(x, y, degree, gamma, coef0)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    return float(
        Kxx.sum() / (m * (m - 1))
        + Kyy.sum() / (n * (n - 1))
        - 2.0 * Kxy.mean()
    )


def kernel_inception_distance(real_feats, fake_feats, subset_size=1000, n_subsets=50, seed=42):
    rng = np.random.default_rng(seed)
    N = min(len(real_feats), len(fake_feats))
    if N < 2:
        raise ValueError("Not enough samples to compute KID.")

    subset_size = min(subset_size, N)
    scores = []
    for _ in range(n_subsets):
        ridx = rng.choice(len(real_feats), subset_size, replace=False)
        fidx = rng.choice(len(fake_feats), subset_size, replace=False)
        scores.append(mmd2_unbiased_polynomial(real_feats[ridx], fake_feats[fidx]))

    return float(np.mean(scores)), float(np.std(scores))


def compute_fid_kid_from_datasets(
    real_ds,
    fake_ds,
    *,
    max_samples=5000,
    kid_subset_size=1000,
    kid_n_subsets=50,
    seed=42,
):
    """Compute FID and KID(mean, std) from two tf.data.Datasets (image batches)."""
    extractor = make_inception_extractor()

    real_feats = collect_activations(real_ds, extractor, max_samples=max_samples)
    fake_feats = collect_activations(fake_ds, extractor, max_samples=max_samples)

    mu_r, sig_r = compute_stats(real_feats)
    mu_f, sig_f = compute_stats(fake_feats)
    fid = frechet_distance(mu_r, sig_r, mu_f, sig_f)

    kid_mean, kid_std = kernel_inception_distance(
        real_feats,
        fake_feats,
        subset_size=kid_subset_size,
        n_subsets=kid_n_subsets,
        seed=seed,
    )

    return fid, kid_mean, kid_std


def folder_to_ds(folder, *, image_size=(224, 224), batch_size=32, shuffle=False):
    """Load folder (with optional class subdirs) as dataset; labels not used (for FID/KID)."""
    folder = str(folder)
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode=None,
    ).prefetch(tf.data.AUTOTUNE)


def real_ds_to_images_only(ds):
    """Convert (images, labels) dataset to images-only."""
    return ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def eval_folder_vs_real(
    real_images_ds,
    fake_folder,
    *,
    max_samples=2000,
    kid_subset_size=500,
    kid_n_subsets=50,
    batch_size=32,
    image_size=(224, 224),
    seed=42,
):
    """
    Compare one fake folder vs real images dataset.
    Returns (fid, kid_mean, kid_std).
    """
    fake_ds = folder_to_ds(
        fake_folder,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return compute_fid_kid_from_datasets(
        real_images_ds,
        fake_ds,
        max_samples=max_samples,
        kid_subset_size=kid_subset_size,
        kid_n_subsets=kid_n_subsets,
        seed=seed,
    )
