"""
Full experiment pipeline per dataset:
1. Data load (Kaggle) -> train/val/test
2. cGAN train -> save fake samples
3. Show 3~5 cGAN-generated images
4. CNN performance: 100% orig, 90%+10% cGAN, ..., 50%+50% cGAN
5. FID/KID: real vs cGAN fake; optional real vs noisy

Usage (Colab):
  from colab.run_pipeline import run_pipeline_one, run_all_datasets
  run_pipeline_one("sports_ball", out_root="/content/results")
  run_all_datasets(out_root="/content/results")
"""
from pathlib import Path

import tensorflow as tf

from .datasets_config import DATASETS
from .data_prep import prepare_ds
from .cgan import (
    build_generator,
    build_discriminator,
    train_cgan,
    make_noisy_clone,
)
from .fid_kid import eval_folder_vs_real, real_ds_to_images_only
from .classifier import run_ratio_experiments

# GAN defaults
GAN_IMG = (64, 64)
GAN_BATCH = 64
LATENT_DIM = 128
BASE_CH = 64
CGAN_EPOCHS = 30
SAMPLE_EVERY = 5

# Classifier defaults
CLS_IMG = (224, 224)
CLS_BATCH = 32
STEPS_PER_EPOCH = 200
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10

# FID/KID
FID_MAX_SAMPLES = 2000
FID_BATCH = 32


def _prepare_ds_for_dataset(cfg, **kwargs):
    """Explicitly route dataset loading by configured dataset_format and subdirs."""
    return prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        train_subdir=cfg.get("train_subdir", "train"),
        test_subdir=cfg.get("test_subdir", "test"),
        **kwargs,
    )


def _build_gan_ds_from_train(train_ds, *, img_size=(64, 64)):
    """Create cGAN dataset from (image,label) training dataset regardless of source layout."""
    def to_gan(x, y):
        x = tf.image.resize(x, img_size)
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = tf.cast(x, tf.float32)
        x = (x / 127.5) - 1.0
        y = tf.cast(y, tf.int32)
        return x, y
    return train_ds.map(to_gan, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def show_cgan_samples(fake_epoch_dir, n_show=5, figsize=(12, 3)):
    """
    Load and display n_show images from a cGAN epoch folder (class subdirs).
    Picks one image per class until n_show total. Returns the figure (call plt.show() in Colab).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip display.")
        return None

    fake_epoch_dir = Path(fake_epoch_dir)
    if not fake_epoch_dir.exists():
        print(f"Folder not found: {fake_epoch_dir}")
        return None

    class_dirs = sorted([p for p in fake_epoch_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        print(f"No class subdirs in {fake_epoch_dir}")
        return None

    images, labels = [], []
    for cdir in class_dirs:
        files = list(cdir.glob("*.png")) + list(cdir.glob("*.jpg"))
        if not files:
            continue
        # load first image
        img = tf.io.read_file(str(files[0]))
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        images.append(img.numpy())
        labels.append(cdir.name)
        if len(images) >= n_show:
            break

    if not images:
        print("No images found.")
        return None

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (im, label) in enumerate(zip(images, labels)):
        axes[i].imshow(im)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    return fig


def run_pipeline_one(
    dataset_key,
    out_root="/content/results",
    *,
    cgan_epochs=CGAN_EPOCHS,
    sample_every=SAMPLE_EVERY,
    n_show_samples=5,
    cnn_ratios=None,
    cnn_epochs_stage1=EPOCHS_STAGE1,
    cnn_epochs_stage2=EPOCHS_STAGE2,
    run_noisy_clone=False,
    run_fid_kid=True,
    verbose=1,
):
    """
    Run full pipeline for one dataset.

    - dataset_key: "sports_ball" | "mnist" | "animals"
    - out_root: base path for outputs (fake_samples, csv, etc.)
    - n_show_samples: number of cGAN samples to display (3~5)
    - run_noisy_clone: whether to create Gaussian/JPEG noisy copies and report FID/KID for them
    - run_fid_kid: whether to compute FID/KID (real vs cGAN fake)

    Returns dict with keys: data_dir, class_names, fake_epoch_dir, cnn_results, fid, kid_mean, kid_std, (noisy FID/KID if run_noisy_clone).
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset_key: {dataset_key}. Choose from {list(DATASETS)}")

    cfg = DATASETS[dataset_key]
    name = cfg["name"]
    dataset_format = cfg["dataset_format"]

    out_base = Path(out_root) / dataset_key
    out_base.mkdir(parents=True, exist_ok=True)
    out_fake = out_base / "fake_samples" / "baseline"
    out_noisy = out_base / "noisy"
    out_csv = out_base / "cnn_ratio_results.csv"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {name} ({dataset_key})")
        print(f"{'='*60}\n")

    # ---------- 1) Data ----------
    if verbose:
        print("[1] Loading data...")
    train_ds, val_ds, test_ds, class_names, data_dir, extras = _prepare_ds_for_dataset(
        cfg,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        val_split=0.2,
    )
    num_classes = len(class_names)
    train_dir = str(Path(data_dir) / cfg.get("train_subdir", "train"))
    test_dir = str(Path(data_dir) / cfg.get("test_subdir", "test"))

    # Ensure RGB for classifier (MNIST etc. may be grayscale)
    def ensure_rgb(x, y):
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        return x, y
    train_ds = train_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    if verbose:
        print(f"    Classes: {num_classes}")

    # ---------- 2) cGAN ----------
    if verbose:
        print("\n[2] Training cGAN...")
    gan_ds = _build_gan_ds_from_train(train_ds.unbatch().batch(GAN_BATCH), img_size=GAN_IMG)
    G = build_generator(LATENT_DIM, num_classes, img_size=GAN_IMG, base_ch=BASE_CH)
    D = build_discriminator(num_classes, img_size=GAN_IMG, base_ch=BASE_CH)
    train_cgan(
        gan_ds,
        G,
        D,
        class_names,
        LATENT_DIM,
        num_classes,
        label_noise_p=0.0,
        epochs=cgan_epochs,
        sample_every=sample_every,
        out_dir=str(out_fake),
    )
    fake_epoch_dir = out_fake / f"epoch_{cgan_epochs:03d}"
    if not fake_epoch_dir.exists():
        fake_epoch_dir = out_fake / f"epoch_{sample_every:03d}"  # fallback to first sample epoch
    if verbose:
        print(f"    Fake samples: {fake_epoch_dir}")

    # ---------- 3) Show 3~5 cGAN samples ----------
    if verbose:
        print("\n[3] cGAN samples (3~5 images):")
    fig = show_cgan_samples(fake_epoch_dir, n_show=min(n_show_samples, 5))
    if fig is not None:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    results = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "fake_epoch_dir": str(fake_epoch_dir),
        "dataset_format": dataset_format,
        "has_inference_ds": "inference_ds" in extras,
    }

    # ---------- 4) CNN ratio experiments ----------
    if verbose:
        print("\n[4] CNN performance (100% orig -> 50% orig + 50% cGAN):")
    cnn_results = run_ratio_experiments(
        train_ds,
        val_ds,
        test_ds,
        class_names,
        fake_epoch_dir,
        ratios=cnn_ratios,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs_stage1=cnn_epochs_stage1,
        epochs_stage2=cnn_epochs_stage2,
        out_csv=str(out_csv),
        verbose=verbose,
    )
    results["cnn_results"] = cnn_results
    if verbose:
        for r, acc in cnn_results:
            print(f"    {int(r*100)}% orig + {int((1-r)*100)}% cGAN -> test_acc = {acc:.4f}")

    # ---------- 5) FID/KID: real vs cGAN ----------
    fid, kid_mean, kid_std = None, None, None
    if run_fid_kid and fake_epoch_dir.exists():
        if verbose:
            print("\n[5] FID/KID (real test vs cGAN fake):")
        real_images_ds = real_ds_to_images_only(test_ds)
        fid, kid_mean, kid_std = eval_folder_vs_real(
            real_images_ds,
            fake_epoch_dir,
            max_samples=FID_MAX_SAMPLES,
            batch_size=FID_BATCH,
            image_size=CLS_IMG,
            seed=42,
        )
        results["fid"] = fid
        results["kid_mean"] = kid_mean
        results["kid_std"] = kid_std
        if verbose:
            print(f"    FID: {fid:.2f}")
            print(f"    KID: {kid_mean:.4f} ± {kid_std:.4f}")

    # ---------- 6) Optional: noisy clone + FID/KID ----------
    if run_noisy_clone:
        if dataset_format == "mnist_idx":
            if verbose:
                print("\n[6] Skip noisy clone: MNIST IDX has no class-folder split to clone directly.")
        else:
            if verbose:
                print("\n[6] Noisy clone (Gaussian train, JPEG test) + FID/KID vs real:")
            make_noisy_clone(
                train_dir,
                str(out_noisy / "train_gaussian_s0.1"),
                noise="gaussian",
                sigma=0.10,
            )
            make_noisy_clone(
                test_dir,
                str(out_noisy / "test_jpeg_q35"),
                noise="jpeg",
                jpeg_quality=35,
            )
            real_images_ds = real_ds_to_images_only(test_ds)
            noisy_test_dir = out_noisy / "test_jpeg_q35"
            if noisy_test_dir.exists():
                fid_n, kid_m_n, kid_s_n = eval_folder_vs_real(
                    real_images_ds,
                    noisy_test_dir,
                    max_samples=FID_MAX_SAMPLES,
                    batch_size=FID_BATCH,
                    image_size=CLS_IMG,
                    seed=42,
                )
                results["noisy_fid"] = fid_n
                results["noisy_kid_mean"] = kid_m_n
                results["noisy_kid_std"] = kid_s_n
                if verbose:
                    print(f"    Noisy (JPEG) vs Real - FID: {fid_n:.2f}, KID: {kid_m_n:.4f} ± {kid_s_n:.4f}")

    return results


def run_all_datasets(
    out_root="/content/results",
    dataset_keys=None,
    **kwargs,
):
    """
    Run run_pipeline_one for each dataset. dataset_keys defaults to all three.
    kwargs passed to run_pipeline_one (e.g. cgan_epochs, n_show_samples).
    """
    if dataset_keys is None:
        dataset_keys = list(DATASETS.keys())
    all_results = {}
    for key in dataset_keys:
        all_results[key] = run_pipeline_one(key, out_root=out_root, **kwargs)
    return all_results
