# colab/run_pipeline.py
from pathlib import Path
import tensorflow as tf

from .datasets_config import DATASETS
from .data_prep import prepare_ds
from .cgan import (
    build_generator,
    build_discriminator,
    train_cgan,
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
    # mnist_keras는 kagglehub 안 씀
    if cfg["dataset_format"] == "mnist_keras":
        return prepare_ds(
            kaggle_dataset="",
            dataset_format=cfg["dataset_format"],
            **kwargs,
        )

    return prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        train_subdir=cfg.get("train_subdir", "train"),
        val_subdir=cfg.get("val_subdir", None),
        test_subdir=cfg.get("test_subdir", "test"),
        **kwargs,
    )


def _build_gan_ds_from_train(train_ds, *, img_size=(64, 64)):
    """train_ds: (image,label) batches. Convert to [-1,1] RGB and resize to GAN size."""
    def to_gan(x, y):
        x = tf.image.resize(x, img_size)
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = tf.cast(x, tf.float32)
        x = (x / 127.5) - 1.0
        y = tf.cast(y, tf.int32)
        return x, y

    return train_ds.map(to_gan, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def _pick_epoch_dir(out_dir: Path, cgan_epochs: int, sample_every: int):
    """epoch_{cgan_epochs} 없으면 epoch_{sample_every}로 fallback."""
    d1 = out_dir / f"epoch_{cgan_epochs:03d}"
    if d1.exists():
        return d1
    d2 = out_dir / f"epoch_{sample_every:03d}"
    return d2


def run_pipeline_one(
    dataset_key,
    out_root="results",
    *,
    # cGAN
    cgan_epochs=CGAN_EPOCHS,
    sample_every=SAMPLE_EVERY,
    label_noise_p=0.2,
    reuse_if_exists=True,     # ✅ 이미 생성된 fake가 있으면 학습 스킵
    # CNN
    cnn_ratios=None,
    cnn_epochs_stage1=EPOCHS_STAGE1,
    cnn_epochs_stage2=EPOCHS_STAGE2,
    # Eval
    run_fid_kid=True,
    verbose=1,
):
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset_key: {dataset_key}. Choose from {list(DATASETS)}")

    cfg = DATASETS[dataset_key]
    name = cfg["name"]
    dataset_format = cfg["dataset_format"]

    if cnn_ratios is None:
        cnn_ratios = [1.0, 0.8, 0.7, 0.6, 0.5]

    out_base = Path(out_root) / dataset_key
    out_base.mkdir(parents=True, exist_ok=True)

    out_fake_base = out_base / "fake_samples" / "baseline"
    out_fake_ln = out_base / "fake_samples" / f"label_noise_p{int(label_noise_p*100):02d}"

    out_csv_base = out_base / "cnn_ratio_results_baseline.csv"
    out_csv_ln = out_base / f"cnn_ratio_results_label_noise_p{int(label_noise_p*100):02d}.csv"

    if verbose:
        print("\n" + "=" * 70)
        print(f"Dataset: {name} ({dataset_key}) | format={dataset_format}")
        print("=" * 70)

    # ---------- 1) Data ----------
    if verbose:
        print("[1] Loading data via prepare_ds(...) ...")

    train_ds, val_ds, test_ds, class_names, data_dir, extras = _prepare_ds_for_dataset(
        cfg,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        val_split=0.2,
    )

    num_classes = len(class_names)
    if verbose:
        print(f"    data_dir: {data_dir}")
        print(f"    num_classes: {num_classes}")
        print(f"    class_names[:10]: {class_names[:10]}")
        if "inference_ds" in extras:
            print("    NOTE: unlabeled TEST detected; metrics test_ds=val_ds fallback.")

    # Ensure RGB for classifier (MNIST etc.)
    def ensure_rgb(x, y):
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        return x, y

    train_ds = train_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # GAN ds
    gan_ds = _build_gan_ds_from_train(train_ds.unbatch().batch(GAN_BATCH), img_size=GAN_IMG)

    # ---------- 2) cGAN baseline ----------
    if verbose:
        print("\n[2] Training cGAN baseline (label_noise_p=0.0) ...")

    out_fake_base.mkdir(parents=True, exist_ok=True)
    fake_base_dir = _pick_epoch_dir(out_fake_base, cgan_epochs, sample_every)

    if reuse_if_exists and fake_base_dir.exists():
        if verbose:
            print(f"    ✅ reuse baseline fake dir: {fake_base_dir}")
    else:
        G0 = build_generator(LATENT_DIM, num_classes, img_size=GAN_IMG, base_ch=BASE_CH)
        D0 = build_discriminator(num_classes, img_size=GAN_IMG, base_ch=BASE_CH)

        train_cgan(
            gan_ds,
            G0,
            D0,
            class_names,
            LATENT_DIM,
            num_classes,
            label_noise_p=0.0,
            epochs=cgan_epochs,
            sample_every=sample_every,
            out_dir=str(out_fake_base),
        )
        fake_base_dir = _pick_epoch_dir(out_fake_base, cgan_epochs, sample_every)

    if verbose:
        print(f"    baseline fake dir: {fake_base_dir}")

    # ---------- 3) cGAN label-noise ----------
    if verbose:
        print(f"\n[3] Training cGAN label-noise (label_noise_p={label_noise_p}) ...")

    out_fake_ln.mkdir(parents=True, exist_ok=True)
    fake_ln_dir = _pick_epoch_dir(out_fake_ln, cgan_epochs, sample_every)

    if reuse_if_exists and fake_ln_dir.exists():
        if verbose:
            print(f"    ✅ reuse label-noise fake dir: {fake_ln_dir}")
    else:
        G1 = build_generator(LATENT_DIM, num_classes, img_size=GAN_IMG, base_ch=BASE_CH)
        D1 = build_discriminator(num_classes, img_size=GAN_IMG, base_ch=BASE_CH)

        train_cgan(
            gan_ds,
            G1,
            D1,
            class_names,
            LATENT_DIM,
            num_classes,
            label_noise_p=float(label_noise_p),
            epochs=cgan_epochs,
            sample_every=sample_every,
            out_dir=str(out_fake_ln),
        )
        fake_ln_dir = _pick_epoch_dir(out_fake_ln, cgan_epochs, sample_every)

    if verbose:
        print(f"    label-noise fake dir: {fake_ln_dir}")

    results = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "dataset_format": dataset_format,
        "has_inference_ds": "inference_ds" in extras,
        "fake_epoch_dir_baseline": str(fake_base_dir),
        "fake_epoch_dir_label_noise": str(fake_ln_dir),
    }

    # ---------- 4) CNN ratio experiments ----------
    if verbose:
        print("\n[4] CNN performance with baseline fake:")
        print(f"    ratios: {cnn_ratios}")

    cnn_results_base = run_ratio_experiments(
        train_ds, val_ds, test_ds,
        class_names,
        fake_base_dir,
        ratios=cnn_ratios,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs_stage1=cnn_epochs_stage1,
        epochs_stage2=cnn_epochs_stage2,
        out_csv=str(out_csv_base),
        verbose=verbose,
    )
    results["cnn_results_baseline"] = cnn_results_base

    if verbose:
        print("\n[4b] CNN performance with label-noise fake:")
        print(f"    ratios: {cnn_ratios}")

    cnn_results_ln = run_ratio_experiments(
        train_ds, val_ds, test_ds,
        class_names,
        fake_ln_dir,
        ratios=cnn_ratios,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs_stage1=cnn_epochs_stage1,
        epochs_stage2=cnn_epochs_stage2,
        out_csv=str(out_csv_ln),
        verbose=verbose,
    )
    results["cnn_results_label_noise"] = cnn_results_ln

    # ---------- 5) FID/KID ----------
    if run_fid_kid:
        if verbose:
            print("\n[5] FID/KID (real test vs fake folders) ...")

        real_images_ds = real_ds_to_images_only(test_ds)

        fid0, kidm0, kids0 = eval_folder_vs_real(
            real_images_ds,
            fake_base_dir,
            max_samples=FID_MAX_SAMPLES,
            batch_size=FID_BATCH,
            image_size=CLS_IMG,
            seed=42,
        )
        results["fid_baseline"] = fid0
        results["kid_baseline_mean"] = kidm0
        results["kid_baseline_std"] = kids0

        fid1, kidm1, kids1 = eval_folder_vs_real(
            real_images_ds,
            fake_ln_dir,
            max_samples=FID_MAX_SAMPLES,
            batch_size=FID_BATCH,
            image_size=CLS_IMG,
            seed=42,
        )
        results["fid_label_noise"] = fid1
        results["kid_label_noise_mean"] = kidm1
        results["kid_label_noise_std"] = kids1

        if verbose:
            print(f"    Baseline   FID: {fid0:.2f} | KID: {kidm0:.4f} ± {kids0:.4f}")
            print(f"    LabelNoise FID: {fid1:.2f} | KID: {kidm1:.4f} ± {kids1:.4f}")

    return results
