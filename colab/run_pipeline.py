# colab/run_pipeline.py
from pathlib import Path
import json
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

try:
    import pandas as pd
except Exception:
    pd = None

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

    # kagglehub 데이터셋들
    return prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        train_subdir=cfg.get("train_subdir", "train"),
        val_subdir=cfg.get("val_subdir", None),  # animals에 필요
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


def _resolve_epoch_dir(root_dir: Path, preferred_epoch: int):
    """
    Prefer epoch_{preferred_epoch:03d}.
    If not exist, pick the latest epoch_* dir.
    """
    p = root_dir / f"epoch_{preferred_epoch:03d}"
    if p.exists():
        return p
    cand = sorted([d for d in root_dir.glob("epoch_*") if d.is_dir()])
    return cand[-1] if cand else None


def _load_cnn_results_from_csv(csv_path: Path):
    """
    Return list[dict] with at least keys: ratio + test_acc/accuracy if possible.
    This matches tools/_pick_acc_at_ratio() expectations.
    """
    if pd is None or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def run_pipeline_one(
    dataset_key,
    out_root="results",
    *,
    # cGAN
    cgan_epochs=CGAN_EPOCHS,
    sample_every=SAMPLE_EVERY,
    label_noise_p=0.2,
    # CNN
    cnn_ratios=None,
    cnn_epochs_stage1=EPOCHS_STAGE1,
    cnn_epochs_stage2=EPOCHS_STAGE2,
    # Eval
    run_fid_kid=True,
    reuse_if_exists=True,   # ✅ NEW
    verbose=1,
):
    """
    Runs:
      - cGAN baseline + label-noise (optionally reuse)
      - CNN ratio experiments using baseline fake + label-noise fake (optionally reuse CSV)
      - FID/KID for baseline + label-noise (optionally reuse JSON)

    Returns dict with:
      cnn_results_baseline, cnn_results_label_noise
      fid/kid for baseline + label-noise
      fake dirs
    """
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

    # CNN csvs (baseline / label-noise 분리)
    out_csv_base = out_base / "cnn_ratio_results_baseline.csv"
    out_csv_ln = out_base / f"cnn_ratio_results_label_noise_p{int(label_noise_p*100):02d}.csv"

    # metrics cache
    metrics_dir = out_base / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_label_noise_p{int(label_noise_p*100):02d}.json"

    if verbose:
        print("\n" + "=" * 70)
        print(f"Dataset: {name} ({dataset_key}) | format={dataset_format}")
        print("=" * 70)

    # ---------- 1) Data ----------
    if verbose:
        print("[1] Loading data via prepare_ds ...")

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
            print("    NOTE: this dataset has unlabeled TEST; metrics test_ds=val_ds fallback.")

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
    fake_base_dir = _resolve_epoch_dir(out_fake_base, cgan_epochs)
    if reuse_if_exists and fake_base_dir is not None and fake_base_dir.exists():
        if verbose:
            print("\n[2] Baseline cGAN reuse:", fake_base_dir)
    else:
        if verbose:
            print("\n[2] Training cGAN baseline (label_noise_p=0.0)...")

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
        fake_base_dir = _resolve_epoch_dir(out_fake_base, cgan_epochs)

    if fake_base_dir is None:
        raise RuntimeError(f"Baseline fake dir not found under {out_fake_base}")

    if verbose:
        print(f"    baseline fake dir: {fake_base_dir}")

    # ---------- 3) cGAN label-noise ----------
    fake_ln_dir = _resolve_epoch_dir(out_fake_ln, cgan_epochs)
    if reuse_if_exists and fake_ln_dir is not None and fake_ln_dir.exists():
        if verbose:
            print(f"\n[3] Label-noise cGAN reuse (p={label_noise_p}):", fake_ln_dir)
    else:
        if verbose:
            print(f"\n[3] Training cGAN label-noise (p={label_noise_p})...")

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
        fake_ln_dir = _resolve_epoch_dir(out_fake_ln, cgan_epochs)

    if fake_ln_dir is None:
        raise RuntimeError(f"Label-noise fake dir not found under {out_fake_ln}")

    if verbose:
        print(f"    label-noise fake dir: {fake_ln_dir}")

    results = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "dataset_format": dataset_format,
        "fake_epoch_dir_baseline": str(fake_base_dir),
        "fake_epoch_dir_label_noise": str(fake_ln_dir),
        "has_inference_ds": "inference_ds" in extras,
    }

    # ---------- 4) CNN ratio experiments ----------
    # baseline fake
    if reuse_if_exists and out_csv_base.exists():
        if verbose:
            print("\n[4] CNN baseline reuse CSV:", out_csv_base)
        cnn_results_base = _load_cnn_results_from_csv(out_csv_base)
    else:
        if verbose:
            print("\n[4] CNN performance with baseline cGAN fake:")
            print(f"    ratios: {cnn_ratios}")
        cnn_results_base = run_ratio_experiments(
            train_ds,
            val_ds,
            test_ds,
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

    # label-noise fake
    if reuse_if_exists and out_csv_ln.exists():
        if verbose:
            print("\n[4b] CNN label-noise reuse CSV:", out_csv_ln)
        cnn_results_ln = _load_cnn_results_from_csv(out_csv_ln)
    else:
        if verbose:
            print("\n[4b] CNN performance with label-noise cGAN fake:")
            print(f"    ratios: {cnn_ratios} | p={label_noise_p}")
        cnn_results_ln = run_ratio_experiments(
            train_ds,
            val_ds,
            test_ds,
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

    results["cnn_results_baseline"] = cnn_results_base
    results["cnn_results_label_noise"] = cnn_results_ln
    results["cnn_csv_baseline"] = str(out_csv_base)
    results["cnn_csv_label_noise"] = str(out_csv_ln)

    # ---------- 5) FID/KID ----------
    if run_fid_kid:
        if reuse_if_exists and metrics_path.exists():
            if verbose:
                print("\n[5] FID/KID reuse:", metrics_path)
            with open(metrics_path, "r") as f:
                cached = json.load(f)
            results.update(cached)
        else:
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
            fid1, kidm1, kids1 = eval_folder_vs_real(
                real_images_ds,
                fake_ln_dir,
                max_samples=FID_MAX_SAMPLES,
                batch_size=FID_BATCH,
                image_size=CLS_IMG,
                seed=42,
            )

            fidkid = {
                "fid_baseline": fid0,
                "kid_baseline_mean": kidm0,
                "kid_baseline_std": kids0,
                "fid_label_noise": fid1,
                "kid_label_noise_mean": kidm1,
                "kid_label_noise_std": kids1,
            }
            results.update(fidkid)

            with open(metrics_path, "w") as f:
                json.dump(fidkid, f, indent=2)

            if verbose:
                print(f"    Baseline   FID: {fid0:.2f} | KID: {kidm0:.4f} ± {kids0:.4f}")
                print(f"    LabelNoise FID: {fid1:.2f} | KID: {kidm1:.4f} ± {kids1:.4f}")

    return results
