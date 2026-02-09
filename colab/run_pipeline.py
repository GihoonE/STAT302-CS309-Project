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
    return prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        train_subdir=cfg.get("train_subdir", "train"),
        test_subdir=cfg.get("test_subdir", "test"),
        **kwargs,
    )


def _build_gan_ds_from_train(train_ds, *, img_size=(64, 64)):
    """train_ds: (image,label) batches. Convert to [-1,1] RGB and resize to GAN size."""
    def to_gan(x, y):
        x = tf.image.resize(x, img_size)
        # ensure RGB
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = tf.cast(x, tf.float32)
        x = (x / 127.5) - 1.0
        y = tf.cast(y, tf.int32)
        return x, y
    return train_ds.map(to_gan, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def run_pipeline_one(
    dataset_key,
    out_root="/content/results",
    *,
    # cGAN
    cgan_epochs=CGAN_EPOCHS,
    sample_every=SAMPLE_EVERY,
    label_noise_p=0.2,          # << 핵심: label-noise cGAN 확률
    n_show_samples=5,           # 현재 show 함수는 그대로 두고 싶으면 유지 (아래에서 skip 가능)
    # CNN
    cnn_ratios=None,            # 기본은 아래에서 [1.0,0.8,0.7,0.6,0.5]
    cnn_epochs_stage1=EPOCHS_STAGE1,
    cnn_epochs_stage2=EPOCHS_STAGE2,
    # Eval
    run_fid_kid=True,
    verbose=1,
):
    """
    목표:
    1) 데이터셋별 cGAN 학습 후 이미지 생성 (baseline + label-noise 둘 다)
    2) CNN classifier 성능 체크 (100%, 80%+20%, 70%+30%, 60%+40%, 50%+50%)
    3) FID/KID 성능 체크 (baseline cGAN vs real, label-noise cGAN vs real)

    Returns dict:
      - data_dir, class_names
      - fake_epoch_dir_baseline, fake_epoch_dir_label_noise
      - cnn_results_baseline (baseline fake 사용)
      - fid_baseline / kid_baseline
      - fid_label_noise / kid_label_noise
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset_key: {dataset_key}. Choose from {list(DATASETS)}")

    cfg = DATASETS[dataset_key]
    name = cfg["name"]
    dataset_format = cfg["dataset_format"]

    # CNN ratios default: 100/80/70/60/50
    if cnn_ratios is None:
        cnn_ratios = [1.0, 0.8, 0.7, 0.6, 0.5]

    out_base = Path(out_root) / dataset_key
    out_base.mkdir(parents=True, exist_ok=True)

    out_fake_base = out_base / "fake_samples" / "baseline"
    out_fake_ln = out_base / "fake_samples" / f"label_noise_p{int(label_noise_p*100):02d}"
    out_csv = out_base / "cnn_ratio_results.csv"

    if verbose:
        print("\n" + "=" * 70)
        print(f"Dataset: {name} ({dataset_key}) | format={dataset_format}")
        print("=" * 70)

    # ---------- 1) Data ----------
    if verbose:
        print("[1] Loading data via prepare_ds(kagglehub)...")

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

    # Ensure RGB for classifier (MNIST etc.)
    def ensure_rgb(x, y):
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        return x, y

    train_ds = train_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # ---------- 2) cGAN baseline ----------
    if verbose:
        print("\n[2] Training cGAN baseline (label_noise_p=0.0)...")

    # GAN ds: take finite train stream, convert to GAN size and [-1,1]
    gan_ds = _build_gan_ds_from_train(train_ds.unbatch().batch(GAN_BATCH), img_size=GAN_IMG)

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

    fake_base_dir = out_fake_base / f"epoch_{cgan_epochs:03d}"
    if not fake_base_dir.exists():
        fake_base_dir = out_fake_base / f"epoch_{sample_every:03d}"  # fallback
    if verbose:
        print(f"    baseline fake dir: {fake_base_dir}")

    # ---------- 3) cGAN label-noise ----------
    if verbose:
        print(f"\n[3] Training cGAN label-noise (label_noise_p={label_noise_p})...")

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

    fake_ln_dir = out_fake_ln / f"epoch_{cgan_epochs:03d}"
    if not fake_ln_dir.exists():
        fake_ln_dir = out_fake_ln / f"epoch_{sample_every:03d}"  # fallback
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

    # ---------- 4) CNN ratio experiments (baseline fake 사용) ----------
    if verbose:
        print("\n[4] CNN performance with baseline cGAN fake:")
        print(f"    ratios: {cnn_ratios}")

    cnn_results = run_ratio_experiments(
        train_ds,
        val_ds,
        test_ds,
        class_names,
        fake_base_dir,                 # baseline fake만 섞어서 성능 체크
        ratios=cnn_ratios,
        img_size=CLS_IMG,
        batch_size=CLS_BATCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs_stage1=cnn_epochs_stage1,
        epochs_stage2=cnn_epochs_stage2,
        out_csv=str(out_csv),
        verbose=verbose,
    )
    results["cnn_results_baseline"] = cnn_results

    # ---------- 5) FID/KID: real vs baseline, real vs label-noise ----------
    if run_fid_kid:
        if verbose:
            print("\n[5] FID/KID (real test vs fake folders) ...")

        real_images_ds = real_ds_to_images_only(test_ds)

        # baseline
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

        # label-noise
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
