"""
Dataset download + load sanity check.
- downloads via kagglehub (needs auth on that machine)
- builds train/val/test (+ optional inference_ds)
- prints shapes, class names, and grabs 1 batch
"""

import argparse
import tensorflow as tf

from colab.datasets_config import DATASETS
from colab.data_prep import prepare_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_key", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--test_split", type=float, default=0.2, help="Used only for mnist_csv")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset_key]

    print("=" * 70)
    print(f"Dataset key: {args.dataset_key}")
    print(f"Name       : {cfg['name']}")
    print(f"Slug       : {cfg['slug']}")
    print(f"Format     : {cfg['dataset_format']}")
    print("=" * 70)

    # New: pass val_subdir + test_split if provided
    train_ds, val_ds, test_ds, class_names, data_dir, extras = prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=cfg.get("seed", 42),
        val_split=args.val_split,
        test_split=args.test_split,
        shuffle_train=cfg.get("shuffle_train", True),
        cache=cfg.get("cache", True),
        prefetch=cfg.get("prefetch", True),
        train_subdir=cfg.get("train_subdir", "train"),
        val_subdir=cfg.get("val_subdir", None),          # NEW
        test_subdir=cfg.get("test_subdir", "test"),
    )

    print(f"\n✅ data_dir: {data_dir}")
    print(f"✅ num_classes: {len(class_names)}")
    print(f"✅ class_names[:10]: {class_names[:10]}")
    print(f"✅ has_inference_ds: {'inference_ds' in extras}")

    # --- Train batch ---
    x, y = next(iter(train_ds))
    print(f"\nTrain batch images shape: {x.shape}, dtype={x.dtype}")
    print(f"Train batch labels shape: {y.shape}, dtype={y.dtype}")
    try:
        print(f"Labels sample: {y[:10].numpy()}")
    except Exception:
        # just in case y is not a tensor-like
        print("Labels sample: <unavailable>")

    # --- Test batch ---
    # Some formats may set test_ds = val_ds; still labeled.
    if test_ds is not None:
        batch = next(iter(test_ds))
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x2, y2 = batch
            print(f"\nTest batch images shape: {x2.shape}, dtype={x2.dtype}")
            print(f"Test batch labels shape: {y2.shape}, dtype={y2.dtype}")
        else:
            # unlabeled images-only dataset (rare here, but safe)
            x2 = batch
            print(f"\nTest batch images shape: {x2.shape}, dtype={x2.dtype}")
            print("Test batch labels shape: <none> (unlabeled)")

    # --- Inference batch (unlabeled) ---
    if "inference_ds" in extras and extras["inference_ds"] is not None:
        xb = next(iter(extras["inference_ds"]))
        print(f"\nInference batch images shape: {xb.shape}, dtype={xb.dtype}")

    print("\n✅ DATA LOADING OK")


if __name__ == "__main__":
    main()
