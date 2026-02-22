"""
Dataset download + load sanity check.
- downloads via kagglehub (needs auth on that machine)
- builds train/val/test
- prints shapes, class names, and grabs 1 batch
"""

import argparse
import tensorflow as tf

from experiment_part23.datasets_config import DATASETS
from experiment_part23.data_prep import prepare_ds


def _peek_one_batch(ds):
    b = next(iter(ds))
    if isinstance(b, (tuple, list)) and len(b) == 2:
        x, y = b
        return x, y
    # images-only (unlabeled)
    return b, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_key", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--test_split", type=float, default=0.2, help="Only used for mnist_csv")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset_key]
    print("=" * 70)
    print(f"Dataset key: {args.dataset_key}")
    print(f"Name       : {cfg['name']}")
    print(f"Slug       : {cfg['slug']}")
    print(f"Format     : {cfg['dataset_format']}")
    print("=" * 70)

    train_ds, val_ds, test_ds, class_names, data_dir, extras = prepare_ds(
        cfg["slug"],
        dataset_format=cfg["dataset_format"],
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=cfg.get("seed", 42),
        val_split=args.val_split,
        test_split=cfg.get("test_split", args.test_split),  # mnist_csv only
        shuffle_train=cfg.get("shuffle_train", True),
        cache=cfg.get("cache", True),
        prefetch=cfg.get("prefetch", True),
        train_subdir=cfg.get("train_subdir", "train"),
        val_subdir=cfg.get("val_subdir", None),              # NEW
        test_subdir=cfg.get("test_subdir", "test"),
    )

    print(f"\n✅ data_dir: {data_dir}")
    print(f"✅ num_classes: {len(class_names)}")
    print(f"✅ class_names[:10]: {class_names[:10]}")
    print(f"✅ has_inference_ds: {'inference_ds' in extras}")
    if "source" in extras:
        print(f"✅ extras['source']: {extras['source']}")

    # Grab one batch from train
    x, y = _peek_one_batch(train_ds)
    print(f"\nTrain batch images shape: {x.shape}, dtype={x.dtype}")
    if y is not None:
        print(f"Train batch labels shape: {y.shape}, dtype={y.dtype}")
        print(f"Labels sample: {y[:10].numpy()}")
    else:
        print("Train batch is unlabeled (images-only).")

    # Grab one batch from test (might be val fallback or real test)
    x2, y2 = _peek_one_batch(test_ds)
    print(f"\nTest batch images shape: {x2.shape}, dtype={x2.dtype}")
    if y2 is not None:
        print(f"Test batch labels shape: {y2.shape}, dtype={y2.dtype}")
    else:
        print("Test batch is unlabeled (images-only).")

    # If we have inference_ds, peek it too
    if "inference_ds" in extras:
        xinfer, yinfer = _peek_one_batch(extras["inference_ds"])
        print(f"\nInference batch images shape: {xinfer.shape}, dtype={xinfer.dtype}")
        if yinfer is not None:
            print(f"Inference labels shape: {yinfer.shape}, dtype={yinfer.dtype}")
        else:
            print("Inference batch is unlabeled (images-only).")

    print("\n✅ DATA LOADING OK")


if __name__ == "__main__":
    main()
