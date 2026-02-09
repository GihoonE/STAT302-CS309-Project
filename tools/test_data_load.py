"""
Dataset download + load sanity check.
- downloads via kagglehub (needs auth on that machine)
- builds train/val/test
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
        train_subdir=cfg.get("train_subdir", "train"),
        test_subdir=cfg.get("test_subdir", "test"),
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        val_split=0.2,
    )

    print(f"\n✅ data_dir: {data_dir}")
    print(f"✅ num_classes: {len(class_names)}")
    print(f"✅ class_names[:10]: {class_names[:10]}")
    print(f"✅ has_inference_ds: {'inference_ds' in extras}")

    # Grab one batch
    x, y = next(iter(train_ds))
    print(f"\nTrain batch images shape: {x.shape}, dtype={x.dtype}")
    print(f"Train batch labels shape: {y.shape}, dtype={y.dtype}")
    print(f"Labels sample: {y[:10].numpy()}")

    x2, y2 = next(iter(test_ds))
    print(f"\nTest batch images shape: {x2.shape}, dtype={x2.dtype}")
    print(f"Test batch labels shape: {y2.shape}, dtype={y2.dtype}")

    print("\n✅ DATA LOADING OK")


if __name__ == "__main__":
    main()
