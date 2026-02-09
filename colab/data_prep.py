"""
Dataset-aware data preparation for Colab.

Supports three explicit dataset formats:
1) imagefolder_train_labeled_test_unlabeled (Cats-and-Dogs-Breed style)
2) imagefolder_supervised_train_test (SportsBall style)
3) mnist_idx (MNIST IDX binary files)
"""
from pathlib import Path
import struct

import numpy as np
import tensorflow as tf

try:
    import kagglehub
except ImportError:
    kagglehub = None


def _pipe(ds, *, cache=True, prefetch=True, seed=42, shuffle=False):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _imagefolder_train_val(train_dir, *, img_size, batch_size, seed, val_split, shuffle_train):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    return train_ds, val_ds


def _load_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file magic for {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _load_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file magic for {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if len(data) != n:
        raise ValueError(f"Label count mismatch in {path}: header={n}, read={len(data)}")
    return data


def _mnist_idx_ds(images: np.ndarray, labels: np.ndarray, *, img_size, batch_size, shuffle, seed):
    x = tf.convert_to_tensor(images[..., np.newaxis], dtype=tf.uint8)
    y = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)

    def _resize_norm(im, lab):
        im = tf.image.resize(im, img_size, method="bilinear")
        im = tf.cast(tf.round(im), tf.uint8)
        return im, lab

    ds = ds.map(_resize_norm, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size)


def _infer_unlabeled_ds(test_dir: Path, *, img_size, batch_size):
    files = sorted([p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    paths = tf.constant([str(p) for p in files])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        b = tf.io.read_file(path)
        img = tf.io.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(tf.round(img), tf.uint8)
        return img

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def prepare_ds(
    kaggle_dataset: str,
    *,
    dataset_format: str,
    img_size=(224, 224),
    batch_size=32,
    seed=42,
    val_split=0.2,
    shuffle_train=True,
    cache=True,
    prefetch=True,
    train_subdir="train",
    test_subdir="test",
):
    """
    Download a Kaggle dataset and build standardized tf.data datasets.

    Returns:
      train_ds, val_ds, test_ds, class_names, data_dir, extras

    extras may include:
      - inference_ds: unlabeled inference dataset (for Cats-and-Dogs-Breed TEST split)
    """
    if kagglehub is None:
        raise ImportError("pip install kagglehub")
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1), got {val_split}")

    data_dir = Path(kagglehub.dataset_download(kaggle_dataset))
    train_dir = data_dir / train_subdir
    test_dir = data_dir / test_subdir
    extras = {}

    if dataset_format == "imagefolder_supervised_train_test":
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"Expected train/test dirs not found under: {data_dir}")

        train_ds, val_ds = _imagefolder_train_val(
            train_dir,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed,
            val_split=val_split,
            shuffle_train=shuffle_train,
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
        )
        class_names = train_ds.class_names

    elif dataset_format == "imagefolder_train_labeled_test_unlabeled":
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"Expected TRAIN/TEST dirs not found under: {data_dir}")

        train_ds, val_ds = _imagefolder_train_val(
            train_dir,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed,
            val_split=val_split,
            shuffle_train=shuffle_train,
        )
        # No labels in TEST: use validation as supervised test fallback for metrics.
        test_ds = val_ds
        class_names = train_ds.class_names
        extras["inference_ds"] = _infer_unlabeled_ds(
            test_dir,
            img_size=img_size,
            batch_size=batch_size,
        )

    elif dataset_format == "mnist_idx":
        req = {
            "train_images": data_dir / "train-images.idx3-ubyte",
            "train_labels": data_dir / "train-labels.idx1-ubyte",
            "test_images": data_dir / "t10k-images.idx3-ubyte",
            "test_labels": data_dir / "t10k-labels.idx1-ubyte",
        }
        missing = [str(p) for p in req.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"MNIST IDX files missing: {missing}")

        train_images = _load_idx_images(req["train_images"])
        train_labels = _load_idx_labels(req["train_labels"])
        test_images = _load_idx_images(req["test_images"])
        test_labels = _load_idx_labels(req["test_labels"])

        n_train = len(train_labels)
        idx = np.arange(n_train)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = int(n_train * val_split)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        train_ds = _mnist_idx_ds(
            train_images[tr_idx], train_labels[tr_idx],
            img_size=img_size, batch_size=batch_size, shuffle=shuffle_train, seed=seed,
        )
        val_ds = _mnist_idx_ds(
            train_images[val_idx], train_labels[val_idx],
            img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed,
        )
        test_ds = _mnist_idx_ds(
            test_images, test_labels,
            img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed,
        )
        class_names = [str(i) for i in range(10)]

    else:
        raise ValueError(
            f"Unsupported dataset_format: {dataset_format}. "
            "Use one of imagefolder_supervised_train_test, "
            "imagefolder_train_labeled_test_unlabeled, mnist_idx."
        )

    train_ds = _pipe(train_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=shuffle_train)
    val_ds = _pipe(val_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=False)
    test_ds = _pipe(test_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=False)

    print(f"Prepared {len(class_names)} classes | format={dataset_format} | data_dir={data_dir}")
    return train_ds, val_ds, test_ds, class_names, data_dir, extras
