"""
Dataset-aware data preparation.

Supports formats:
1) imagefolder_train_labeled_test_unlabeled
2) imagefolder_supervised_train_test
3) mnist_idx
4) mnist_csv
5) mnist_keras   (tf.keras.datasets.mnist)
"""
from pathlib import Path
import struct

import numpy as np
import tensorflow as tf

try:
    import kagglehub
except ImportError:
    kagglehub = None


# -----------------------------
# Pipeline helpers
# -----------------------------
def _pipe(ds, *, cache=True, prefetch=True, seed=42, shuffle=False):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _imagefolder_train_val_split(train_dir, *, img_size, batch_size, seed, val_split, shuffle_train):
    """Make train/val by splitting train_dir."""
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


def _imagefolder_dir(dir_path: Path, *, img_size, batch_size, shuffle, label_mode="int", seed=42):
    """Load one labeled imagefolder directory."""
    return tf.keras.utils.image_dataset_from_directory(
        dir_path,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode=label_mode,
        seed=seed if shuffle else None,
    )


# -----------------------------
# IDX loaders (MNIST)
# -----------------------------
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


def _mnist_np_to_ds(images: np.ndarray, labels: np.ndarray, *, img_size, batch_size, shuffle, seed):
    """
    images: (N,28,28) uint8
    labels: (N,) int
    Output: (N,H,W,1) resized uint8 + labels
    """
    x = tf.convert_to_tensor(images[..., np.newaxis], dtype=tf.uint8)
    y = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)

    def _resize(im, lab):
        im = tf.image.resize(im, img_size, method="bilinear")
        im = tf.cast(tf.round(im), tf.uint8)
        return im, lab

    ds = ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size)


# -----------------------------
# Unlabeled inference folder -> images only
# -----------------------------
def _infer_unlabeled_ds(test_dir: Path, *, img_size, batch_size):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted([p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
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


# -----------------------------
# MNIST CSV loader
# -----------------------------
def _load_mnist_csv(csv_path: Path):
    """
    Expect: label + 784 pixel columns (0..255).
    Works for: mnist_dataset.csv (single file).
    """
    # np.loadtxt is too slow for big CSV; use tf.io + tf.strings? simplest robust: numpy genfromtxt.
    # For 70k rows x 785 cols, genfromtxt is ok-ish on server; if slow, we can swap to pandas.
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected CSV shape: {data.shape} for {csv_path}")

    labels = data[:, 0].astype(np.int32)
    pixels = data[:, 1:]
    if pixels.shape[1] != 784:
        raise ValueError(f"Expected 784 pixel columns, got {pixels.shape[1]} in {csv_path}")

    images = pixels.reshape(-1, 28, 28)
    images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
    return images, labels


def _split_train_val_test(images, labels, *, seed=42, val_split=0.2, test_split=0.2):
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_split)
    test_idx = idx[:n_test]
    rem_idx = idx[n_test:]

    n_val = int(len(rem_idx) * val_split)
    val_idx = rem_idx[:n_val]
    tr_idx = rem_idx[n_val:]

    return (images[tr_idx], labels[tr_idx]), (images[val_idx], labels[val_idx]), (images[test_idx], labels[test_idx])


# -----------------------------
# Main API
# -----------------------------
def prepare_ds(
    kaggle_dataset: str,
    *,
    dataset_format: str,
    img_size=(224, 224),
    batch_size=32,
    seed=42,
    val_split=0.2,
    test_split=0.2,          # used for mnist_csv only
    shuffle_train=True,
    cache=True,
    prefetch=True,
    train_subdir="train",
    val_subdir=None,
    test_subdir="test",
):
    """
    Returns:
      train_ds, val_ds, test_ds, class_names, data_dir, extras
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1), got {val_split}")
    if dataset_format == "mnist_csv" and not (0.0 < test_split < 1.0):
        raise ValueError(f"test_split must be in (0,1), got {test_split}")

    extras = {}

    # ✅ mnist_keras는 kagglehub 자체를 사용하지 않음
    if dataset_format == "mnist_keras":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train: (60000,28,28) uint8 / y_train: (60000,)
        # split train -> train/val
        n_train = len(y_train)
        idx = np.arange(n_train)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        n_val = int(n_train * val_split)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        train_ds = _mnist_np_to_ds(
            x_train[tr_idx], y_train[tr_idx],
            img_size=img_size, batch_size=batch_size, shuffle=shuffle_train, seed=seed
        )
        val_ds = _mnist_np_to_ds(
            x_train[val_idx], y_train[val_idx],
            img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
        )
        test_ds = _mnist_np_to_ds(
            x_test, y_test,
            img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
        )
        class_names = [str(i) for i in range(10)]
        data_dir = Path("<keras_builtin_mnist>")
        extras["source"] = "mnist_keras"

    else:
        # ✅ 나머지(sports_ball/animals/idx/csv)는 kagglehub 필요
        if kagglehub is None:
            raise ImportError("pip install kagglehub")

        data_dir = Path(kagglehub.dataset_download(kaggle_dataset))

        train_dir = data_dir / train_subdir if train_subdir else None
        val_dir = data_dir / val_subdir if val_subdir else None
        test_dir = data_dir / test_subdir if test_subdir else None

        if dataset_format == "imagefolder_supervised_train_test":
            if not train_dir or not train_dir.exists() or not test_dir or not test_dir.exists():
                raise FileNotFoundError(f"Expected train/test dirs not found under: {data_dir}")

            train_ds, val_ds = _imagefolder_train_val_split(
                train_dir,
                img_size=img_size,
                batch_size=batch_size,
                seed=seed,
                val_split=val_split,
                shuffle_train=shuffle_train,
            )
            test_ds = _imagefolder_dir(
                test_dir, img_size=img_size, batch_size=batch_size, shuffle=False, label_mode="int"
            )
            class_names = train_ds.class_names

        elif dataset_format == "imagefolder_train_labeled_test_unlabeled":
            if not train_dir or not train_dir.exists():
                raise FileNotFoundError(f"Expected TRAIN dir not found under: {data_dir}")
            if not test_dir or not test_dir.exists():
                raise FileNotFoundError(f"Expected TEST/INF dir not found under: {data_dir}")

            if val_dir and val_dir.exists():
                train_ds = _imagefolder_dir(
                    train_dir, img_size=img_size, batch_size=batch_size,
                    shuffle=shuffle_train, label_mode="int", seed=seed
                )
                val_ds = _imagefolder_dir(
                    val_dir, img_size=img_size, batch_size=batch_size,
                    shuffle=False, label_mode="int"
                )
            else:
                train_ds, val_ds = _imagefolder_train_val_split(
                    train_dir,
                    img_size=img_size,
                    batch_size=batch_size,
                    seed=seed,
                    val_split=val_split,
                    shuffle_train=shuffle_train,
                )

            test_ds = val_ds
            class_names = train_ds.class_names
            extras["inference_ds"] = _infer_unlabeled_ds(test_dir, img_size=img_size, batch_size=batch_size)

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

            train_ds = _mnist_np_to_ds(
                train_images[tr_idx], train_labels[tr_idx],
                img_size=img_size, batch_size=batch_size, shuffle=shuffle_train, seed=seed
            )
            val_ds = _mnist_np_to_ds(
                train_images[val_idx], train_labels[val_idx],
                img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
            )
            test_ds = _mnist_np_to_ds(
                test_images, test_labels,
                img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
            )
            class_names = [str(i) for i in range(10)]

        elif dataset_format == "mnist_csv":
            csv_path = data_dir / "mnist_dataset.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"MNIST CSV missing: {csv_path}")

            images, labels = _load_mnist_csv(csv_path)
            (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = _split_train_val_test(
                images, labels, seed=seed, val_split=val_split, test_split=test_split
            )

            train_ds = _mnist_np_to_ds(x_tr, y_tr, img_size=img_size, batch_size=batch_size, shuffle=shuffle_train, seed=seed)
            val_ds = _mnist_np_to_ds(x_val, y_val, img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed)
            test_ds = _mnist_np_to_ds(x_te, y_te, img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed)
            class_names = [str(i) for i in range(10)]
            extras["source"] = "mnist_csv"

        else:
            raise ValueError(
                f"Unsupported dataset_format: {dataset_format}. "
                "Use one of imagefolder_supervised_train_test, "
                "imagefolder_train_labeled_test_unlabeled, mnist_idx, mnist_csv, mnist_keras."
            )

    # pipeline
    train_ds = _pipe(train_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=shuffle_train)
    val_ds = _pipe(val_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=False)
    test_ds = _pipe(test_ds, cache=cache, prefetch=prefetch, seed=seed, shuffle=False)

    print(f"Prepared {len(class_names)} classes | format={dataset_format} | data_dir={data_dir}")
    return train_ds, val_ds, test_ds, class_names, data_dir, extras

