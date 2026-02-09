"""
Data preparation for Colab.
Downloads Kaggle dataset via kagglehub and builds train/val/test tf.data.Dataset.
Expected structure: <data_dir>/train/<class>/*.jpg, <data_dir>/test/<class>/*.jpg
"""
from pathlib import Path

import tensorflow as tf

try:
    import kagglehub
except ImportError:
    kagglehub = None


def prepare_ds(
    kaggle_dataset: str,
    *,
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
    Download Kaggle dataset via kagglehub and build train/val/test tf.data.Dataset.

    Expected structure after download:
      <data_dir>/train/<class>/*.(jpg|png|...)
      <data_dir>/test/<class>/*.(jpg|png|...)

    Returns:
      train_ds, val_ds, test_ds, class_names, data_dir (Path)
    """
    if kagglehub is None:
        raise ImportError("pip install kagglehub")

    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1), got {val_split}")

    data_dir = Path(kagglehub.dataset_download(kaggle_dataset))
    train_dir = data_dir / train_subdir
    test_dir = data_dir / test_subdir

    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

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

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    if class_names != test_ds.class_names:
        print("Warning: class_names differ between train and test.")
        print("train:", class_names)
        print("test :", test_ds.class_names)
    else:
        print("class_names match.")

    AUTOTUNE = tf.data.AUTOTUNE

    def _pipe(ds, *, do_shuffle=False):
        if cache:
            ds = ds.cache()
        if do_shuffle:
            ds = ds.shuffle(1000, seed=seed)
        if prefetch:
            ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = _pipe(train_ds, do_shuffle=shuffle_train)
    val_ds = _pipe(val_ds, do_shuffle=False)
    test_ds = _pipe(test_ds, do_shuffle=False)

    print(f"Prepared {len(class_names)} classes | data_dir={data_dir}")
    return train_ds, val_ds, test_ds, class_names, data_dir
