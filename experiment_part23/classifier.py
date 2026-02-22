"""
CNN classification for performance check.
Runs experiments: 100% original, 90% orig + 10% cGAN, 80%+20%, 70%+30%, 60%+40%, 50%+50%.
Uses EfficientNetB0; train/val/test from prepare_ds; fake images from a folder (e.g. cGAN samples).
"""
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DEFAULT_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def build_effnet_classifier(img_size=(224, 224), num_classes=15, dropout=0.3):
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.10)(x)

    # EfficientNet preprocess expects float input; typically works well with 0..255 float
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,),
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="effnetb0_classifier")
    return model, base


def _force_float32_255(ds):
    """
    Make (x,y) -> (float32 image in 0..255, int32 label)
    - if x is uint8: cast to float32 (values stay 0..255)
    - if x is float32 but in 0..1: scale up to 0..255
    """
    def _fix(x, y):
        x = tf.cast(x, tf.float32)
        # if looks like 0..1, scale to 0..255
        x = tf.cond(tf.reduce_max(x) <= 1.5, lambda: x * 255.0, lambda: x)
        y = tf.cast(y, tf.int32)
        return x, y

    return ds.map(_fix, num_parallel_calls=tf.data.AUTOTUNE)


def _make_fake_ds(fake_folder, *, img_size=(224, 224), batch_size=32, shuffle=True, seed=42):
    """Load fake image folder (with class subdirs) as (images, labels) dataset."""
    ds = tf.keras.utils.image_dataset_from_directory(
        str(fake_folder),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        labels="inferred",
        label_mode="int",
    )
    ds = _force_float32_255(ds)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_mixed_train_ds(
    original_train_ds,
    fake_folder,
    orig_ratio,
    *,
    img_size=(224, 224),
    batch_size=32,
    steps_per_epoch=200,
    seed=42,
):
    """
    Build training dataset: orig_ratio from original, (1 - orig_ratio) from fake.
    """
    # ✅ ALWAYS force original to float32 (even for 100% original)
    original_train_ds2 = _force_float32_255(original_train_ds).prefetch(tf.data.AUTOTUNE)

    if orig_ratio >= 1.0:
        return original_train_ds2, None

    fake_ds2 = _make_fake_ds(fake_folder, img_size=img_size, batch_size=batch_size, shuffle=True, seed=seed)

    mixed = tf.data.Dataset.sample_from_datasets(
        [original_train_ds2.repeat(), fake_ds2.repeat()],
        weights=[orig_ratio, 1.0 - orig_ratio],
        seed=seed,
    ).prefetch(tf.data.AUTOTUNE)

    return mixed, steps_per_epoch


def train_and_evaluate(
    train_ds,
    val_ds,
    test_ds,
    num_classes,
    *,
    img_size=(224, 224),
    steps_per_epoch=None,
    epochs_stage1=15,
    epochs_stage2=10,
    dropout=0.3,
    fine_tune_layers=40,
    verbose=1,
):
    model, base = build_effnet_classifier(img_size=img_size, num_classes=num_classes, dropout=dropout)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6, monitor="val_loss"),
    ]

    fit_kw = dict(
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks,
        verbose=verbose,
    )
    if steps_per_epoch is not None:
        fit_kw["steps_per_epoch"] = steps_per_epoch

    model.fit(train_ds, **fit_kw)

    # Stage 2: fine-tune
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fit_kw["epochs"] = epochs_stage2
    model.fit(train_ds, **fit_kw)

    _, test_acc = model.evaluate(test_ds, verbose=verbose)
    return float(test_acc)


def run_ratio_experiments(
    original_train_ds,
    val_ds,
    test_ds,
    class_names,
    fake_folder,
    *,
    ratios=None,
    img_size=(224, 224),
    batch_size=32,
    steps_per_epoch=200,
    epochs_stage1=15,
    epochs_stage2=10,
    seed=42,
    out_csv=None,
    verbose=1,
):
    if ratios is None:
        ratios = DEFAULT_RATIOS

    num_classes = len(class_names)
    fake_folder = Path(fake_folder)
    results = []

    # ✅ ALSO force val/test to float32 0..255
    val_ds2 = _force_float32_255(val_ds).prefetch(tf.data.AUTOTUNE)
    test_ds2 = _force_float32_255(test_ds).prefetch(tf.data.AUTOTUNE)

    for orig_ratio in ratios:
        if verbose:
            print(f"\n--- {int(orig_ratio*100)}% original + {int((1-orig_ratio)*100)}% cGAN ---")

        train_ds, steps = make_mixed_train_ds(
            original_train_ds,
            fake_folder,
            orig_ratio,
            img_size=img_size,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
        )

        test_acc = train_and_evaluate(
            train_ds,
            val_ds2,
            test_ds2,
            num_classes,
            img_size=img_size,
            steps_per_epoch=steps,
            epochs_stage1=epochs_stage1,
            epochs_stage2=epochs_stage2,
            verbose=verbose,
        )

        results.append((orig_ratio, test_acc))
        if verbose:
            print(f"  Test accuracy: {test_acc:.4f}")

        if out_csv:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            file_exists = out_csv.exists()
            with open(out_csv, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("orig_ratio,cgan_ratio,test_accuracy\n")
                f.write(f"{orig_ratio:.2f},{1-orig_ratio:.2f},{test_acc:.6f}\n")

    return results
