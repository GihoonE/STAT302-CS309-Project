"""
cGAN module for Colab: Generator, Discriminator, training, sampling, noisy clone.
TensorFlow/Keras. Expects RGB images (e.g. 64x64x3); [-1,1] normalized.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_gan_ds(train_dir, *, img_size=(64, 64), batch_size=64, seed=42, shuffle=True):
    """Build tf.data.Dataset from train dir; images normalized to [-1, 1]."""
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    def norm(x, y):
        x = tf.cast(x, tf.float32)
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = (x / 127.5) - 1.0
        return x, y

    return ds.map(norm, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_generator(latent_dim, num_classes, img_size=(64, 64), base_ch=64, out_channels=3):
    """Conditional Generator: (z, label) -> image in [-1,1]. img_size must be divisible by 8."""
    H, W = img_size
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError("img_size must be divisible by 8.")

    z_in = keras.Input(shape=(latent_dim,), name="z")
    y_in = keras.Input(shape=(), dtype=tf.int32, name="label")

    y_emb = layers.Embedding(num_classes, latent_dim)(y_in)
    y_emb = layers.Flatten()(y_emb)
    x = layers.Concatenate()([z_in, y_emb])

    h0, w0 = H // 8, W // 8
    x = layers.Dense(h0 * w0 * base_ch * 4, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((h0, w0, base_ch * 4))(x)

    x = layers.Conv2DTranspose(base_ch * 2, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(base_ch, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(base_ch // 2, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out = layers.Conv2D(out_channels, 3, padding="same", activation="tanh")(x)
    return keras.Model([z_in, y_in], out, name="Generator")


def build_discriminator(num_classes, img_size=(64, 64), base_ch=64, in_channels=3):
    """Conditional Discriminator: (image, label) -> logits."""
    H, W = img_size
    x_in = keras.Input(shape=(H, W, in_channels), name="img")
    y_in = keras.Input(shape=(), dtype=tf.int32, name="label")

    y_emb = layers.Embedding(num_classes, H * W)(y_in)
    y_emb = layers.Reshape((H, W, 1))(y_emb)
    x = layers.Concatenate()([x_in, y_emb])

    x = layers.Conv2D(base_ch, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(base_ch * 2, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(base_ch * 4, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    return keras.Model([x_in, y_in], out, name="Discriminator")


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_loss_fn(real_logits, fake_logits, smooth=0.9):
    real_loss = bce(tf.ones_like(real_logits) * smooth, real_logits)
    fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def g_loss_fn(fake_logits):
    return bce(tf.ones_like(fake_logits), fake_logits)


def corrupt_labels(y, num_classes, p=0.2):
    """With prob p, replace y with a random class label."""
    bs = tf.shape(y)[0]
    mask = tf.random.uniform((bs,)) < p
    random_y = tf.random.uniform((bs,), 0, num_classes, dtype=tf.int32)
    return tf.where(mask, random_y, y)


def save_fake_images(G, out_root, class_names, latent_dim, *, n_per_class=50, seed=123):
    """Save generated images per class: out_root/<class_name>/img_0000.png, ..."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cid, cname in enumerate(class_names):
        (out_root / cname).mkdir(parents=True, exist_ok=True)

        z = tf.random.normal((n_per_class, latent_dim), seed=seed)
        y = tf.constant([cid] * n_per_class, dtype=tf.int32)
        imgs = G([z, y], training=False)
        imgs = (imgs + 1.0) * 127.5
        imgs = tf.cast(tf.clip_by_value(imgs, 0, 255), tf.uint8)

        for i in range(n_per_class):
            png = tf.io.encode_png(imgs[i])
            tf.io.write_file(str(out_root / cname / f"img_{i:04d}.png"), png)


def train_cgan(
    gan_ds,
    G,
    D,
    class_names,
    latent_dim,
    num_classes,
    *,
    label_noise_p=0.0,
    epochs=30,
    sample_every=5,
    out_dir="/content/fake_samples",
):
    """Train cGAN; save fake samples every sample_every epochs under out_dir."""
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(real_x, real_y):
        bs = tf.shape(real_x)[0]
        z = tf.random.normal((bs, latent_dim))

        noisy_y = corrupt_labels(real_y, num_classes, p=label_noise_p) if label_noise_p > 0.0 else real_y
        fake_y = noisy_y

        with tf.GradientTape() as dtape, tf.GradientTape() as gtape:
            fake_x = G([z, fake_y], training=True)
            real_logits = D([real_x, noisy_y], training=True)
            fake_logits = D([fake_x, fake_y], training=True)

            d_loss = d_loss_fn(real_logits, fake_logits)
            g_loss = g_loss_fn(fake_logits)

        d_grads = dtape.gradient(d_loss, D.trainable_variables)
        g_grads = gtape.gradient(g_loss, G.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, D.trainable_variables))
        g_opt.apply_gradients(zip(g_grads, G.trainable_variables))
        return d_loss, g_loss

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for real_x, real_y in gan_ds:
            d_loss, g_loss = train_step(real_x, real_y)
            d_losses.append(float(d_loss.numpy()))
            g_losses.append(float(g_loss.numpy()))

        print(f"Epoch {epoch:03d} | D_loss={np.mean(d_losses):.4f} | G_loss={np.mean(g_losses):.4f}")

        if epoch % sample_every == 0:
            epoch_dir = out_dir / f"epoch_{epoch:03d}"
            save_fake_images(G, epoch_dir, class_names, latent_dim, n_per_class=50, seed=epoch)


# --------------- Noisy clone (real images with gaussian/jpeg noise) ---------------


def _read_uint8(path: str) -> tf.Tensor:
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    return img


def _write_image_uint8(img: tf.Tensor, out_path: str, ext: str = "jpg"):
    img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    if ext.lower() == "png":
        encoded = tf.io.encode_png(img)
    else:
        encoded = tf.io.encode_jpeg(img, quality=95)
    tf.io.write_file(out_path, encoded)


def _gaussian(img, sigma=0.10):
    x = tf.cast(img, tf.float32) / 255.0
    x = x + tf.random.normal(tf.shape(x), stddev=sigma)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return tf.cast(x * 255.0, tf.uint8)


def _jpeg(img, quality=40):
    jpg = tf.io.encode_jpeg(img, quality=int(quality))
    return tf.io.decode_jpeg(jpg, channels=3)


def make_noisy_clone(
    src_dir,
    dst_dir,
    *,
    noise="gaussian",
    sigma=0.10,
    jpeg_quality=40,
    out_ext="jpg",
    suffix=None,
):
    """
    Copy src_dir (train/test with class subdirs) to dst_dir, applying noise per image.
    noise: "gaussian" | "jpeg"
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    class_dirs = [p for p in src_dir.iterdir() if p.is_dir()]
    if suffix is None:
        suffix = noise

    for cdir in class_dirs:
        out_cdir = dst_dir / cdir.name
        out_cdir.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        for p in files:
            img = _read_uint8(str(p))
            if noise == "gaussian":
                out = _gaussian(img, sigma=sigma)
            elif noise == "jpeg":
                out = _jpeg(img, quality=jpeg_quality)
            else:
                raise ValueError(f"Unknown noise type: {noise}")

            out_path = out_cdir / f"{p.stem}_{suffix}.{out_ext}"
            _write_image_uint8(out, str(out_path), ext=out_ext)

    print("✅ Done:", dst_dir)
