# Colab-ready modules: data_prep, cgan, fid_kid, classifier
# Usage: ensure project root is on sys.path, then:
#   from colab import prepare_ds, build_gan_ds, train_cgan, run_ratio_experiments, ...

from .data_prep import prepare_ds
from .cgan import (
    build_gan_ds,
    build_generator,
    build_discriminator,
    train_cgan,
    save_fake_images,
    make_noisy_clone,
    corrupt_labels,
)
from .fid_kid import (
    eval_folder_vs_real,
    compute_fid_kid_from_datasets,
    real_ds_to_images_only,
    folder_to_ds,
)
from .classifier import (
    build_effnet_classifier,
    make_mixed_train_ds,
    train_and_evaluate,
    run_ratio_experiments,
    DEFAULT_RATIOS,
)
from .datasets_config import DATASETS, KAGGLE_URLS
from .run_pipeline import run_pipeline_one, run_all_datasets, show_cgan_samples

__all__ = [
    "prepare_ds",
    "build_gan_ds",
    "build_generator",
    "build_discriminator",
    "train_cgan",
    "save_fake_images",
    "make_noisy_clone",
    "corrupt_labels",
    "eval_folder_vs_real",
    "compute_fid_kid_from_datasets",
    "real_ds_to_images_only",
    "folder_to_ds",
    "build_effnet_classifier",
    "make_mixed_train_ds",
    "train_and_evaluate",
    "run_ratio_experiments",
    "DEFAULT_RATIOS",
    "DATASETS",
    "KAGGLE_URLS",
    "run_pipeline_one",
    "run_all_datasets",
    "show_cgan_samples",
]
