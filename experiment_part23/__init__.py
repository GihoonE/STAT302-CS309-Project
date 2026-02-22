# experiment_part23/__init__.py
"""
Lightweight package init.

Do NOT import heavy or optional modules here.
Import pipeline components explicitly where needed.
"""

from .datasets_config import DATASETS, KAGGLE_URLS
from .data_prep import prepare_ds

__all__ = [
    "DATASETS",
    "KAGGLE_URLS",
    "prepare_ds",
]