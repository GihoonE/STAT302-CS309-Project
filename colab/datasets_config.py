"""
Kaggle dataset configs for the experiment pipeline.
Each entry: slug (for kagglehub), display name, optional train/test subdir names.
Pipeline expects: <downloaded_dir>/train_subdir/<class>/*.jpg and test_subdir/<class>/*.jpg
"""

DATASETS = {
    "sports_ball": {
        "slug": "samuelcortinhas/sports-balls-multiclass-image-classification",
        "name": "Sports Ball",
        "train_subdir": "train",
        "test_subdir": "test",
    },
    "mnist": {
        "slug": "arnavsharma45/mnist-dataset",
        "name": "MNIST",
        # Common MNIST Kaggle layouts: "Train"/"Test" or "train"/"test" with 0..9 subdirs
        "train_subdir": "Train",
        "test_subdir": "Test",
    },
    "animals": {
        "slug": "antobenedetti/animals",
        "name": "Animals",
        "train_subdir": "train",
        "test_subdir": "test",
    },
}

# Kaggle URLs for reference
KAGGLE_URLS = {
    "sports_ball": "https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification",
    "mnist": "https://www.kaggle.com/datasets/arnavsharma45/mnist-dataset",
    "animals": "https://www.kaggle.com/datasets/antobenedetti/animals",
}
