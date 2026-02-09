"""
Kaggle dataset configs for the experiment pipeline.
Each entry includes an explicit dataset_format so loading does not rely on guessed layouts.
"""

DATASETS = {
    "sports_ball": {
        "slug": "samuelcortinhas/sports-balls-multiclass-image-classification",
        "name": "Sports Ball",
        "dataset_format": "imagefolder_supervised_train_test",
        "train_subdir": "train",
        "test_subdir": "test",
    },
    "mnist": {
        "slug": "arnavsharma45/mnist-dataset",
        "name": "MNIST",
        "dataset_format": "mnist_idx",
    },
    "animals": {
        "slug": "antobenedetti/animals",
        "name": "Cats-and-Dogs-Breed",
        "dataset_format": "imagefolder_train_labeled_test_unlabeled",
        "train_subdir": "TRAIN",
        "test_subdir": "TEST",
    },
}

KAGGLE_URLS = {
    "sports_ball": "https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification",
    "mnist": "https://www.kaggle.com/datasets/arnavsharma45/mnist-dataset",
    "animals": "https://www.kaggle.com/datasets/antobenedetti/animals",
}
