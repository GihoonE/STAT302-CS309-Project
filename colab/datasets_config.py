DATASETS = {
  "sports_ball": dict(
    name="Sports Ball",
    slug="samuelcortinhas/sports-balls-multiclass-image-classification",
    dataset_format="imagefolder_supervised_train_test",
    train_subdir="train",
    test_subdir="test",
  ),
  "animals": dict(
    name="Cats-and-Dogs-Breed",
    slug="antobenedetti/animals",
    dataset_format="imagefolder_train_labeled_test_unlabeled",
    train_subdir="animals/train",
    val_subdir="animals/val",
    test_subdir="animals/inf",
  ),
  "mnist": dict(
    name="MNIST",
    slug="arnavsharma45/mnist-dataset",
    dataset_format="mnist_csv",
    # subdir 없음
  ),
}
