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
    name="MNIST (Keras builtin)",
    slug="",                       # kagglehub 안 씀
    dataset_format="mnist_keras",   # ✅ 여기만 바꿔
  ),
}

# KAGGLE_URLS: slug 없는 애는 빼거나 안전하게 처리
KAGGLE_URLS = {
    k: f"https://www.kaggle.com/datasets/{v['slug']}"
    for k, v in DATASETS.items()
    if v.get("slug")  # ✅ slug 비어있으면 제외
}
