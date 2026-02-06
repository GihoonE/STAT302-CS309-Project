# mnist_mirror.py
from torchvision.datasets import MNIST


class MNISTMirror(MNIST):
    """
    MNIST dataset with a stable HTTPS mirror.
    Fixes SSL and 404 issues on macOS.
    """
    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/"
    ]
