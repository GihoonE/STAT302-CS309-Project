"""
다중 데이터셋 로딩 테스트 스크립트
MNIST
SportsBall
Cats-and-Dogs-Breed
"""

import os
import argparse
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import struct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# MNIST
# =========================
def test_mnist_loading(data_dir, batch_size=4, num_samples=5):
    print("\n" + "=" * 50)
    print("MNIST 데이터셋 로딩 테스트")
    print("=" * 50)

    if not os.path.exists(data_dir):
        print(f"오류: 디렉토리가 존재하지 않습니다: {data_dir}")
        return False, None

    img_file = os.path.join(data_dir, "train-images.idx3-ubyte")
    lbl_file = os.path.join(data_dir, "train-labels.idx1-ubyte")

    if not (os.path.exists(img_file) and os.path.exists(lbl_file)):
        print("오류: MNIST IDX 파일이 없습니다.")
        return False, None

    try:
        with open(img_file, "rb") as f:
            magic, n, r, c = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

        with open(lbl_file, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        class MNISTDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                lbl = int(self.labels[idx])
                if self.transform:
                    img = self.transform(img)
                return img, lbl

        dataset = MNISTDataset(images, labels, transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        samples = []
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            samples.append(batch)

        return True, samples

    except Exception as e:
        print(f"MNIST 로딩 오류: {e}")
        return False, None


# =========================
# SportsBall
# =========================
def test_sportsball_loading(data_dir, batch_size=4, num_samples=5):
    print("\n" + "=" * 50)
    print("SportsBall 데이터셋 로딩 테스트")
    print("=" * 50)

    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print("오류: train 디렉토리가 없습니다.")
        return False, None

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        samples = []
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            samples.append(batch)

        return True, samples

    except Exception as e:
        print(f"SportsBall 로딩 오류: {e}")
        return False, None


# =========================
# Cats-and-Dogs-Breed
# =========================
def test_cats_and_dogs_loading(data_dir, batch_size=4, num_samples=5):
    print("\n" + "=" * 50)
    print("Cats-and-Dogs-Breed 데이터셋 로딩 테스트")
    print("=" * 50)

    train_dir = os.path.join(data_dir, "TRAIN")
    if not os.path.exists(train_dir):
        print("오류: TRAIN 디렉토리가 없습니다.")
        return False, None

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        samples = []
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            samples.append(batch)

        return True, samples

    except Exception as e:
        print(f"Cats&Dogs 로딩 오류: {e}")
        return False, None


# =========================
# Visualization
# =========================
def visualize_samples(sample_batches, dataset_type):
    if not sample_batches:
        return

    def denorm_mnist(x):
        return x * 0.3081 + 0.1307

    def denorm_rgb(x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return x * std + mean

    images, labels = sample_batches[0]
    fig, axes = plt.subplots(1, min(8, len(images)), figsize=(12, 3))

    for i, ax in enumerate(axes):
        img = images[i]
        if dataset_type == "mnist":
            img = denorm_mnist(img)[0]
            ax.imshow(img, cmap="gray")
        else:
            img = denorm_rgb(img).permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img)
        ax.set_title(f"{labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{dataset_type}_samples.png")
    plt.close()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", required=True,
                        choices=["mnist", "sportsball", "cats-and-dogs"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()

    if args.dataset_type == "mnist":
        ok, samples = test_mnist_loading(args.data_dir, args.batch_size, args.num_samples)
    elif args.dataset_type == "sportsball":
        ok, samples = test_sportsball_loading(args.data_dir, args.batch_size, args.num_samples)
    else:
        ok, samples = test_cats_and_dogs_loading(args.data_dir, args.batch_size, args.num_samples)

    if ok:
        visualize_samples(samples, args.dataset_type)
        print(f"{args.dataset_type} 로딩 성공")
    else:
        print(f"{args.dataset_type} 로딩 실패")


if __name__ == "__main__":
    main()
