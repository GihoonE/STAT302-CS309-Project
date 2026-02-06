# models.py
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Baseline cGAN Generator:
      input: z (noise) + y (one-hot label)
      output: 1x28x28 image in [-1, 1] using tanh
    """
    def __init__(self, z_dim=100, n_classes=10, img_dim=28*28, hidden=256):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.img_dim = img_dim

        in_dim = z_dim + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(True),
            nn.Linear(hidden * 2, img_dim),
            nn.Tanh()
        )

    def forward(self, z, y_onehot):
        x = torch.cat([z, y_onehot], dim=1)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Baseline cGAN Discriminator:
      input: image + y (one-hot label)
      output: probability real/fake in [0,1] using sigmoid
    """
    def __init__(self, n_classes=10, img_dim=28*28, hidden=256):
        super().__init__()
        self.n_classes = n_classes
        self.img_dim = img_dim

        in_dim = img_dim + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x_img, y_onehot):
        x = x_img.view(x_img.size(0), -1)
        x = torch.cat([x, y_onehot], dim=1)
        return self.net(x)
