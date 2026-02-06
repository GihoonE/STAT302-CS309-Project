# cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28x28
        self.pool = nn.MaxPool2d(2, 2)               # 14x14

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14x14
        self.pool2 = nn.MaxPool2d(2, 2)               # 7x7

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))   # -> 64x14x14
        x = self.pool2(F.relu(self.conv3(x)))  # -> 128x7x7
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
