import torch
from torch import nn

from base.discriminator_labeled import BaseDiscriminatorLabeled


class ConvDiscriminatorV3(BaseDiscriminatorLabeled):
    n_labels = 10

    def __init__(self, config):
        super().__init__(config)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # 32 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 12 x 12
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5),  # 64 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 8 * 8, self.n_labels + 1),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
