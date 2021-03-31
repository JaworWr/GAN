import torch
from torch import nn

from base.discriminator_labeled import BaseDiscriminatorLabeled


class ConvDiscriminatorV5(BaseDiscriminatorLabeled):
    token_size = 40
    n_labels = 10

    def __init__(self, config):
        super().__init__(config)

        self.tokens = nn.Embedding(self.n_labels, self.token_size)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # 32 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 12 x 12
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5),  # 64 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 5),  # 256 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(256 * 4 * 4 + self.token_size, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        y = self.tokens(y)
        x = self.conv(x)
        x = torch.cat([x, y], 1)
        x = self.linear(x)
        return x
