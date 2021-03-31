import torch
from torch import nn

from base.discriminator_labeled import BaseDiscriminatorLabeled


class ConvDiscriminatorV4(BaseDiscriminatorLabeled):
    token_size = 20
    n_labels = 10

    def __init__(self, config):
        super().__init__(config)

        self.tokens = nn.Embedding(self.n_labels, self.token_size)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # 16 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 x 12 x 12
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5),  # 32 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 8 * 8 + self.token_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        y = self.tokens(y)
        x = self.conv(x)
        x = torch.cat([x, y], 1)
        x = self.linear(x)
        return x
