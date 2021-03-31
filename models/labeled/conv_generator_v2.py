import torch
from torch import nn

from base.generator import BaseGenerator
from base.generator_labeled import BaseGeneratorLabeled


class ConvGeneratorV2(BaseGeneratorLabeled):
    input_size = 100
    n_labels = 10

    def __init__(self, config):
        super().__init__(config)

        self.tokens = nn.Embedding(self.n_labels, self.input_size)
        self.linear = nn.Linear(self.input_size, 256 * 7 * 7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, y):
        y = self.tokens(y)
        x = x + y
        x = self.linear(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv(x)
        assert x.shape[1:] == (1, 28, 28), f"Wrong output shape: {x.shape}"
        return x
