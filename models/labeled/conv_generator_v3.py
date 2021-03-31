import torch
from torch import nn

from base.generator import BaseGenerator
from base.generator_labeled import BaseGeneratorLabeled


class ConvGeneratorV3(BaseGeneratorLabeled):
    input_size = 100
    token_size = 20
    n_labels = 10

    def __init__(self, config):
        super().__init__(config)

        self.tokens = nn.Embedding(self.n_labels, self.token_size)
        self.linear = nn.Sequential(
            nn.Linear(self.input_size + self.token_size, 128 * 7 * 7, bias=False),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, y):
        y = self.tokens(y)
        x = torch.cat([x, y], 1)
        x = self.linear(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv(x)
        assert x.shape[1:] == (1, 28, 28), f"Wrong output shape: {x.shape}"
        return x
