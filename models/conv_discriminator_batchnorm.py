from torch import nn

from base.discriminator import BaseDiscriminator


class ConvDiscriminatorBatchnorm(BaseDiscriminator):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5),  # 64 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 12 x 12
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5),  # 128 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
