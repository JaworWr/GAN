from torch import nn

from base.discriminator import BaseDiscriminator


class ConvDiscriminator(BaseDiscriminator):

    def __init__(self, config):
        super().__init__(config)
        p_dropout = config.discriminator.get("dropout", 0)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5),  # 64 x 24 x 24
            # nn.ReLU(),
            # nn.Dropout(p_dropout),
            # nn.Conv2d(64, 64, 5),  # 64 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 12 x 12
            nn.Dropout(p_dropout),
            nn.Conv2d(64, 128, 5),  # 128 x 8 x 8
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
