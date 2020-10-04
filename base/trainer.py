from torch.utils.data import DataLoader

from base.discriminator import BaseDiscriminator
from base.generator import BaseGenerator


class Trainer:
    def __init__(self, config, data: DataLoader, discriminator: BaseDiscriminator, generator: BaseGenerator):
        self.config = config
        self.data = data
        self.discriminator = discriminator
        self.generator = generator

    def train(self):
        raise NotImplementedError
