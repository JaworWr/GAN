import os

from torch.utils.data import DataLoader

from base.discriminator import BaseDiscriminator
from base.generator import BaseGenerator


class BaseTrainer:
    DISCRIMINATOR_CHECKPOINT_NAME = "discriminator{}.p"
    GENERATOR_CHECKPOINT_NAME = "generator{}.p"

    def __init__(self, config, data: DataLoader, discriminator: BaseDiscriminator, generator: BaseGenerator,
                 checkpoint_path: str, **kwargs):
        self.config = config
        self.data = data
        self.discriminator = discriminator
        self.generator = generator
        self.device = config.pop("device", "cpu")
        self.discriminator.to(self.device)
        self.generator.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.n_checkpoints = 0

    def train(self):
        raise NotImplementedError

    def make_checkpoint(self):
        if self.checkpoint_path is not None:
            disc_fname = os.path.join(self.checkpoint_path,
                                      self.DISCRIMINATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
            self.discriminator.save(disc_fname)
            gen_fname = os.path.join(self.checkpoint_path, self.GENERATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
            self.generator.save(gen_fname)
            self.n_checkpoints += 1
