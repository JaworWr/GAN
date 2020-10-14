import os
import shutil
from datetime import datetime

from torch.utils.data import DataLoader

from base.discriminator import BaseDiscriminator
from base.generator import BaseGenerator


class BaseTrainer:
    DISCRIMINATOR_CHECKPOINT_NAME = "discriminator{}.p"
    GENERATOR_CHECKPOINT_NAME = "generator{}.p"

    def __init__(self, config, data: DataLoader, discriminator: BaseDiscriminator, generator: BaseGenerator, **kwargs):
        self.config = config
        self.data = data
        self.discriminator = discriminator
        self.generator = generator
        self.device = config.pop("device", "cpu")
        self.discriminator.to(self.device)
        self.generator.to(self.device)
        self.experiment_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.checkpoint_path = os.path.join(os.path.join("model_checkpoints", self.experiment_name))
        self.n_checkpoints = 0
        if config.trainer.checkpoint_steps:
            os.makedirs(self.checkpoint_path)

    def train(self):
        raise NotImplementedError

    def make_checkpoint(self):
        disc_fname = os.path.join(self.checkpoint_path, self.DISCRIMINATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
        self.discriminator.save(disc_fname)
        gen_fname = os.path.join(self.checkpoint_path, self.GENERATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
        self.generator.save(gen_fname)
        self.n_checkpoints += 1
