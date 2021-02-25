import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    @classmethod
    def discriminator_loss(cls, pred, target):
        raise NotImplementedError

    @classmethod
    def generator_loss(cls, pred):
        raise NotImplementedError

    @property
    def discriminator_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @property
    def generator_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def discriminator_step(self, data_iter, step):
        raise NotImplementedError

    def generator_step(self, data_iter, step):
        raise NotImplementedError

    def training_step(self, data_iter, step):
        self.discriminator_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.discriminator_step(data_iter, step)
        self.discriminator_optimizer.step()
        self.generator_step(data_iter, step)
        self.generator_optimizer.step()

        if self.config.trainer.checkpoint_steps and step % self.config.trainer.checkpoint_steps == 0:
            self.make_checkpoint()

    def train(self):
        data_iter = iter(self.data)
        for step in tqdm(range(self.config.trainer.steps)):
            self.training_step(data_iter, step)
        if self.config.trainer.checkpoint_steps:
            self.make_checkpoint()

    def make_checkpoint(self):
        if self.checkpoint_path is not None:
            disc_fname = os.path.join(self.checkpoint_path,
                                      self.DISCRIMINATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
            self.discriminator.save(disc_fname)
            gen_fname = os.path.join(self.checkpoint_path, self.GENERATOR_CHECKPOINT_NAME.format(self.n_checkpoints))
            self.generator.save(gen_fname)
            self.n_checkpoints += 1
