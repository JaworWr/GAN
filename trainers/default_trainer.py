import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from base.discriminator import BaseDiscriminator
from base.generator import BaseGenerator
from base.trainer import BaseTrainer
from utils.data import interleave


class DefaultTrainer(BaseTrainer):

    def __init__(self, config, data: DataLoader, discriminator: BaseDiscriminator, generator: BaseGenerator,
                 checkpoint_path: str, **kwargs):
        super().__init__(config, data, discriminator, generator, checkpoint_path, **kwargs)
        self._disc_optimizer = torch.optim.Adam(discriminator.parameters(), config.trainer.discriminator.lr)
        self._gen_optimizer = torch.optim.Adam(generator.parameters(), config.trainer.generator.lr)
        log_path = kwargs.get("log_path")
        if log_path is not None:
            self.summary_writer = SummaryWriter(log_dir=log_path)
        else:
            self.summary_writer = None

    _disc_loss = nn.BCELoss()

    @classmethod
    def discriminator_loss(cls, pred, target):
        return cls._disc_loss(pred, target)

    @classmethod
    def generator_loss(cls, pred):
        return cls.discriminator_loss(pred, torch.ones_like(pred))

    @property
    def discriminator_optimizer(self) -> torch.optim.Optimizer:
        return self._disc_optimizer

    @property
    def generator_optimizer(self) -> torch.optim.Optimizer:
        return self._gen_optimizer

    def discriminator_step(self, data_iter, step):
        self.discriminator.train()
        self.generator.eval()
        bs = self.config.data.batch_size
        c = self.config.trainer.get("label_smoothing", 1)
        step_losses = []
        for iter_ in range(self.config.trainer.discriminator.get("training_batches", 1)):
            X_true = next(data_iter).to(self.device)
            with torch.no_grad():
                X_fake = self.generator.generate_batch(bs, self.device)
            X, y = interleave([X_true, X_fake],
                              [c * torch.ones((bs, 1), device=self.device), torch.zeros((bs, 1), device=self.device)])
            pred = self.discriminator(X)
            loss = self.discriminator_loss(pred, y)
            loss.backward()
            step_losses.append(loss.item())
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("discriminator_loss", torch.tensor(step_losses).mean(), step)

    def generator_step(self, data_iter, step):
        self.discriminator.eval()
        self.generator.train()
        bs = self.config.data.batch_size
        step_losses = []
        for iter_ in range(self.config.trainer.generator.get("training_batches", 1)):
            X = self.generator.generate_batch(bs, self.device)
            pred = self.discriminator(X)
            loss = self.generator_loss(pred)
            loss.backward()
            step_losses.append(loss.item())
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("generator_loss", torch.tensor(step_losses).mean(), step)
