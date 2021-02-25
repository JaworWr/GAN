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
        self.disc_optimizer = torch.optim.Adam(discriminator.parameters(), config.trainer.discriminator.lr)
        self.gen_optimizer = torch.optim.Adam(generator.parameters(), config.trainer.generator.lr)
        self.disc_save_path = kwargs.get("disc_save_path")
        self.gen_save_path = kwargs.get("gen_save_path")
        log_path = kwargs.get("log_path")
        if log_path is not None:
            self.summary_writer = SummaryWriter(log_dir=log_path)
        else:
            self.summary_writer = None

    discriminator_loss = nn.BCELoss()

    @classmethod
    def generator_loss(cls, pred):
        return cls.discriminator_loss(pred, torch.ones_like(pred))

    def train(self):
        ds = iter(self.data)
        for step in tqdm(range(self.config.trainer.steps)):
            self.discriminator.train()
            self.generator.eval()
            bs = self.config.data.batch_size
            step_losses = []
            for iter_ in range(self.config.trainer.discriminator.get("training_batches", 1)):
                self.disc_optimizer.zero_grad()
                X_true = next(ds).to(self.device)
                with torch.no_grad():
                    X_fake = self.generator.generate_batch(bs, self.device)
                X, y = interleave([X_true, X_fake],
                                  [torch.ones((bs, 1), device=self.device), torch.zeros((bs, 1), device=self.device)])
                pred = self.discriminator(X)
                loss = self.discriminator_loss(pred, y)
                loss.backward()
                step_losses.append(loss.item())
                self.disc_optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar("discriminator_loss", torch.tensor(step_losses).mean(), step)

            self.discriminator.eval()
            self.generator.train()
            step_losses = []
            for iter_ in range(self.config.trainer.generator.get("training_batches", 1)):
                self.gen_optimizer.zero_grad()
                X = self.generator.generate_batch(bs, self.device)
                pred = self.discriminator(X)
                loss = self.generator_loss(pred)
                loss.backward()
                step_losses.append(loss.item())
                self.gen_optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar("generator_loss", torch.tensor(step_losses).mean(), step)

            if self.config.trainer.checkpoint_steps and step % self.config.trainer.checkpoint_steps == 0:
                self.make_checkpoint()
        if self.config.trainer.checkpoint_steps:
            self.make_checkpoint()
