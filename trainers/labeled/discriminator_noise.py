import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from base.discriminator_labeled import BaseDiscriminatorLabeled
from base.generator_labeled import BaseGeneratorLabeled
from base.trainer import BaseTrainer
from utils.data import interleave


class DiscriminatorNoiseTrainer(BaseTrainer):
    def __init__(self, config, data: DataLoader, discriminator: BaseDiscriminatorLabeled,
                 generator: BaseGeneratorLabeled, checkpoint_path: str, **kwargs):
        super().__init__(config, data, discriminator, generator, checkpoint_path, **kwargs)
        self._disc_optimizer = torch.optim.Adam(discriminator.parameters(),
                                                **config.trainer.discriminator.optimizer_args)
        self._gen_optimizer = torch.optim.Adam(generator.parameters(), **config.trainer.generator.optimizer_args)
        log_path = kwargs.get("log_path")
        if log_path is not None:
            self.summary_writer = SummaryWriter(log_dir=log_path)
        else:
            self.summary_writer = None

    _disc_loss = torch.nn.NLLLoss()

    @classmethod
    def discriminator_loss(cls, pred, target):
        return cls._disc_loss(pred, target)

    @classmethod
    def generator_loss(cls, pred, target):
        return cls.discriminator_loss(pred, target + 1)

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
        step_losses = []
        for iter_ in range(self.config.trainer.discriminator.get("training_batches", 1)):
            X_true, y_true = next(data_iter)
            X_true = X_true.to(self.device)
            y_true = y_true.to(self.device)
            with torch.no_grad():
                X_fake, y_fake = self.generator.generate_batch(bs, self.device)
            X, y = interleave([X_true, X_fake], [y_true + 1, torch.zeros_like(y_fake)])
            X = X + torch.randn(X.shape, device=self.device) * self.config.trainer.noise_sigma
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
            X, y = self.generator.generate_batch(bs, self.device)
            X = X + torch.randn(X.shape, device=self.device) * self.config.trainer.noise_sigma
            pred = self.discriminator(X)
            loss = self.generator_loss(pred, y)
            loss.backward()
            step_losses.append(loss.item())
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("generator_loss", torch.tensor(step_losses).mean(), step)
