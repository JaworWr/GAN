import torch

from trainers.default_trainer import DefaultTrainer
from utils.data import interleave


class DiscriminatorNoiseTrainer(DefaultTrainer):
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
            X = self.generator.generate_batch(bs, self.device)
            X = X + torch.randn(X.shape, device=self.device) * self.config.trainer.noise_sigma
            pred = self.discriminator(X)
            loss = self.generator_loss(pred)
            loss.backward()
            step_losses.append(loss.item())
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("generator_loss", torch.tensor(step_losses).mean(), step)
