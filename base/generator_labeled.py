from typing import Any

import torch

from base.generator import BaseGenerator


class BaseGeneratorLabeled(BaseGenerator):
    n_labels: int

    def generate_batch_with_labels(self, labels: torch.Tensor, device: torch.device = torch.device("cpu")) -> Any:
        noise = torch.randn((labels.shape[0], self.input_size), device=device, requires_grad=False)
        return self(noise, labels)

    def generate_batch(self, n: int, device: torch.device = torch.device("cpu")) -> Any:
        labels = torch.randint(self.n_labels, (n,), device=device)
        return (self.generate_batch_with_labels(labels, device), labels)
