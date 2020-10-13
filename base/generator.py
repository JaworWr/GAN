import torch

from base.model import Model


class BaseGenerator(Model):
    input_size: int

    def __init__(self, config):
        super().__init__()
    
    def generate_batch(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        noise = torch.randn((n, self.input_size), device=device, requires_grad=False)
        return self(noise)
