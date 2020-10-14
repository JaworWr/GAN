import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def save(self, path: str):
        d = self.state_dict()
        with open(path, "wb") as f:
            torch.save(d, f)

    def load(self, path: str, **kwargs):
        with open(path, "rb") as f:
            d = torch.load(f, **kwargs)
        self.load_state_dict(d)
