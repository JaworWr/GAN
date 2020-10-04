import torch
from torch import nn


class Model(nn.Module):
    def save(self, path: str):
        d = self.state_dict()
        with open(path, "wb") as f:
            torch.save(d, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = torch.load(f)
        self.load_state_dict(d)
