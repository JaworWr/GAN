from typing import Any

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torchvision.models.inception import inception_v3


def frechet_gaussian_distance(X1: np.ndarray, X2: np.ndarray,
                              eps: float = 1e-6) -> float:
    """
    Frechet distance between the distributions of X1 and X2, calculated in numpy.
    """

    mu1 = np.mean(X1, 0)
    mu2 = np.mean(X2, 0)
    sigma1 = np.cov(X1)
    sigma2 = np.cov(X2)

    mu_norm = np.sum((mu1 - mu2) ** 2)
    sigma_sqrt = linalg.sqrtm(sigma1 @ sigma2)

    if not np.isfinite(sigma_sqrt).all():
        offset = np.eye(sigma1.shape[0]) * eps
        sigma_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # numerical errors may cause an imaginary component to appear
    sigma_sqrt = np.real(sigma_sqrt)
    sigma_trace = np.trace(sigma1 + sigma2 - 2 * sigma_sqrt)
    return mu_norm + sigma_trace


def cov_matrix(X: Any) -> Any:
    n = X.shape[0]
    return X @ X.T / (n - 1)


class FrechetInceptionDistance:
    def __init__(self, device: torch.device):
        inception_model = inception_v3(pretrained=True)
        # remove the output layer
        inception_model.fc = nn.Identity()
        self.model = inception_model
        self.model.eval()
        self.model.to(device)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

        self.upsample = nn.Upsample(size=(299, 299), mode="bicubic", align_corners=True)
        self.upsample.to(device=device)
        self.device = device

        self.X_true = []
        self.X_fake = []

    def preprocess(self, X: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing of inputs for the inception_v3 model
        """
        # conversion back to [0, 1]
        X = (X + 1) * 0.5
        # a silly conversion from greyscale to RGB
        X = X.repeat(1, 3, 1, 1)
        # normalization
        X = X - self.mean[None, :, None, None]
        X = X / self.std[None, :, None, None]
        return X

    def inception_vector(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device)
        X = self.preprocess(X)
        X = self.upsample(X)
        X = self.model(X)
        return X

    def add_batch(self, X_true: torch.Tensor, X_fake: torch.Tensor):
        with torch.no_grad():
            X_true = self.inception_vector(X_true)
            X_fake = self.inception_vector(X_fake)
            self.X_true.append(X_true.cpu().numpy())
            self.X_fake.append(X_fake.cpu().numpy())

    def calculate(self) -> float:
        X_true = np.concatenate(self.X_true, 0)
        X_fake = np.concatenate(self.X_fake, 0)
        self.X_true = []
        self.X_fake = []

        return frechet_gaussian_distance(X_true, X_fake)
