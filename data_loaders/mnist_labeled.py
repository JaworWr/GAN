from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from base.data_loader import BaseDataLoaderFactory
from utils.data import ShuffleLoopDataset


class MnistDataLoaderFactory(BaseDataLoaderFactory):
    @classmethod
    def get_data_loader(cls, config) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        mnist_dataset = MNIST(
            config.data.root,
            train=True,
            transform=transform,
            download=config.data.get("download", True)
        )
        dataset = ShuffleLoopDataset(mnist_dataset, ignore_labels=False)
        dl = DataLoader(
            dataset,
            batch_size=config.data.get("batch_size", 1),
            **config.data.loader_kwargs,
        )
        return dl
