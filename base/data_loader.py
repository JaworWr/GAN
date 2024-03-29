from torch.utils.data import DataLoader


class BaseDataLoaderFactory:
    @classmethod
    def get_data_loader(cls, config) -> DataLoader:
        raise NotImplementedError
