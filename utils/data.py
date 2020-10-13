from torch.utils import data


class ShuffleLoopDataset(data.IterableDataset):
    # Single process for now.
    def __init__(self, dataset: data.Dataset, ignore_labels=True):
        self.dataset = dataset
        self.samples = None
        self.ignore_labels = ignore_labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples is None:
            self.samples = iter(data.RandomSampler(self.dataset))

        try:
            idx = next(self.samples)
            sample = self.dataset[idx]
            return sample[0] if self.ignore_labels else sample
        except StopIteration:
            self.samples = None
            return next(self)
