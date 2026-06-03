import numpy as np

from mtorch.tensor import Tensor
from mtorch.config import Device, to_cpu


class Dataset:

    def __init__(self) -> None:
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class DataLoader:

    def __init__(self, dataset, batch_size=32, shuffle=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration

        batch_idx = self.indices[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size

        samples = [self.dataset[s] for s in batch_idx]
        batch_x = np.array([to_cpu(s[0]) for s in samples])
        batch_y = np.array([to_cpu(s[1]) for s in samples])

        return Tensor(batch_x, requires_grad=False), Tensor(
            batch_y, requires_grad=False
        )
