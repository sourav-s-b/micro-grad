import numpy as np
import queue
import threading
import math

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

        batch = [self.dataset[i] for i in batch_idx]

        X_batch = np.stack([item[0] for item in batch])
        Y_batch = np.stack([item[1] for item in batch])

        return Tensor(X_batch, requires_grad=False), Tensor(
            Y_batch, requires_grad=False
        )

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class PrefetchedDataLoader:

    def __init__(self, dataloader, max_prefetch=3) -> None:
        self.dataloader = dataloader
        self.max_prefetch = max_prefetch

    def __iter__(self):

        q = queue.Queue(maxsize=self.max_prefetch)
        xp = Device.xp

        def worker():
            for X, Y in self.dataloader:

                if hasattr(X, "data"):
                    X = X.data
                if hasattr(Y, "data"):
                    Y = Y.data

                X_np = np.stack(X) if isinstance(X, list) else np.asarray(X)
                Y_np = np.stack(Y) if isinstance(Y, list) else np.asarray(Y)

                X_gpu = xp.array(X_np.astype(np.int32))
                Y_gpu = xp.array(Y_np.astype(np.int32))

                q.put((X_gpu, Y_gpu))

            q.put(None)

        t = threading.Thread(target=worker, daemon=None)

        t.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

    def __len__(self):
        return len(self.dataloader)
