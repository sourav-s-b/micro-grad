# mtorch/nn/modules.py
import numpy as np

from mtorch.tensor import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__()

        bound = 1.0 / np.sqrt(nin)
        self.W = Tensor(np.random.uniform(-bound, bound, (nin, nout)))
        self.B = Tensor(np.zeros((1,nout)))

    def __call__(self , x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B]
