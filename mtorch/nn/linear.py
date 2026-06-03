import numpy as np

from mtorch.nn.base import Module
from mtorch.tensor import Tensor


class Linear(Module):

    def __init__(self, nin, nout):
        super().__init__()

        bound = 1 / np.sqrt(nin)

        self.W = Tensor(
            np.random.uniform(-bound, bound, (nin, nout)), requires_grad=True
        )
        self.B = Tensor(np.zeros((1, nout)), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B]


class Dropout(Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if self.training and self.p > 0:
            mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            mask_np = Tensor(mask, requires_grad=False)
            return x * mask_np
        return x
