# mtorch/nn/modules.py
import numpy as np

from mtorch.tensor import Tensor


class Module:

    def __init__(self):
        self.training = True

    def train(self, mode = True):
        self.training = mode
        for attr in self.__dict__.values():
            if isinstance(attr,Module):
                attr.train(mode)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item , Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
        return params


class Linear(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__()

        bound = 1.0 / np.sqrt(nin)
        self.W = Tensor(np.random.uniform(-bound, bound, (nin, nout)),requires_grad=True)
        self.B = Tensor(np.zeros((1,nout)),requires_grad=True)

    def __call__(self , x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B]

class Dropout(Module):

    def __init__(self,p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if self.training and self.p>0:
            mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            return x * Tensor(mask,requires_grad=False)
        return x

class Sequential(Module):

    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

class ReLU(Module):
    def __call__(self,x):
        return x.relu()


class Tanh(Module):
    def __call__(self,x):
        return x.tanh()


class Sigmoid(Module):
    def __call__(self,x):
        return x.sigmoid()
