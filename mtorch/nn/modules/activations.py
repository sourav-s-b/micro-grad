from mtorch.nn import Module


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()
