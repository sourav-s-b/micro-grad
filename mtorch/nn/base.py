class Module:

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
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
