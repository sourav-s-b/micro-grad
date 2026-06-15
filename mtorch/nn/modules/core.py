from mtorch.config import Device
from mtorch.nn import Module
from mtorch.tensor_compiled import Tensor
import mtorch.nn.functional.core as F


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.w = Tensor(
            Device.xp.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True,
        )

    def __call__(self, indices):
        return F.embedding(indices, self.w, self.embedding_dim)

    def parameters(self):
        return [self.w]


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(Device.xp.ones(dim), requires_grad=True)
        self.beta = Tensor(Device.xp.zeros(dim), requires_grad=True)

    def __call__(self, x):
        return F.layernorm(x, self.gamma, self.beta, eps=self.eps)

    def parameters(self):
        return [self.gamma, self.beta]


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(Device.xp.ones(dim), requires_grad=True)

    def __call__(self, x):
        return F.rmsnorm(x, self.weight, eps=self.eps)

    def parameters(self):
        return [self.weight]
