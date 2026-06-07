from mtorch.config import Device
from mtorch.nn.base import Module
from mtorch.tensor import Tensor


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

        idx_array = indices.data.astype(int)

        out_data = self.w.data[idx_array]

        out = Tensor(
            out_data, (self.w, indices), "Embedding", requires_grad=self.w.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            if out.requires_grad is None or not self.w.requires_grad:
                return

            dW = Device.xp.zeros_like(self.w.data)

            flat_idx = idx_array.reshape(-1)
            flat_grad = out.grad.reshape(-1, self.embedding_dim)

            if Device.device == "cuda":
                from cupyx import scatter_add

                scatter_add(dW, flat_idx, flat_grad)
            else:
                Device.xp.add.at(dW, flat_idx, flat_grad)
            self.w._accumulate_grad(dW)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.w]


class LayerNorm(Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(Device.xp.ones(dim), requires_grad=True)
        self.beta = Tensor(Device.xp.zeros(dim), requires_grad=True)

    def __call__(self, x):
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)

        std_inv = 1.0 / Device.xp.sqrt(var + self.eps)
        x_norm = (x.data - mean) * std_inv
        out_data = self.gamma.data * x_norm + self.beta.data

        requires_grad = (
            x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        )
        out = Tensor(
            out_data,
            (x, self.gamma, self.beta),
            "LayerNorm",
            requires_grad=requires_grad,
        )

        if requires_grad:

            def _backward():
                if out.grad is None:
                    return

                if self.gamma.requires_grad:
                    self.gamma._accumulate_grad((out.grad * x_norm).sum(axis=(0, 1)))
                if self.beta.requires_grad:
                    self.beta._accumulate_grad((out.grad.sum(axis=(0, 1))))
                if x.requires_grad:

                    g = out.grad * self.gamma.data

                    dx = std_inv * (
                        g
                        - (g.mean(axis=-1, keepdims=True))
                        - x_norm * (g * x_norm).mean(axis=-1, keepdims=True)
                    )
                    x._accumulate_grad(dx)

            out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class RMSNorm(Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = Tensor(Device.xp.ones(dim))

    def __call__(self, x):
        xp = Device.xp

        x_data = x.data
        w_data = self.weight.data

        rms = xp.sqrt(xp.mean(x_data**2, axis=-1, keepdims=True) + self.eps)
        normed_x = x_data / rms
        out_data = normed_x * w_data

        out = Tensor(
            out_data, (x, self.weight), "RMSNorm", requires_grad=x.requires_grad
        )

        if x.requires_grad:

            def _backward():

                if out.grad is None:
                    return

                dW = xp.sum(out.grad * normed_x, axis=tuple(range(x_data.ndim - 1)))
                self.weight._accumulate_grad(dW)

                dx_normed = out.grad * w_data
                d_rms = xp.sum(
                    dx_normed * x_data * (-1.0 / (rms**2)), axis=-1, keepdims=True
                )
                dX = (dx_normed / rms) + (d_rms * x_data / x_data.shape[-1])

                x._accumulate_grad(dX)

            out._backward = _backward

        return out

    def parameters(self):
        return [self.weight]
