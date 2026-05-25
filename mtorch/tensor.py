# mtorch/tensor.py
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="") -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(self.data.T.copy(), (self,), "transpose")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Tensor(self.data**power, (self,), f"**{power}")

        def _backward():
            self_grad = (power * self.data ** (power - 1)) * out.grad
            self.grad += self._match_shape(self_grad, self.shape)

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data + self.data, (self, other), "+")

        def _backward():
            self.grad += self._match_shape(out.grad, self.shape)
            other.grad += self._match_shape(out.grad, other.shape)

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data * self.data, (self, other), "*")

        def _backward():
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad

            self.grad += self._match_shape(self_grad, self.shape)
            other.grad += self._match_shape(other_grad, other.shape)

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) + -self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(self.shape)
                for ax in axes:
                    shape[ax] = 1
                g = g.reshape(shape)
            self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    def relu(self):
        """Element-wise Rectified Linear Unit activation function"""
        out = Tensor(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            # Mask gradients to zero where the original forward activation was negative
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def _match_shape(self, grad, target_shape):
        if grad.shape == target_shape:
            return grad

        grad_ndim = grad.ndim
        target_ndim = len(target_shape)

        if grad_ndim > target_ndim:
            axes_to_sum = tuple(range(grad_ndim - target_ndim))
            grad = grad.sum(axis=axes_to_sum)

        axes_to_sum = tuple(
            i
            for i, (g, t) in enumerate(zip(grad.shape, target_shape))
            if g != t and t == 1
        )
        if axes_to_sum:
            grad = grad.sum(axis=axes_to_sum, keepdims=True)

        return grad.reshape(target_shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data=\n{self.data})"
