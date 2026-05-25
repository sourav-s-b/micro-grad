# mtorch/tensor.py
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True) -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def _accumulate_grad(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(
            self.data.T.copy(), (self,), "transpose", requires_grad=self.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad.T)

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Tensor(
            self.data**power, (self,), f"**{power}", requires_grad=self.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self_grad = (power * self.data ** (power - 1)) * out.grad
            self._accumulate_grad(self._match_shape(self_grad, self.shape))

        out._backward = _backward
        return out

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        out = Tensor(
            self.data + other.data, (self, other), "+", requires_grad=self.requires_grad or other.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(self._match_shape(out.grad, self.shape))
            other._accumulate_grad(self._match_shape(out.grad, other.shape))

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        out = Tensor(
            self.data @ other.data, (self, other), "@", requires_grad=self.requires_grad or other.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad @ other.data.T)
            other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        out = Tensor(
            other.data * self.data, (self, other), "*", requires_grad=self.requires_grad or other.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad

            self._accumulate_grad(self._match_shape(self_grad, self.shape))
            other._accumulate_grad(self._match_shape(other_grad, other.shape))

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg", requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(-out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other - self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            requires_grad=self.requires_grad,
        )

        def _backward():

            if out.grad is None:
                return
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(self.shape)
                for ax in axes:
                    shape[ax] = 1
                g = g.reshape(shape)
            self._accumulate_grad(np.ones_like(self.data) * g)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            (self,),
            "mean",
            requires_grad=self.requires_grad,
        )

        def _backward():
            if out.grad is None:
                return
            num_elements = (
                self.data.size if axis is None else np.prod(np.array(self.shape)[axis])
            )
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(self.shape)
                for ax in axes:
                    shape[ax] = 1
                g = g.reshape(shape)
            self._accumulate_grad((np.ones_like(self.data) * g) / num_elements)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), "max", requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(self.shape)
                for ax in axes:
                    shape[ax] = 1
                g = g.reshape(shape)
            mask = self.data == self.data.max(axis=axis, keepdims=True)
            mask = mask / mask.sum(axis=axis, keepdims=True)
            self._accumulate_grad(mask * g)

        out._backward = _backward
        return out

    # Activation Functions
    def log(self):
        out = Tensor(
            np.log(self.data + 1e-15), (self,), "log", requires_grad=self.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad / (self.data + 1e-15))

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(
            np.exp(self.data), (self,), "exp", requires_grad=self.requires_grad
        )

        def _backward():
            if out.grad is None:
                return

            self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def sigmoid(self):
        res = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(res, (self,), "sigmoid", requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.data * (1.0 - out.data) * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        res = np.tanh(self.data)
        out = Tensor(res, (self,), "tanh", requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((1.0 - res**2) * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        """Element-wise Rectified Linear Unit activation function"""
        out = Tensor(
            np.maximum(0, self.data), (self,), "ReLU", requires_grad=self.requires_grad
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        out = Tensor(
            self.data.reshape(shape),
            (self,),
            "reshape",
            requires_grad=self.requires_grad,
        )

        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def backward(self):
        if not self.requires_grad:
            return
        if self.data.size != 1:
            raise RuntimeError("Grad can only be implicitly created for scalar outputs")
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
