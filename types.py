import math


class Value:
    def __init__(self, data, children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = tuple(children)
        self._op = _op
        self._backward = lambda: None

    # Basic operations

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "Only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data + self.data, (self, other), "+")

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data * self.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    # -- Right Side operation
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # -- Sub & Div
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return Value(other) + -self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    # -- Activation Function

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value((math.exp(x)), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        out = Value(0.0 if x < 0 else x, (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), "Sigmoid")

        def _backward():
            self.grad += (s * (1 - s)) * out.grad

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

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
