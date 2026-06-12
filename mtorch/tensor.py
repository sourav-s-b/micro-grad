from mtorch.config import Device

import numpy as np

class Tensor:

    __array_priority__ = 1000

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.xp = Device.xp
        if isinstance(data, self.xp.ndarray) and data.dtype == self.xp.float32:
            self.data = data
        else:
            self.data = self.xp.array(data, dtype=self.xp.float32)
        self.grad = None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None
        self.requires_grad = requires_grad and not Device.no_grad

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)
        else:
            self.grad = self.xp.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(
            self.xp.swapaxes(self.data, -1, -2).copy(),
            (self,),
            _op="T",
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:

            def _backward():
                if out.grad is None:
                    return
                self._accumulate_grad(self.xp.swapaxes(out.grad, -1, -2))

            out._backward = _backward

        return out

    def transpose(self, *axes):

        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]

        out_data = self.xp.transpose(self.data, axes)
        out = Tensor(out_data, (self,), _op='transpose', requires_grad=self.requires_grad)
        
        if self.requires_grad:

            
            inv_axes = np.argsort(axes).tolist()

            def _backward():
                if out.grad is None:
                    return
                
                self._accumulate_grad(self.xp.transpose(out.grad, inv_axes))

            out._backward = _backward

        return out




    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Tensor(
            self.data**power,
            (self,),
            _op=f"**{power}",
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
           

            def _backward():
                if out.grad is None:
                    return
                grad = (power * self.data ** (power - 1)) * out.grad
                self._accumulate_grad(grad)

            out._backward = _backward
        return out

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(
            self.data + other.data, (self, other), "+", requires_grad=requires_grad
        )

        if requires_grad:

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

        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(
            self.data @ other.data, (self, other), _op="@", requires_grad=requires_grad
        )

        if requires_grad:

            def _backward():
                if out.grad is None:
                    return
                g_self = out.grad @ self.xp.swapaxes(other.data, -1, -2)
                g_other = self.xp.swapaxes(self.data, -1, -2) @ out.grad

                self._accumulate_grad(self._match_shape(g_self, self.shape))
                other._accumulate_grad(self._match_shape(g_other, other.shape))

            out._backward = _backward

        return out

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(
            self.data * other.data, (self, other), _op="*", requires_grad=requires_grad
        )

        if requires_grad:

            def _backward():
                if out.grad is None:
                    return
                g_self = other.data * out.grad
                g_other = self.data * out.grad
                self._accumulate_grad(self._match_shape(g_self, self.shape))
                other._accumulate_grad(self._match_shape(g_other, other.shape))

            out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(
        self,
    ):

        out = Tensor(-self.data, (self,), _op="neg", requires_grad=self.requires_grad)

        if self.requires_grad:

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
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return other - self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __getitem__(self, idx):

        out = Tensor(self.data[idx] , (self,) , 'getitem' , requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None or not self.requires_grad: return

            dx = self.xp.zeros_like(self.data)
            dx[idx] = out.grad

            self._accumulate_grad(dx)

        out._backward = _backward
        return out

    # util functions

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            requires_grad=self.requires_grad,
        )
        
        if self.requires_grad:

            def _backward():
                g = out.grad
                if g is None:
                    return
                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis,int) else axis
                    shape = list(self.shape)
                    axes = tuple(
                        ax if ax >= 0 else ax + len(self.shape) for ax in axes
                    )
                    for ax in axes:
                        shape[ax] = 1
                    g = g.reshape(shape)
                self._accumulate_grad(self.xp.ones_like(self.data) * g)
            out._backward = _backward
        return out

    def mean(self, axis= None, keepdims= False):
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            (self,),
            "mean",
            requires_grad=self.requires_grad,
        )
        
        if self.requires_grad:

            def _backward():
                g = out.grad
                if g is None:
                    return

                if axis is None:
                    num_elements = self.data.size
                else:
                    axes = (axis, ) if isinstance(axis, int) else axis
                    num_elements = self.xp.prod([self.shape[a] for a in axes])

                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis,int) else axis
                    shape = list(self.shape)
                    axes = tuple(
                        ax if ax >= 0 else ax + len(self.shape) for ax in axes
                    )
                    for ax in axes:
                        shape[ax] = 1
                    g = g.reshape(shape)
                self._accumulate_grad(( self.xp.ones_like(self.data) * g ) / num_elements )
            out._backward = _backward
        return out


    def max(self, axis= None, keepdims= False):
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            (self,),
            "max",
            requires_grad=self.requires_grad,
        )
        
        if self.requires_grad:

            def _backward():
                g = out.grad
                if g is None:
                    return
                axes = (axis,) if isinstance(axis, int) else axis
                if axes is not None:
                    axes = tuple(
                        ax if ax >= 0 else ax + len(self.shape)
                        for ax in axes
                    )

                max_vals = out.data

                if not keepdims and axes is not None:
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    max_vals = max_vals.reshape(shape)
                    g = g.reshape(shape)

                mask = self.data == max_vals
                mask = mask / mask.sum(axis=axes, keepdims=True)

                self._accumulate_grad(mask * g)
            out._backward = _backward
        return out

    # activation function

    def log(self):
        out = Tensor( self.xp.log(self.data + 1e-15), (self,), 'log', requires_grad= self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                self._accumulate_grad(out.grad / (self.data + 1e-15))

            out._backward = _backward

        return out

    def exp(self):
        out = Tensor(self.xp.exp(self.data), (self,) , 'exp' , requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                self._accumulate_grad(out.grad * out.data)

            out._backward = _backward

        return out
    
    def sigmoid(self):
        res = 1.0 / (1.0 + self.xp.exp(-self.xp.clip(self.data , -500, 500)))
        out = Tensor(res, (self,) , 'sigmoid' , requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                self._accumulate_grad(out.grad * out.data *  (1 - out.data) )
            out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(self.xp.tanh(self.data), (self,) , 'tanh' , requires_grad= self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                self._accumulate_grad((1 - out.data ** 2) * out.grad)

            out._backward = _backward

        return out

    def relu(self):
        out = Tensor(self.xp.maximum(0, self.data),(self,) , 'ReLU' , requires_grad= self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return 
                self._accumulate_grad((self.data > 0) * out.grad)
            out._backward = _backward

        return out

    def silu(self):
        xp = Device.xp

        sigmoid = 1.0 / (1.0 + xp.exp(-self.data))
        out_data = self.data * sigmoid

        out = Tensor(out_data, (self, ) , 'SiLU', requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None: return

                dx = sigmoid + out_data * (1.0 - sigmoid)
                self._accumulate_grad(out.grad * dx)

            out._backward = _backward
        return out

    def softmax(self, axis=-1):

        max_val = self.xp.max(self.data, axis = axis, keepdims=True)
        exp = self.xp.exp(self.data - max_val)
        out_data = exp / self.xp.sum(exp, axis=axis, keepdims=True)

        out = Tensor(out_data , (self,) , 'softmax', requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                
                sum_ds = self.xp.sum(out.grad * out.data, axis=axis, keepdims=True)
                d_self = out.data * (out.grad - sum_ds)
                self._accumulate_grad(d_self)
            
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

        if self.requires_grad:
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
        stack = [self]

        while stack:
            node = stack[-1]
            if node not in visited:
                visited.add(node)
                univisited_children = [c for c in node._prev if c not in visited]
                if univisited_children:
                    stack.extend(univisited_children)
                else:
                    topo.append(node)
                    stack.pop()
            else:
                if node not in topo:
                    topo.append(node)
                stack.pop()

        self.grad = self.xp.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
            node._backward = lambda: None

        return topo

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = self.xp.zeros_like(self.data)
        if grad.shape != self.data.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} "
                f"does not match tensor shape {self.data.shape}"
            )
        self.grad += grad

    def _match_shape(self, grad, target_shape):
        if grad.shape == target_shape:
            return grad
        if target_shape == ():
            return self.xp.array(grad.sum())

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
        return grad


    def __repr__(self):
        return f"Tensor(shape={self.shape}, data=\n{self.data}), op={self._op}"

