from mtorch.config import Device
import numpy as np
from mtorch import kernels


def _get_bcast_axes(in_shape, out_shape):
    """Calculates the axes that were broadcasted so they can be summed out later."""
    if in_shape == out_shape:
        return None
    ndim_diff = len(out_shape) - len(in_shape)
    axes = list(range(ndim_diff))
    for i, s in enumerate(in_shape):
        if s == 1 and out_shape[i + ndim_diff] > 1:
            axes.append(i + ndim_diff)
    return tuple(axes) if axes else None


def _get_sum_shape(out_shape, axes):
    if axes is None:
        return out_shape
    return tuple(s for i, s in enumerate(out_shape) if i not in axes)


class GraphEngine:

    _current: GraphEngine | None = None
    _custom_ops = {}

    def __init__(self) -> None:
        self.fwd_tape = []
        self.bwd_tape = []

        self.tensor_shapes = {}
        self.tensor_req_grad = {}

        self.data_buffers = {}
        self.grad_buffers = {}

        self.next_id = 0
        self.is_tracing = False
        self.is_compiled = False

    def __enter__(self):
        GraphEngine._current = self
        self.is_tracing = True
        self.fwd_tape.clear()
        self.bwd_tape.clear()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphEngine._current = None
        self.is_tracing = False

    @classmethod
    def register_op(cls, op_name, fwd_fn, bwd_fn):
        if op_name in cls._custom_ops:
            raise ValueError(f"Op '{op_name}' already registered. Use a unique name.")
        cls._custom_ops[op_name] = fwd_fn
        cls._custom_ops[op_name + "_bwd"] = bwd_fn

    def get_new_id(self):
        tid = self.next_id
        self.next_id += 1
        return tid

    def compile(self, parameters=None):
        xp = Device.xp
        self.parameter_ids = set()

        if parameters:
            for param in parameters:
                param.engine = self
                self.parameter_ids.add(param.id)

                if param.id not in self.tensor_shapes:
                    self.tensor_shapes[param.id] = param.shape
                    self.tensor_req_grad[param.id] = param.requires_grad
        for node_id, shape in self.tensor_shapes.items():

            self.data_buffers[node_id] = xp.empty(shape, dtype=xp.float32)

            if self.tensor_req_grad.get(node_id, False):
                self.grad_buffers[node_id] = xp.empty(shape, dtype=xp.float32)

        self.is_compiled = True

        if parameters:
            for param in parameters:
                if hasattr(param, "load_data_to_arena"):
                    param.load_data_to_arena()

        print(f"Graph compiled successfully")
        print(
            f"Total tape Operations: {len(self.fwd_tape)} FWD, {len(self.bwd_tape)} bwd"
        )

    def get_data_view(self, node_id):
        return self.data_buffers[node_id]

    def get_grad_view(self, node_id):
        return self.grad_buffers[node_id]

    def forward(self):
        xp = Device.xp

        for node in self.fwd_tape:
            op = node["op"]

            if op in self._custom_ops:
                out_view = self.get_data_view(node["out_id"])
                self._custom_ops[op](self, node, xp, out_view)
                continue

            out_view = self.get_data_view(node["out_id"])

            if op == "add":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                xp.add(a, b, out=out_view)

            elif op == "matmul":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                xp.matmul(a, b, out=out_view)

            elif op == "mul":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                xp.multiply(a, b, out=out_view)

            elif op == "neg":
                a = self.get_data_view(node["in_a_id"])
                xp.negative(a, out=out_view)

            elif op == "pow":
                a = self.get_data_view(node["in_a_id"])
                xp.power(a, node["power"], out=out_view)

            elif op == "sum":
                a = self.get_data_view(node["in_a_id"])
                out_view[...] = xp.sum(a, axis=node["axis"], keepdims=node["keepdims"])

            elif op == "reshape":
                a = self.get_data_view(node["in_a_id"])
                out_view[...] = a.reshape(node["shape"])

            elif op == "transpose":
                a = self.get_data_view(node["in_a_id"])
                out_view[...] = xp.transpose(a, node["axes"])

            elif op == "softmax":
                a = self.get_data_view(node["in_a_id"])
                axis = node["axis"]
                max_val = xp.max(a, axis=axis, keepdims=True)
                exp_a = xp.exp(a - max_val)
                out_view[...] = exp_a / xp.sum(exp_a, axis=axis, keepdims=True)

            elif op in ("log", "exp", "sigmoid", "tanh", "relu", "silu"):
                a = self.get_data_view(node["in_a_id"])
                kernel_func = getattr(kernels, f"{op}_fwd")
                out_view[...] = kernel_func(xp, a)

            elif op == "max":
                a = self.get_data_view(node["in_a_id"])
                out_view[...] = xp.max(a, axis=node["axis"], keepdims=node["keepdims"])

            elif op == "getitem":
                a = self.get_data_view(node["in_a_id"])
                out_view[...] = a[node["idx"]]

    def get_grad_view(self, node_id):
        return self.grad_buffers.get(node_id, None)

    def backward(self, loss_id):
        xp = Device.xp

        # purging intermediate gradients
        for node_id, grad_buf in self.grad_buffers.items():
            if node_id not in getattr(self, "parameter_ids", set()):
                grad_buf.fill(0.0)

        self.get_grad_view(loss_id).fill(1.0)
        for node in reversed(self.bwd_tape):
            op = node["op"]

            if op in self._custom_ops:
                self._custom_ops[op](self, node, xp)
                continue

            if op == "add_bwd":
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                g_b = self.get_grad_view(node["in_b_grad_id"])

                if g_a is not None:
                    if node["axes_a"] is not None:
                        scratch_a = self.get_data_view(node["scratch_a_id"])
                        scratch_a_view = scratch_a.reshape(node["sum_shape_a"])
                        xp.sum(out_grad, axis=node["axes_a"], out=scratch_a_view)
                        xp.add(g_a, scratch_a, out=g_a)
                    else:
                        xp.add(g_a, out_grad, out=g_a)

                if g_b is not None:
                    if node["axes_b"] is not None:
                        scratch_b = self.get_data_view(node["scratch_b_id"])
                        scratch_b_view = scratch_b.reshape(node["sum_shape_b"])
                        xp.sum(out_grad, axis=node["axes_b"], out=scratch_b_view)
                        xp.add(g_b, scratch_b, out=g_b)
                    else:
                        xp.add(g_b, out_grad, out=g_b)

            elif op == "matmul_bwd":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                g_b = self.get_grad_view(node["in_b_grad_id"])

                if g_a is not None:
                    scratch_a = self.get_data_view(node["scratch_a_id"])
                    xp.matmul(out_grad, xp.swapaxes(b, -1, -2), out=scratch_a)
                    if node["axes_a"] is not None:
                        scratch_a_sum = self.get_data_view(node["scratch_a_sum_id"])
                        scratch_a_view = scratch_a_sum.reshape(node["sum_shape_a"])
                        xp.sum(scratch_a, axis=node["axes_a"], out=scratch_a_view)
                        xp.add(g_a, scratch_a_sum, out=g_a)
                    else:
                        xp.add(g_a, scratch_a, out=g_a)

                if g_b is not None:
                    scratch_b = self.get_data_view(node["scratch_b_id"])
                    xp.matmul(xp.swapaxes(a, -1, -2), out_grad, out=scratch_b)
                    if node["axes_b"] is not None:
                        scratch_b_sum = self.get_data_view(node["scratch_b_sum_id"])
                        scratch_b_view = scratch_b_sum.reshape(node["sum_shape_b"])
                        xp.sum(scratch_b, axis=node["axes_b"], out=scratch_b_view)
                        xp.add(g_b, scratch_b_sum, out=g_b)
                    else:
                        xp.add(g_b, scratch_b, out=g_b)

            elif op == "mul_bwd":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                g_b = self.get_grad_view(node["in_b_grad_id"])

                if g_a is not None:
                    scratch_a_prod = self.get_data_view(node["scratch_a_prod_id"])
                    xp.multiply(out_grad, b, out=scratch_a_prod)
                    if node["axes_a"] is not None:
                        scratch_a_sum = self.get_data_view(node["scratch_a_sum_id"])
                        scratch_a_view = scratch_a_sum.reshape(node["sum_shape_a"])
                        xp.sum(scratch_a_prod, axis=node["axes_a"], out=scratch_a_view)
                        xp.add(g_a, scratch_a_sum, out=g_a)
                    else:
                        xp.add(g_a, scratch_a_prod, out=g_a)

                if g_b is not None:
                    scratch_b_prod = self.get_data_view(node["scratch_b_prod_id"])
                    xp.multiply(out_grad, a, out=scratch_b_prod)
                    if node["axes_b"] is not None:
                        scratch_b_sum = self.get_data_view(node["scratch_b_sum_id"])
                        scratch_b_view = scratch_b_sum.reshape(node["sum_shape_b"])
                        xp.sum(scratch_b_prod, axis=node["axes_b"], out=scratch_b_view)
                        xp.add(g_b, scratch_b_sum, out=g_b)
                    else:
                        xp.add(g_b, scratch_b_prod, out=g_b)

            elif op == "neg_bwd":
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    xp.subtract(g_a, out_grad, out=g_a)

            elif op == "pow_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    a = self.get_data_view(node["in_a_id"])
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    scratch = self.get_data_view(node["scratch_id"])
                    power = node["power"]

                    xp.power(a, power - 1, out=scratch)
                    xp.multiply(scratch, power, out=scratch)
                    xp.multiply(scratch, out_grad, out=scratch)
                    xp.add(g_a, scratch, out=g_a)

            elif op == "sum_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    shape_expanded = node["shape_expanded"]
                    xp.add(g_a, out_grad.reshape(shape_expanded), out=g_a)

            elif op == "reshape_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    xp.add(g_a, out_grad.reshape(g_a.shape), out=g_a)

            elif op == "transpose_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    inv_axes = node["inv_axes"]
                    xp.add(g_a, xp.transpose(out_grad, inv_axes), out=g_a)

            elif op == "softmax_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    a = self.get_data_view(node["in_a_id"])
                    out_data = self.get_data_view(node["out_id"])
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    axis = node["axis"]
                    sum_ds = xp.sum(out_grad * out_data, axis=axis, keepdims=True)
                    xp.add(g_a, out_data * (out_grad - sum_ds), out=g_a)

            elif op in (
                "log_bwd",
                "exp_bwd",
                "sigmoid_bwd",
                "tanh_bwd",
                "relu_bwd",
                "silu_bwd",
            ):
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    base_op = op.replace("_bwd", "")
                    a = self.get_data_view(node["in_a_id"])
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    kernel_func = getattr(kernels, f"{base_op}_bwd")
                    xp.add(g_a, kernel_func(xp, out_grad, a), out=g_a)

            elif op == "max_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    a = self.get_data_view(node["in_a_id"])
                    out_data = self.get_data_view(node["out_id"])
                    out_grad = self.get_grad_view(node["out_grad_id"])

                    axis = node["axis"]
                    keepdims = node["keepdims"]
                    axes = (axis,) if isinstance(axis, int) else axis

                    if axes is not None:
                        axes = tuple(
                            ax if ax >= 0 else ax + len(a.shape) for ax in axes
                        )

                    max_vals = out_data
                    if not keepdims and axes is not None:
                        shape = list(a.shape)
                        for ax in axes:
                            shape[ax] = 1
                        max_vals = max_vals.reshape(shape)
                        out_grad_reshaped = out_grad.reshape(shape)
                    else:
                        out_grad_reshaped = out_grad

                    mask = (a == max_vals).astype(xp.float32)
                    mask = mask / mask.sum(axis=axes, keepdims=True)
                    xp.add(g_a, mask * out_grad_reshaped, out=g_a)

            elif op == "getitem_bwd":
                g_a = self.get_grad_view(node["in_a_grad_id"])
                if g_a is not None:
                    out_grad = self.get_grad_view(node["out_grad_id"])
                    idx = node["idx"]
                    xp.add(g_a[idx], out_grad, out=g_a[idx])

    def alloc_scratch(self, shape):

        sid = self.get_new_id()
        self.tensor_shapes[sid] = shape
        self.tensor_req_grad[sid] = False
        return sid

    def register_parameters(self, parameters):
        for param in parameters:
            self.tensor_shapes[param.id] = param.shape
            self.tensor_req_grad[param.id] = param.requires_grad


class Tensor:

    def __init__(self, data=None, shape=None, requires_grad=True, _id=None) -> None:
        self.engine: GraphEngine | None = GraphEngine._current
        self.requires_grad = requires_grad

        if self.engine is not None and self.engine.is_tracing:
            self.id = _id if _id is not None else self.engine.get_new_id()
        else:
            self.id = _id if _id is not None else id(self)

        self.xp = Device.xp

        if data is not None:
            if not isinstance(data, self.xp.ndarray):
                data = self.xp.array(data, dtype=self.xp.float32)
            elif data.dtype != self.xp.float32:
                data = data.astype(self.xp.float32)
            self.shape = data.shape
            self._temp_data = data
            self._temp_grad = self.xp.zeros_like(data) if requires_grad else None

        else:
            self.shape = shape
            self._temp_data = None
            self._temp_grad = None

        if self.engine is not None and self.engine.is_tracing:
            self.engine.tensor_shapes[self.id] = self.shape
            self.engine.tensor_req_grad[self.id] = self.requires_grad

    def load_data_to_arena(self):

        if self._temp_data is not None and self.engine and self.engine.is_compiled:
            view = self.engine.get_data_view(self.id)
            view[...] = self._temp_data
            self._temp_data = None

    def update_input(self, data):
        if self.engine:
            view = self.engine.get_data_view(self.id)

            if hasattr(data, "data"):
                raw_data = data.data
            else:
                raw_data = data
            view[...] = raw_data

    @property
    def data(self):
        if self.engine and self.engine.is_compiled:
            return self.engine.get_data_view(self.id)
        return self._temp_data

    @property
    def grad(self):
        if self.engine and self.engine.is_compiled:
            return self.engine.get_grad_view(self.id)
        return self._temp_grad

    def zero_grad(self):
        if self.engine and self.engine.is_compiled:
            self.engine.get_grad_view(self.id).fill(0.0)
        elif self._temp_grad is not None:
            self._temp_grad.fill(0.0)

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        assert self.shape is not None and other.shape is not None
        out_shape = np.broadcast_shapes(self.shape, other.shape)

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(
                shape=out_shape, requires_grad=self.requires_grad or other.requires_grad
            )

            self.engine.tensor_shapes[self.id] = self.shape
            self.engine.tensor_req_grad[self.id] = self.requires_grad
            self.engine.tensor_shapes[other.id] = other.shape
            self.engine.tensor_req_grad[other.id] = other.requires_grad

            axes_a = _get_bcast_axes(self.shape, out_shape)
            axes_b = _get_bcast_axes(other.shape, out_shape)
            sum_shape_a = _get_sum_shape(out_shape, axes_a)
            sum_shape_b = _get_sum_shape(out_shape, axes_b)

            scratch_a_id = self.engine.alloc_scratch(self.shape) if axes_a else None
            scratch_b_id = self.engine.alloc_scratch(other.shape) if axes_b else None

            self.engine.fwd_tape.append(
                {"op": "add", "in_a_id": self.id, "in_b_id": other.id, "out_id": out.id}
            )

            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "add_bwd",
                        "in_a_id": self.id,
                        "in_b_id": other.id,
                        "in_a_grad_id": self.id,
                        "in_b_grad_id": other.id,
                        "out_grad_id": out.id,
                        "axes_a": axes_a,
                        "axes_b": axes_b,
                        "sum_shape_a": sum_shape_a,
                        "sum_shape_b": sum_shape_b,
                        "scratch_a_id": scratch_a_id,
                        "scratch_b_id": scratch_b_id,
                    }
                )
            return out

        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph structure is locked! Cannot instantiate new structural tensors post-compilation."
            )
        else:
            assert (
                self.data is not None and other.data is not None
            ), "Eager tensors must contain data."
            out_data = self.data + other.data
            return Tensor(
                out_data, requires_grad=self.requires_grad or other.requires_grad
            )

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        assert self.shape is not None and other.shape is not None
        out_shape = np.broadcast_shapes(self.shape, other.shape)

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(
                shape=out_shape, requires_grad=self.requires_grad or other.requires_grad
            )

            self.engine.tensor_shapes[self.id] = self.shape
            self.engine.tensor_req_grad[self.id] = self.requires_grad
            self.engine.tensor_shapes[other.id] = other.shape
            self.engine.tensor_req_grad[other.id] = other.requires_grad
            axes_a = _get_bcast_axes(self.shape, out_shape)
            axes_b = _get_bcast_axes(other.shape, out_shape)

            sum_shape_a = _get_sum_shape(out_shape, axes_a)
            sum_shape_b = _get_sum_shape(out_shape, axes_b)

            scratch_a_prod_id = self.engine.alloc_scratch(out_shape)
            scratch_b_prod_id = self.engine.alloc_scratch(out_shape)

            scratch_a_sum_id = self.engine.alloc_scratch(self.shape) if axes_a else None
            scratch_b_sum_id = (
                self.engine.alloc_scratch(other.shape) if axes_b else None
            )

            self.engine.fwd_tape.append(
                {"op": "mul", "in_a_id": self.id, "in_b_id": other.id, "out_id": out.id}
            )

            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "mul_bwd",
                        "in_a_id": self.id,
                        "in_b_id": other.id,
                        "in_a_grad_id": self.id,
                        "in_b_grad_id": other.id,
                        "out_grad_id": out.id,
                        "axes_a": axes_a,
                        "axes_b": axes_b,
                        "sum_shape_a": sum_shape_a,
                        "sum_shape_b": sum_shape_b,
                        "scratch_a_prod_id": scratch_a_prod_id,
                        "scratch_b_prod_id": scratch_b_prod_id,
                        "scratch_a_sum_id": scratch_a_sum_id,
                        "scratch_b_sum_id": scratch_b_sum_id,
                    }
                )
            return out

        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph structure is locked! Cannot instantiate new structural tensors post-compilation."
            )
        else:
            assert (
                self.data is not None and other.data is not None
            ), "Eager tensors must contain data."
            out_data = self.xp.multiply(self.data, other.data)
            return Tensor(
                out_data, requires_grad=self.requires_grad or other.requires_grad
            )

    def __matmul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        assert self.shape is not None and other.shape is not None
        out_shape = list(self.shape[:-1]) + [other.shape[-1]]

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(
                shape=tuple(out_shape),
                requires_grad=self.requires_grad or other.requires_grad,
            )

            self.engine.tensor_shapes[self.id] = self.shape
            self.engine.tensor_req_grad[self.id] = self.requires_grad
            self.engine.tensor_shapes[other.id] = other.shape
            self.engine.tensor_req_grad[other.id] = other.requires_grad
            scratch_a_shape = tuple(out_shape[:-2]) + (self.shape[-2], other.shape[-2])
            scratch_b_shape = tuple(out_shape[:-2]) + (self.shape[-1], other.shape[-1])

            scratch_a_id = self.engine.alloc_scratch(scratch_a_shape)
            scratch_b_id = self.engine.alloc_scratch(scratch_b_shape)

            axes_a = _get_bcast_axes(self.shape, scratch_a_shape)
            axes_b = _get_bcast_axes(other.shape, scratch_b_shape)
            sum_shape_a = _get_sum_shape(out_shape, axes_a)
            sum_shape_b = _get_sum_shape(out_shape, axes_b)

            scratch_a_sum_id = self.engine.alloc_scratch(self.shape) if axes_a else None
            scratch_b_sum_id = (
                self.engine.alloc_scratch(other.shape) if axes_b else None
            )

            self.engine.fwd_tape.append(
                {
                    "op": "matmul",
                    "in_a_id": self.id,
                    "in_b_id": other.id,
                    "out_id": out.id,
                }
            )

            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "matmul_bwd",
                        "in_a_id": self.id,
                        "in_b_id": other.id,
                        "in_a_grad_id": self.id,
                        "in_b_grad_id": other.id,
                        "out_grad_id": out.id,
                        "axes_a": axes_a,
                        "axes_b": axes_b,
                        "sum_shape_a": sum_shape_a,
                        "sum_shape_b": sum_shape_b,
                        "scratch_a_id": scratch_a_id,
                        "scratch_b_id": scratch_b_id,
                        "scratch_a_sum_id": scratch_a_sum_id,
                        "scratch_b_sum_id": scratch_b_sum_id,
                    }
                )
            return out

        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph structure is locked! Cannot instantiate new structural tensors post-compilation."
            )
        else:
            assert (
                self.data is not None and other.data is not None
            ), "Eager tensors must contain data."
            out_data = self.xp.matmul(self.data, other.data)
            return Tensor(
                out_data, requires_grad=self.requires_grad or other.requires_grad
            )

    def __neg__(self):

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(shape=self.shape, requires_grad=self.requires_grad)

            self.engine.fwd_tape.append(
                {"op": "neg", "in_a_id": self.id, "out_id": out.id}
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {"op": "neg_bwd", "out_grad_id": out.id, "in_a_grad_id": self.id}
                )

            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph structure is locked! Cannot instantiate new structural tensors post-compilation."
            )
        else:
            assert self.data is not None, "Eager tensors must contain data."

            return Tensor(-self.data, requires_grad=self.requires_grad)

    def __pow__(self, power):
        assert isinstance(power, (int, float))

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(shape=self.shape, requires_grad=self.requires_grad)

            scratch_id = self.engine.alloc_scratch(self.shape)

            self.engine.fwd_tape.append(
                {"op": "pow", "in_a_id": "self.id", "power": power, "out_id": out.id}
            )

            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "pow_bwd",
                        "in_a_id": self.id,
                        "power": power,
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                        "scratch_id": scratch_id,
                    }
                )

            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph structure is locked! Cannot instantiate new structural tensors post-compilation."
            )
        else:
            assert self.data is not None, "Eager tensors must contain data."

            return Tensor(self.data, requires_grad=self.requires_grad)

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def mean(self, axis=None, keepdims=False):
        assert self.shape is not None
        if axis is None:
            num_elements = np.prod(self.shape)

        else:
            axes = (axis,) if isinstance(axis, int) else axis
            num_elements = np.prod([self.shape[a] for a in axes])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / num_elements)

    def sum(self, axis=None, keepdims=False):
        if self.engine is not None and self.engine.is_tracing:
            assert self.shape is not None
            axes = (axis,) if isinstance(axis, int) else axis
            if axes is None:
                out_shape = ()
            else:
                _REMOVED = object()
                out_shape_list = list(self.shape)
                for ax in axes:
                    out_shape_list[ax] = 1 if keepdims else _REMOVED
                out_shape = tuple([s for s in out_shape_list if s is not _REMOVED])

            shape_expanded = list(self.shape)
            if axes is not None:
                for ax in axes:
                    shape_expanded[ax if ax >= 0 else ax + len(self.shape)] = 1
            else:
                shape_expanded = [1] * len(self.shape)

            out = Tensor(shape=out_shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {
                    "op": "sum",
                    "in_a_id": self.id,
                    "axis": axis,
                    "keepdims": keepdims,
                    "out_id": out.id,
                }
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "sum_bwd",
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                        "shape_expanded": tuple(shape_expanded),
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            return Tensor(
                self.data.sum(axis=axis, keepdims=keepdims),
                requires_grad=self.requires_grad,
            )

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(shape=tuple(shape), requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {
                    "op": "reshape",
                    "in_a_id": self.id,
                    "shape": tuple(shape),
                    "out_id": out.id,
                }
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "reshape_bwd",
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        if self.engine is not None and self.engine.is_tracing:
            assert self.shape is not None
            inv_axes = np.argsort(axes).tolist()
            out_shape = tuple([self.shape[i] for i in axes])
            out = Tensor(shape=out_shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {"op": "transpose", "in_a_id": self.id, "axes": axes, "out_id": out.id}
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "transpose_bwd",
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                        "inv_axes": inv_axes,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            return Tensor(
                self.xp.transpose(self.data, axes), requires_grad=self.requires_grad
            )

    @property
    def T(self):
        assert self.shape is not None
        axes = list(range(len(self.shape)))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        if self.engine is not None and self.engine.is_tracing:
            # We must run a dummy eager slice to figure out the resulting shape
            dummy_slice = self.xp.empty(self.shape)[idx]
            out_shape = dummy_slice.shape

            out = Tensor(shape=out_shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {"op": "getitem", "in_a_id": self.id, "idx": idx, "out_id": out.id}
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "getitem_bwd",
                        "in_a_grad_id": self.id,
                        "out_grad_id": out.id,
                        "idx": idx,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            return Tensor(self.data[idx], requires_grad=self.requires_grad)

    def max(self, axis=None, keepdims=False):
        if self.engine is not None and self.engine.is_tracing:
            assert self.shape is not None

            # Calculate output shape
            axes = (axis,) if isinstance(axis, int) else axis
            if axes is None:
                out_shape = ()
            else:
                _REMOVED = object()
                out_shape_list = list(self.shape)
                for ax in axes:
                    out_shape_list[ax] = 1 if keepdims else _REMOVED
                out_shape = tuple([s for s in out_shape_list if s is not _REMOVED])

            out = Tensor(shape=out_shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {
                    "op": "max",
                    "in_a_id": self.id,
                    "axis": axis,
                    "keepdims": keepdims,
                    "out_id": out.id,
                }
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "max_bwd",
                        "in_a_id": self.id,
                        "in_a_grad_id": self.id,
                        "out_id": out.id,
                        "out_grad_id": out.id,
                        "axis": axis,
                        "keepdims": keepdims,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            return Tensor(
                self.data.max(axis=axis, keepdims=keepdims),
                requires_grad=self.requires_grad,
            )

    def _activation_helper(self, op_name):
        from mtorch import kernels

        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {"op": op_name, "in_a_id": self.id, "out_id": out.id}
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": f"{op_name}_bwd",
                        "in_a_id": self.id,
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            kernel_func = getattr(kernels, f"{op_name}_fwd")
            return Tensor(
                kernel_func(self.xp, self.data), requires_grad=self.requires_grad
            )

    def log(self):
        return self._activation_helper("log")

    def exp(self):
        return self._activation_helper("exp")

    def sigmoid(self):
        return self._activation_helper("sigmoid")

    def tanh(self):
        return self._activation_helper("tanh")

    def relu(self):
        return self._activation_helper("relu")

    def silu(self):
        return self._activation_helper("silu")

    def softmax(self, axis=-1):
        if self.engine is not None and self.engine.is_tracing:
            out = Tensor(shape=self.shape, requires_grad=self.requires_grad)
            self.engine.fwd_tape.append(
                {"op": "softmax", "in_a_id": self.id, "axis": axis, "out_id": out.id}
            )
            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "softmax_bwd",
                        "in_a_id": self.id,
                        "out_id": out.id,
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                        "axis": axis,
                    }
                )
            return out
        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError("Graph locked!")
        else:
            assert self.data is not None, "Data missing"
            max_val = self.xp.max(self.data, axis=axis, keepdims=True)
            exp_a = self.xp.exp(self.data - max_val)
            return Tensor(
                exp_a / self.xp.sum(exp_a, axis=axis, keepdims=True),
                requires_grad=self.requires_grad,
            )
