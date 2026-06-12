from mtorch.config import Device
import numpy as np


class GraphEngine:

    _current: GraphEngine | None = None

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

    def get_new_id(self):
        tid = self.next_id
        self.next_id += 1
        return tid

    def compile(self, parameters=None):
        xp = Device.xp

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
            out_view = self.get_data_view(node["out_id"])

            if op == "add":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                xp.add(a, b, out=out_view)

            elif op == "matmul":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                xp.matmul(a, b, out=out_view)

    def backward(self, loss_id):
        xp = Device.xp
        self.get_grad_view(loss_id).fill(1.0)

        for node in reversed(self.bwd_tape):
            op = node["op"]

            if op == "add_bwd":
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                g_b = self.get_grad_view(node["in_b_grad_id"])

                xp.add(g_a, out_grad, out=g_a)
                xp.add(g_b, out_grad, out=g_b)

            elif op == "matmul_bwd":
                a = self.get_data_view(node["in_a_id"])
                b = self.get_data_view(node["in_b_id"])
                out_grad = self.get_grad_view(node["out_grad_id"])
                g_a = self.get_grad_view(node["in_a_grad_id"])
                g_b = self.get_grad_view(node["in_b_grad_id"])

                scratch_a = self.get_data_view(node["scratch_a_id"])
                scratch_b = self.get_data_view(node["scratch_b_id"])

                # Gradients for A
                xp.matmul(out_grad, xp.swapaxes(b, -1, -2), out=scratch_a)
                if scratch_a.shape != g_a.shape:
                    sum_axis = tuple(range(scratch_a.ndim - g_a.ndim))
                    xp.add(g_a, xp.sum(scratch_a, axis=sum_axis), out=g_a)
                else:
                    xp.add(g_a, scratch_a, out=g_a)

                # Gradients for B
                raw_gb = xp.matmul(xp.swapaxes(a, -1, -2), out_grad)
                if raw_gb.shape != g_b.shape:
                    sum_axis = tuple(range(raw_gb.ndim - g_b.ndim))
                    xp.add(g_b, xp.sum(raw_gb, axis=sum_axis), out=g_b)
                else:
                    xp.matmul(xp.swapaxes(a, -1, -2), out_grad, out=scratch_b)
                    xp.add(g_b, scratch_b, out=g_b)


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
            view[:] = self._temp_data
            self._temp_data = None

    def update_input(self, data):
        if self.engine:
            view = self.engine.get_data_view(self.id)
            view[:] = data

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

            self.engine.fwd_tape.append(
                {"op": "add", "in_a_id": self.id, "in_b_id": other.id, "out_id": out.id}
            )

            if out.requires_grad:
                self.engine.bwd_tape.append(
                    {
                        "op": "add_bwd",
                        "in_a_id": self.id,
                        "in_b_id": other.id,
                        "out_grad_id": out.id,
                    }
                )

            return out

        elif self.engine is not None and self.engine.is_compiled:
            raise RuntimeError(
                "Graph is locked ! Cannot instatiate new structural tensors after post-compilation"
            )

        else:
            assert (
                self.data is not None and other.data is not None
            ), "Eager tensors must contain data."
            out_data = self.data + other.data
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

            scratch_a_id = self.engine.get_new_id()
            scratch_b_id = self.engine.get_new_id()

            self.engine.tensor_shapes[scratch_a_id] = tuple(out_shape[:-3]) + (
                self.shape[-2],
                other.shape[-2],
            )
            self.engine.tensor_shapes[scratch_b_id] = tuple(out_shape[:-2]) + (
                self.shape[-1],
                other.shape[-1],
            )

            self.engine.tensor_req_grad[scratch_a_id] = False
            self.engine.tensor_req_grad[scratch_b_id] = False

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
                        "out_grad_id": out.id,
                        "in_a_grad_id": self.id,
                        "in_b_grad_id": other.id,
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
            out_data = self.xp.matmul(self.data, other.data)
            return Tensor(
                out_data, requires_grad=self.requires_grad or other.requires_grad
            )
