from mtorch.config import Device
from mtorch.nn import Module
from mtorch.tensor_compiled import GraphEngine, Tensor
from mtorch.nn.functional import linear as F


class Linear(Module):

    def __init__(self, nin, nout):
        super().__init__()

        bound = 1 / Device.xp.sqrt(nin)

        self.W = Tensor(
            Device.xp.random.uniform(-bound, bound, (nin, nout)), requires_grad=True
        )
        self.B = Tensor(Device.xp.zeros((1, nout)), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B]


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        if "dropout" not in GraphEngine._custom_ops:
            GraphEngine.register_op("dropout", F._dropout_fwd, F._dropout_bwd)

    def __call__(self, x):
        engine = getattr(x, "engine", None)

        # EAGER MODE FALLBACK (For Inference)
        if engine is None or not engine.is_tracing:
            if self.training and self.p > 0:
                xp = Device.xp
                x_data = x.data if hasattr(x, "data") else x
                mask = (xp.random.rand(*x_data.shape) >= self.p) / (1.0 - self.p)
                out_data = x_data * mask
                return Tensor(
                    out_data.astype(xp.float32),
                    requires_grad=getattr(x, "requires_grad", False),
                )
            return x

        # COMPILED GRAPH MODE
        if not self.training or self.p == 0:
            return x

        out = Tensor(shape=x.shape, requires_grad=x.requires_grad)
        scratch_mask = engine.alloc_scratch(x.shape)

        engine.fwd_tape.append(
            {
                "op": "dropout",
                "in_x_id": x.id,
                "out_id": out.id,
                "p": self.p,
                "scratch_mask_id": scratch_mask,
            }
        )

        if out.requires_grad:
            scratch_dx = engine.alloc_scratch(x.shape)
            engine.bwd_tape.append(
                {
                    "op": "dropout_bwd",
                    "in_x_id": x.id,
                    "in_x_grad_id": x.id,
                    "out_grad_id": out.id,
                    "scratch_mask_id": scratch_mask,
                    "scratch_dx_id": scratch_dx,
                }
            )
        return out
