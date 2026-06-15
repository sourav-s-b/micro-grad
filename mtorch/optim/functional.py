from mtorch.config import Device
from mtorch.tensor_compiled import Tensor, GraphEngine
from mtorch.utils.saves import save_model


def _softmax_ce_fwd(engine, node, xp, out_view):
    logits = engine.get_data_view(node["in_logits_id"])
    targets = engine.get_data_view(node["in_targets_id"])
    probs = engine.get_data_view(node["out_probs_id"])
    scratch_loss = engine.get_data_view(node["scratch_loss_id"])

    B = logits.shape[0]

    max_logits = xp.max(logits, axis=-1, keepdims=True)
    xp.subtract(logits, max_logits, out=probs)
    xp.exp(probs, out=probs)
    sum_exp = xp.sum(probs, axis=-1, keepdims=True)
    xp.divide(probs, sum_exp, out=probs)

    xp.add(probs, 1e-15, out=scratch_loss)
    xp.log(scratch_loss, out=scratch_loss)
    xp.multiply(targets, scratch_loss, out=scratch_loss)

    out_view[...] = -xp.sum(scratch_loss) / B


def _softmax_ce_bwd(engine, node, xp):
    targets = engine.get_data_view(node["in_targets_id"])
    probs = engine.get_data_view(node["out_probs_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])
    g_logits = engine.get_grad_view(node["in_logits_grad_id"])
    scratch_dx = engine.get_data_view(node["scratch_dx_id"])

    B = probs.shape[0]

    xp.subtract(probs, targets, out=scratch_dx)
    xp.multiply(scratch_dx, out_grad / B, out=scratch_dx)
    xp.add(g_logits, scratch_dx, out=g_logits)


GraphEngine.register_op("softmax_ce", _softmax_ce_fwd, _softmax_ce_bwd)


def softmax_cross_entropy(logits, targets):
    engine = getattr(logits, "engine", None)

    if engine is None or not engine.is_tracing:
        xp = Device.xp
        logits_data = logits.data if hasattr(logits, "data") else logits
        targets_data = targets.data if hasattr(targets, "data") else targets

        B = logits_data.shape[0]

        max_logits = xp.max(logits_data, axis=-1, keepdims=True)
        exp_logits = xp.exp(logits_data - max_logits)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)
        loss_data = -xp.sum(targets_data * xp.log(probs + 1e-15)) / B

        if hasattr(logits, "requires_grad"):
            return Tensor(loss_data, requires_grad=False), Tensor(
                probs, requires_grad=False
            )
        return loss_data, probs

    out_loss = Tensor(shape=(), requires_grad=logits.requires_grad)
    out_probs = Tensor(shape=logits.shape, requires_grad=False)
    scratch_loss = engine.alloc_scratch(logits.shape)

    engine.fwd_tape.append(
        {
            "op": "softmax_ce",
            "in_logits_id": logits.id,
            "in_targets_id": targets.id,
            "out_id": out_loss.id,
            "out_probs_id": out_probs.id,
            "scratch_loss_id": scratch_loss,
        }
    )

    if out_loss.requires_grad:
        scratch_dx = engine.alloc_scratch(logits.shape)
        engine.bwd_tape.append(
            {
                "op": "softmax_ce_bwd",
                "in_logits_grad_id": logits.id,
                "in_targets_id": targets.id,
                "out_grad_id": out_loss.id,
                "out_probs_id": out_probs.id,
                "scratch_dx_id": scratch_dx,
            }
        )

    return out_loss, out_probs


_fused_ce_backward_kernel = None


def _get_fused_ce_kernel(xp):
    global _fused_ce_backward_kernel
    if _fused_ce_backward_kernel is None:
        _fused_ce_backward_kernel = xp.ElementwiseKernel(
            in_params="T probs, raw int32 targets, T grad_val, int32 batch_size, int32 vocab_size",
            out_params="T d_logits",
            operation="""
                int row = i / vocab_size;
                int col = i % vocab_size;
                T p = probs;
                if (col == targets[row]){
                    p -= 1.0;
                }
                d_logits = p * (grad_val / (T)batch_size);
            """,
            name="fused_cross_entropy_bwd",
        )
    return _fused_ce_backward_kernel


def _cross_entropy_fwd(engine, node, xp, out_view):
    logits = engine.get_data_view(node["in_logits_id"])
    targets = engine.get_data_view(node["in_targets_id"]).astype(xp.int32)
    probs = engine.get_data_view(node["scratch_probs_id"])

    B = logits.shape[0]

    logits_max = xp.max(logits, axis=-1, keepdims=True)
    xp.subtract(logits, logits_max, out=probs)
    xp.exp(probs, out=probs)
    sum_exp = xp.sum(probs, axis=-1, keepdims=True)
    xp.divide(probs, sum_exp, out=probs)

    batch_indices = xp.arange(B)
    correct_probs = probs[batch_indices, targets]
    out_view.fill(-xp.sum(xp.log(correct_probs + 1e-8)) / B)


def _cross_entropy_bwd(engine, node, xp):
    targets = engine.get_data_view(node["in_targets_id"]).astype(xp.int32)
    probs = engine.get_data_view(node["scratch_probs_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])
    g_logits = engine.get_grad_view(node["in_logits_grad_id"])

    B, V = probs.shape[0], probs.shape[-1]

    if hasattr(xp, "ElementwiseKernel"):
        kernel = _get_fused_ce_kernel(xp)
        kernel(probs, targets, out_grad, B, V, g_logits)
    else:
        batch_indices = xp.arange(B)
        dx = probs.copy()
        dx[batch_indices, targets] -= 1.0
        dx *= out_grad / B
        xp.add(g_logits, dx, out=g_logits)


GraphEngine.register_op("cross_entropy", _cross_entropy_fwd, _cross_entropy_bwd)


def cross_entropy_loss(logits, targets):
    engine = getattr(logits, "engine", None)

    if engine is None or not engine.is_tracing:
        xp = Device.xp
        B, V = logits.shape[0], logits.shape[-1]

        logits_data = logits.data if hasattr(logits, "data") else logits
        targets_data = targets.data if hasattr(targets, "data") else targets

        probs = xp.array(logits_data, copy=True)
        logits_max = xp.max(probs, axis=-1, keepdims=True)
        probs -= logits_max
        xp.exp(probs, out=probs)
        sum_exp = xp.sum(probs, axis=-1, keepdims=True)
        probs /= sum_exp

        batch_indices = xp.arange(B)
        correct_probs = probs[batch_indices, targets_data.astype(xp.int32)]
        loss_val = -xp.sum(xp.log(correct_probs + 1e-8)) / B

        if hasattr(logits, "requires_grad"):
            return Tensor(loss_val, requires_grad=False)
        return loss_val

    out = Tensor(shape=(), requires_grad=logits.requires_grad)
    scratch_probs = engine.alloc_scratch(logits.shape)

    engine.fwd_tape.append(
        {
            "op": "cross_entropy",
            "in_logits_id": logits.id,
            "in_targets_id": targets.id,
            "out_id": out.id,
            "scratch_probs_id": scratch_probs,
        }
    )

    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "cross_entropy_bwd",
                "in_logits_grad_id": logits.id,
                "in_targets_id": targets.id,
                "out_grad_id": out.id,
                "scratch_probs_id": scratch_probs,
            }
        )
    return out


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, filepath="best_model.pkl") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.filepath = filepath
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            save_model(model, self.filepath, log=False)
            print(f"[Checkpoint] Saving best weights to: {self.filepath}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def clip_gradients(parameters, max_norm=1.0):
    xp = Device.xp
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += xp.sum(p.grad**2)
    total_norm = xp.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
