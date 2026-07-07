from numpy import gradient

from mtorch.config import Device
from mtorch.tensor import Tensor
from mtorch.utils.saves import save_model

_fused_ce_backward_kernel = None


def softmax_cross_entropy(logits, targets):

    max_logits = Device.xp.max(logits.data, axis=-1, keepdims=True)
    exp_logits = Device.xp.exp(logits.data - max_logits)
    probs = exp_logits / Device.xp.sum(exp_logits, axis=-1, keepdims=True)

    batch_size = logits.shape[0]

    loss_data = -Device.xp.sum(targets.data * Device.xp.log(probs + 1e-15)) / batch_size

    out = Tensor(
        loss_data, (logits, targets), "SoftmaxCrossEntropy", requires_grad=True
    )

    def _backward():
        logits._accumulate_grad(((probs - targets.data) / batch_size) * out.grad)

    out._backward = _backward

    return out, probs


def cross_entropy_loss(logits, targets):

    xp = Device.xp
    B = logits.shape[0]
    V = logits.shape[1]

    if type(logits).__name__ == "Tensor":
        logits_data = logits.data
    else:
        logits_data = logits

    if type(targets).__name__ == "Tensor":
        targets_data = targets.data
    else:
        targets_data = targets

    probs = xp.array(logits.data, copy=True)
    logits_max = xp.max(probs, axis=-1, keepdims=True)
    probs -= logits_max
    xp.exp(probs, out=probs)
    sum_exp = xp.sum(probs, axis=-1, keepdims=True)
    probs /= sum_exp

    batch_indices = xp.arange(B)
    correct_probs = probs[batch_indices, targets_data.astype(xp.int32)]
    loss_val = -xp.sum(xp.log(correct_probs + 1e-8)) / B

    out = Tensor(
        loss_val, (logits,), "CrossEntropy", requires_grad=logits.requires_grad
    )

    if logits.requires_grad:

        def _backward():
            if out.grad is None:
                return
            is_cupy = hasattr(xp, "ElementwiseKernel")

            if is_cupy:
                d_logits = xp.empty_like(probs)
                kernel = _get_fused_ce_kernel(xp)

                kernel(
                    probs,
                    targets_data.astype(xp.int32),
                    out.grad,
                    B,
                    V,
                    d_logits,
                )
            else:
                d_logits = probs

                d_logits[batch_indices, targets_data.astype(xp.int32)] -= 1.0
                d_logits *= out.grad / B

            logits._accumulate_grad(d_logits)

        out._backward = _backward
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


def clip_gradients(parameters, max_norm=1.0):
    """Scales down gradients if they explode past max_norm."""
    xp = Device.xp
    total_norm = 0.0

    # Calculate the total magnitude of all gradients
    for p in parameters:
        if p.grad is not None:
            total_norm += xp.sum(p.grad**2)

    total_norm = xp.sqrt(total_norm)

    # If it's too big, scale everything down proportionally
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef


def checkpoint(layer_function, x):

    Device.no_grad = True
    detached_output = layer_function(x)
    Device.no_grad = False

    out = Tensor(
        detached_output.data,
        _children=(x,),
        _op="checkpoint",
        requires_grad=x.requires_grad,
    )

    if out.requires_grad:

        def _backward():
            if out.grad is None:
                return

            Device.no_grad = False

            recomputed_out = layer_function(x)

            recomputed_out.grad = out.grad

            recomputed_out.backward(gradient=out.grad)

            del recomputed_out

        out._backward = _backward

    return out
