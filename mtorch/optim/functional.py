from mtorch.config import Device

from mtorch.tensor import Tensor
from mtorch.utils.saves import save_model


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

    b = logits.shape[0]
    xp = Device.xp
    logits_data = logits.data
    target_data = (
        targets.data.astype(xp.int32)
        if hasattr(targets, "data")
        else targets.astype(xp.int32)
    )

    logits_max = xp.max(logits_data, axis=-1, keepdims=True)
    shifted_logits = logits_data - logits_max

    exp_logits = xp.exp(shifted_logits)
    sum_exp = xp.sum(exp_logits, axis=-1, keepdims=True)

    log_probs = shifted_logits - xp.log(sum_exp)

    correct_probs = log_probs[xp.arange(b), target_data]

    loss_data = -xp.mean(correct_probs)

    out = Tensor(
        loss_data, (logits,), "CrossEntropy", requires_grad=logits.requires_grad
    )

    if logits.requires_grad:

        def _backward():
            if out.grad is None:
                return
            probs = exp_logits / sum_exp

            dx = probs.copy()
            dx[xp.arange(b), target_data] -= 1.0

            # Scale by the sequence length (N) and multiply by the incoming gradient
            dx = dx * (out.grad / b)

            logits._accumulate_grad(dx)

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
