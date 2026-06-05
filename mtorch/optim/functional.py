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
        loss_data, (logits, targets), "softmax_cross_entropy", requires_grad=True
    )

    def _backward():
        logits._accumulate_grad(((probs - targets.data) / batch_size) * out.grad)

    out._backward = _backward

    return out, probs


def cross_entropy_loss(logits, targets):

    b = logits.shape[0]

    logits_max = Device.xp.max(logits.data, axis=-1, keepdims=True)
    exp_logits = Device.xp.exp(logits.data - logits_max)
    probs = exp_logits / Device.xp.sum(exp_logits, axis=-1, keepdims=True)

    target_data = targets.data.astype(int)
    correct_probs = probs[Device.xp.arange(b), target_data]

    loss_data = -Device.xp.sum(Device.xp.log(correct_probs + 1e-15)) / b

    out = Tensor(loss_data, (logits, targets), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            dlogits = probs.copy()
            dlogits[Device.xp.arange(b), target_data] -= 1.0
            dlogits = dlogits / b
            logits._accumulate_grad(dlogits)

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
