import numpy as np

from mtorch.tensor import Tensor


def softmax_cross_entropy(logits, targets):

    max_logits = np.max(logits.data, axis=-1, keepdims=True)
    exp_logits = np.exp(logits.data - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    batch_size = logits.shape[0]

    loss_data = -np.sum(targets.data * np.log(probs + 1e-15)) / batch_size

    out = Tensor(
        loss_data, (logits, targets), "softmax_cross_entropy", requires_grad=True
    )

    def _backward():
        logits._accumulate_grad(((probs - targets.data) / batch_size) * out.grad)

    out._backward = _backward

    return out, probs
