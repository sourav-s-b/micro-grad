import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ==========================================
# 1. EXP
# ==========================================
def _exp_fwd_np(x):
    return np.exp(x)


def _exp_bwd_np(grad, x):
    return grad * np.exp(x)


if HAS_CUPY:

    @cp.fuse()
    def _exp_fwd_cp(x):
        return cp.exp(x)

    @cp.fuse()
    def _exp_bwd_cp(grad, x):
        return grad * cp.exp(x)


def exp_fwd(xp, x):
    return _exp_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _exp_fwd_np(x)


def exp_bwd(xp, grad, x):
    return (
        _exp_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _exp_bwd_np(grad, x)
    )


# ==========================================
# 2. LOG
# ==========================================
def _log_fwd_np(x):
    return np.log(x + 1e-15)


def _log_bwd_np(grad, x):
    return grad / (x + 1e-15)


if HAS_CUPY:

    @cp.fuse()
    def _log_fwd_cp(x):
        return cp.log(x + 1e-15)

    @cp.fuse()
    def _log_bwd_cp(grad, x):
        return grad / (x + 1e-15)


def log_fwd(xp, x):
    return _log_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _log_fwd_np(x)


def log_bwd(xp, grad, x):
    return (
        _log_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _log_bwd_np(grad, x)
    )


# ==========================================
# 3. RELU
# ==========================================
def _relu_fwd_np(x):
    return np.maximum(0, x)


def _relu_bwd_np(grad, x):
    return grad * (x > 0)


if HAS_CUPY:

    @cp.fuse()
    def _relu_fwd_cp(x):
        return cp.maximum(0, x)

    @cp.fuse()
    def _relu_bwd_cp(grad, x):
        return grad * (x > 0)


def relu_fwd(xp, x):
    return _relu_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _relu_fwd_np(x)


def relu_bwd(xp, grad, x):
    return (
        _relu_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _relu_bwd_np(grad, x)
    )


# ==========================================
# 4. TANH
# ==========================================
def _tanh_fwd_np(x):
    return np.tanh(x)


def _tanh_bwd_np(grad, x):
    return grad * (1 - np.tanh(x) ** 2)


if HAS_CUPY:

    @cp.fuse()
    def _tanh_fwd_cp(x):
        return cp.tanh(x)

    @cp.fuse()
    def _tanh_bwd_cp(grad, x):
        return grad * (1 - cp.tanh(x) ** 2)


def tanh_fwd(xp, x):
    return _tanh_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _tanh_fwd_np(x)


def tanh_bwd(xp, grad, x):
    return (
        _tanh_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _tanh_bwd_np(grad, x)
    )


# ==========================================
# 5. SIGMOID
# ==========================================
def _sigmoid_fwd_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_bwd_np(grad, x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return grad * s * (1 - s)


if HAS_CUPY:

    @cp.fuse()
    def _sigmoid_fwd_cp(x):
        return 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500)))

    @cp.fuse()
    def _sigmoid_bwd_cp(grad, x):
        s = 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500)))
        return grad * s * (1 - s)


def sigmoid_fwd(xp, x):
    return (
        _sigmoid_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _sigmoid_fwd_np(x)
    )


def sigmoid_bwd(xp, grad, x):
    return (
        _sigmoid_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _sigmoid_bwd_np(grad, x)
    )


# ==========================================
# 6. SILU (Swish)
# ==========================================
def _silu_fwd_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def _silu_bwd_np(grad, x):
    s = 1.0 / (1.0 + np.exp(-x))
    return grad * (s + x * s * (1.0 - s))


if HAS_CUPY:

    @cp.fuse()
    def _silu_fwd_cp(x):
        return x * (1.0 / (1.0 + cp.exp(-x)))

    @cp.fuse()
    def _silu_bwd_cp(grad, x):
        s = 1.0 / (1.0 + cp.exp(-x))
        return grad * (s + x * s * (1.0 - s))


def silu_fwd(xp, x):
    return _silu_fwd_cp(x) if HAS_CUPY and xp.__name__ == "cupy" else _silu_fwd_np(x)


def silu_bwd(xp, grad, x):
    return (
        _silu_bwd_cp(grad, x)
        if HAS_CUPY and xp.__name__ == "cupy"
        else _silu_bwd_np(grad, x)
    )
