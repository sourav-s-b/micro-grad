from mtorch.config import Device
from mtorch.tensor_compiled import GraphEngine, Tensor

# 1. Embedding


def _embedding_fwd(engine, node, xp, out_view):
    w = engine.get_data_view(node["in_w_id"])
    idx = engine.get_data_view(node["in_idx_id"]).astype(int)
    out_view[:] = w[idx]


def _embedding_bwd(engine, node, xp):
    w = engine.get_data_view(node["in_w_id"])
    idx = engine.get_data_view(node["in_idx_id"]).astype(int)
    out_grad = engine.get_data_view(node["out_grad_id"])
    gw = engine.get_grad_view(node["in_w_grad_id"])

    flat_idx = idx.reshape(-1)
    # [FIXED]: Corrected typo from "rehape" to "reshape"
    flat_grad = out_grad.reshape(-1, node["embedding_dim"])

    if Device.device == "cuda":
        from cupyx import scatter_add

        scatter_add(gw, flat_idx, flat_grad)
    else:
        xp.add.at(gw, flat_idx, flat_grad)


GraphEngine.register_op("embedding", _embedding_fwd, _embedding_bwd)


def embedding(indices, weight, embedding_dim):
    engine = indices.engine
    if engine is None or not engine.is_tracing:
        idx_data = indices.data if hasattr(indices, "data") else indices
        w_data = weight.data if hasattr(weight, "data") else weight

        out_data = w_data[idx_data.astype(int)]

        if hasattr(indices, "requires_grad"):
            return Tensor(out_data, requires_grad=False)
        return out_data
    out = Tensor(
        shape=tuple(list(indices.shape) + [embedding_dim]),
        requires_grad=weight.requires_grad,
    )

    engine.fwd_tape.append(
        {
            "op": "embedding",
            "in_idx_id": indices.id,
            "in_w_id": weight.id,
            "out_id": out.id,
        }
    )
    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "embedding_bwd",
                "in_idx_id": indices.id,
                "in_w_id": weight.id,
                "in_w_grad_id": weight.id,
                "out_grad_id": out.id,
                "embedding_dim": embedding_dim,
            }
        )
    return out


# 2. LayerNorm


def _layernorm_fwd_kernel(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    gamma = engine.get_data_view(node["in_gamma_id"])
    beta = engine.get_data_view(node["in_beta_id"])

    x_norm = engine.get_data_view(node["scratch_x_norm_id"])
    std_inv = engine.get_data_view(node["scratch_std_inv_id"])
    s_mean = engine.get_data_view(node["scratch_mean_id"])
    s_var = engine.get_data_view(node["scratch_var_id"])

    xp.mean(x, axis=-1, keepdims=True, out=s_mean)
    xp.subtract(x, s_mean, out=x_norm)
    xp.square(x_norm, out=s_var)
    xp.mean(s_var, axis=-1, keepdims=True, out=s_var)

    xp.add(s_var, node["eps"], out=std_inv)
    xp.sqrt(std_inv, out=std_inv)
    xp.divide(1.0, std_inv, out=std_inv)

    xp.multiply(x_norm, std_inv, out=x_norm)
    xp.multiply(x_norm, gamma, out=out_view)
    xp.add(out_view, beta, out=out_view)


def _layernorm_bwd_kernel(engine, node, xp):
    x = engine.get_data_view(node["in_x_id"])
    gamma = engine.get_data_view(node["in_gamma_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])

    gx = engine.get_grad_view(node["in_x_grad_id"])
    g_gamma = engine.get_grad_view(node["in_gamma_grad_id"])
    g_beta = engine.get_grad_view(node["in_beta_grad_id"])

    x_norm = engine.get_data_view(node["scratch_x_norm_id"])
    std_inv = engine.get_data_view(node["scratch_std_inv_id"])

    scratch_g = engine.get_data_view(node["scratch_g_id"])
    scratch_dx = engine.get_data_view(node["scratch_dx_id"])
    s_mean_g = engine.get_data_view(node["scratch_mean_g_id"])
    s_mean_g_xnorm = engine.get_data_view(node["scratch_mean_g_xnorm_id"])

    reduce_axes = tuple(range(x.ndim - 1))

    xp.multiply(out_grad, x_norm, out=scratch_dx)
    xp.add(g_gamma, xp.sum(scratch_dx, axis=reduce_axes), out=g_gamma)
    xp.add(g_beta, xp.sum(out_grad, axis=reduce_axes), out=g_beta)

    xp.multiply(out_grad, gamma, out=scratch_g)
    xp.mean(scratch_g, axis=-1, keepdims=True, out=s_mean_g)

    xp.multiply(scratch_g, x_norm, out=scratch_dx)
    xp.mean(scratch_dx, axis=-1, keepdims=True, out=s_mean_g_xnorm)

    xp.multiply(x_norm, s_mean_g_xnorm, out=scratch_dx)
    xp.subtract(scratch_g, s_mean_g, out=scratch_g)
    xp.subtract(scratch_g, scratch_dx, out=scratch_dx)
    xp.multiply(scratch_dx, std_inv, out=scratch_dx)

    xp.add(gx, scratch_dx, out=gx)


GraphEngine.register_op("layernorm", _layernorm_fwd_kernel, _layernorm_bwd_kernel)


def layernorm(x, gamma, beta, eps=1e-5):
    engine = x.engine
    if engine is None or not engine.is_tracing:
        xp = Device.xp
        x_data = x.data if hasattr(x, "data") else x
        gamma_data = gamma.data if hasattr(gamma, "data") else gamma
        beta_data = beta.data if hasattr(beta, "data") else beta

        mean = xp.mean(x_data, axis=-1, keepdims=True)
        var = xp.var(x_data, axis=-1, keepdims=True)
        std_inv = 1.0 / xp.sqrt(var + eps)
        x_norm = (x_data - mean) * std_inv
        out_data = gamma_data * x_norm + beta_data

        if hasattr(x, "requires_grad"):
            return Tensor(out_data, requires_grad=False)
        return out_data

    out = Tensor(shape=x.shape, requires_grad=x.requires_grad or gamma.requires_grad)

    stat_shape = tuple(list(x.shape)[:-1] + [1])
    s_x_norm = engine.alloc_scratch(x.shape)
    s_std_inv = engine.alloc_scratch(stat_shape)
    s_mean = engine.alloc_scratch(stat_shape)
    s_var = engine.alloc_scratch(stat_shape)

    s_g = engine.alloc_scratch(x.shape)
    s_dx = engine.alloc_scratch(x.shape)
    s_mean_g = engine.alloc_scratch(stat_shape)
    s_mean_g_xn = engine.alloc_scratch(stat_shape)

    engine.fwd_tape.append(
        {
            "op": "layernorm",
            "in_x_id": x.id,
            "in_gamma_id": gamma.id,
            "in_beta_id": beta.id,
            "out_id": out.id,
            "eps": eps,
            "scratch_x_norm_id": s_x_norm,
            "scratch_std_inv_id": s_std_inv,
            "scratch_mean_id": s_mean,
            "scratch_var_id": s_var,
        }
    )

    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "layernorm_bwd",
                "in_x_id": x.id,
                "in_gamma_id": gamma.id,
                "in_x_grad_id": x.id,
                "in_gamma_grad_id": gamma.id,
                "in_beta_grad_id": beta.id,
                "out_grad_id": out.id,
                "scratch_x_norm_id": s_x_norm,
                "scratch_std_inv_id": s_std_inv,
                "scratch_g_id": s_g,
                "scratch_dx_id": s_dx,
                "scratch_mean_g_id": s_mean_g,
                "scratch_mean_g_xnorm_id": s_mean_g_xn,
            }
        )
    return out


# 3. RMSNorm


def _rmsnorm_fwd_kernel(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    rms = engine.get_data_view(node["scratch_rms_id"])
    x_norm = engine.get_data_view(node["scratch_x_norm_id"])

    xp.square(x, out=x_norm)
    xp.mean(x_norm, axis=-1, keepdims=True, out=rms)
    xp.add(rms, node["eps"], out=rms)
    xp.sqrt(rms, out=rms)
    xp.divide(x, rms, out=x_norm)
    xp.multiply(x_norm, w, out=out_view)


def _rmsnorm_bwd_kernel(engine, node, xp):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])
    gx = engine.get_grad_view(node["in_x_grad_id"])
    gw = engine.get_grad_view(node["in_w_grad_id"])
    rms = engine.get_data_view(node["scratch_rms_id"])
    x_norm = engine.get_data_view(node["scratch_x_norm_id"])

    s_dx_norm = engine.get_data_view(node["scratch_dx_norm_id"])
    s_d_rms = engine.get_data_view(node["scratch_d_rms_id"])
    reduce_axes = tuple(range(x.ndim - 1))

    xp.multiply(out_grad, x_norm, out=s_dx_norm)
    xp.add(gw, xp.sum(s_dx_norm, axis=reduce_axes), out=gw)

    xp.multiply(out_grad, w, out=s_dx_norm)
    xp.multiply(s_dx_norm, x, out=gx)
    xp.divide(gx, rms, out=gx)
    xp.divide(gx, rms, out=gx)
    xp.multiply(gx, -1.0, out=gx)
    xp.sum(gx, axis=-1, keepdims=True, out=s_d_rms)
    gx.fill(0.0)

    xp.divide(s_dx_norm, rms, out=s_dx_norm)
    xp.multiply(s_d_rms, x, out=gx)
    xp.divide(gx, x.shape[-1], out=gx)
    xp.add(gx, s_dx_norm, out=gx)


GraphEngine.register_op("rmsnorm", _rmsnorm_fwd_kernel, _rmsnorm_bwd_kernel)


def rmsnorm(x, weight, eps=1e-6):
    engine = x.engine
    if engine is None or not engine.is_tracing:
        xp = Device.xp
        x_data = x.data if hasattr(x, "data") else x
        w_data = weight.data if hasattr(weight, "data") else weight

        rms = xp.sqrt(xp.mean(xp.square(x_data), axis=-1, keepdims=True) + eps)
        out_data = (x_data / rms) * w_data

        if hasattr(x, "requires_grad"):
            return Tensor(out_data, requires_grad=False)
        return out_data

    out = Tensor(shape=x.shape, requires_grad=x.requires_grad or weight.requires_grad)
    rms_shape = tuple(list(x.shape)[:-1] + [1])

    scratch_rms = engine.alloc_scratch(rms_shape)
    scratch_x_norm = engine.alloc_scratch(x.shape)
    scratch_dx_norm = engine.alloc_scratch(x.shape)
    scratch_d_rms = engine.alloc_scratch(rms_shape)

    engine.fwd_tape.append(
        {
            "op": "rmsnorm",
            "in_x_id": x.id,
            "in_w_id": weight.id,
            "out_id": out.id,
            "scratch_rms_id": scratch_rms,
            "scratch_x_norm_id": scratch_x_norm,
            "eps": eps,
        }
    )

    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "rmsnorm_bwd",
                "in_x_id": x.id,
                "in_w_id": weight.id,
                "in_x_grad_id": x.id,
                "in_w_grad_id": weight.id,
                "out_grad_id": out.id,
                "scratch_rms_id": scratch_rms,
                "scratch_x_norm_id": scratch_x_norm,
                "scratch_dx_norm_id": scratch_dx_norm,
                "scratch_d_rms_id": scratch_d_rms,
            }
        )
    return out


# 4. RoPE


def _rope_fwd_kernel(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    cos = node["cos"]
    sin = node["sin"]

    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    out_reshaped = out_view.reshape(x_reshaped.shape)

    out_reshaped[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
    out_reshaped[..., 1] = x_reshaped[..., 1] * cos + x_reshaped[..., 0] * sin


def _rope_bwd_kernel(engine, node, xp):
    out_grad = engine.get_grad_view(node["out_grad_id"])
    gx = engine.get_grad_view(node["in_x_grad_id"])
    cos = node["cos"]
    sin = node["sin"]

    grad_reshaped = out_grad.reshape(*out_grad.shape[:-1], -1, 2)
    gx_reshaped = gx.reshape(grad_reshaped.shape)

    # Inverse RoPE math mapped directly into the gradient views
    dx_0 = grad_reshaped[..., 0] * cos + grad_reshaped[..., 1] * sin
    dx_1 = grad_reshaped[..., 1] * cos - grad_reshaped[..., 0] * sin

    xp.add(gx_reshaped[..., 0], dx_0, out=gx_reshaped[..., 0])
    xp.add(gx_reshaped[..., 1], dx_1, out=gx_reshaped[..., 1])


GraphEngine.register_op("rope", _rope_fwd_kernel, _rope_bwd_kernel)


def apply_rope(x, freqs_cos, freqs_sin):
    engine = getattr(x, "engine", None)

    seq_len = x.shape[1]
    head_dim = x.shape[3]

    # Format freqs for broadcasting
    cos = freqs_cos[:seq_len].reshape(1, seq_len, 1, head_dim // 2)
    sin = freqs_sin[:seq_len].reshape(1, seq_len, 1, head_dim // 2)

    # EAGER FALLBACK
    if engine is None or not engine.is_tracing:
        xp = Device.xp
        x_data = x.data if hasattr(x, "data") else x
        x_reshaped = x_data.reshape(*x_data.shape[:-1], -1, 2)
        out = xp.empty_like(x_reshaped)
        out[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
        out[..., 1] = x_reshaped[..., 1] * cos + x_reshaped[..., 0] * sin
        out = out.reshape(x_data.shape)
        if hasattr(x, "requires_grad"):
            return Tensor(out, requires_grad=False)
        return out

    # COMPILED GRAPH
    out = Tensor(shape=x.shape, requires_grad=x.requires_grad)
    engine.fwd_tape.append(
        {"op": "rope", "in_x_id": x.id, "out_id": out.id, "cos": cos, "sin": sin}
    )

    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "rope_bwd",
                "in_x_id": x.id,
                "in_x_grad_id": x.id,
                "out_grad_id": out.id,
                "cos": cos,
                "sin": sin,
            }
        )
    return out
