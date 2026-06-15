from mtorch.config import Device
from mtorch.tensor_compiled import Tensor, GraphEngine


def _conv2d_fwd_kernel(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    b = engine.get_data_view(node["in_b_id"])
    p, s = node["padding"], node["stride"]

    if p > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

    b_sz, c, hp, wp = x.shape
    out_c, _, kh, kw = w.shape
    out_h, out_w = out_view.shape[2], out_view.shape[3]

    shape = (b_sz, c, out_h, out_w, kh, kw)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * s,
        x.strides[3] * s,
        x.strides[2],
        x.strides[3],
    )

    windows = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    out_view[:] = xp.einsum("bchwkl, ockl->bohw", windows, w) + b.reshape(1, -1, 1, 1)


def _conv2d_bwd_kernel(engine, node, xp):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])

    gx = engine.get_grad_view(node["in_x_grad_id"])
    gw = engine.get_grad_view(node["in_w_grad_id"])
    gb = engine.get_grad_view(node["in_b_grad_id"])

    p, s = node["padding"], node["stride"]

    b_sz, c, hp, wp = x.shape
    out_c, _, kh, kw = w.shape
    out_h, out_w = out_grad.shape[2], out_grad.shape[3]

    # 1. Gradient with respect to bias
    xp.add(gb, xp.sum(out_grad, axis=(0, 2, 3)), out=gb)

    # 2. Gradient with respect to weights
    x_pad = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p > 0 else x
    shape_w = (b_sz, c, out_h, out_w, kh, kw)
    strides_w = (
        x_pad.strides[0],
        x_pad.strides[1],
        x_pad.strides[2] * s,
        x_pad.strides[3] * s,
        x_pad.strides[2],
        x_pad.strides[3],
    )
    windows_w = xp.lib.stride_tricks.as_strided(x_pad, shape=shape_w, strides=strides_w)
    xp.add(gw, xp.einsum("bohw, bchwkl -> ockl", out_grad, windows_w), out=gw)

    # 3. Gradient with respect to input (Transposed Convolution)
    # Dilate out_grad by stride
    dilated_h = (out_h - 1) * s + 1
    dilated_w = (out_w - 1) * s + 1
    dilated_out_grad = xp.zeros(
        (b_sz, out_c, dilated_h, dilated_w), dtype=out_grad.dtype
    )
    dilated_out_grad[:, :, ::s, ::s] = out_grad

    # Flip weights spatially and swap input/output channels
    w_flipped = xp.flip(w, axis=(2, 3))
    w_flipped = xp.swapaxes(w_flipped, 0, 1)  # shape: (in_c, out_c, kh, kw)

    # Pad the dilated gradient
    pad_h = kh - 1
    pad_w = kw - 1
    dilated_padded = xp.pad(
        dilated_out_grad, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
    )

    dp_h, dp_w = dilated_padded.shape[2:]
    shape_dx = (b_sz, out_c, dp_h - kh + 1, dp_w - kw + 1, kh, kw)
    strides_dx = (
        dilated_padded.strides[0],
        dilated_padded.strides[1],
        dilated_padded.strides[2],
        dilated_padded.strides[3],
        dilated_padded.strides[2],
        dilated_padded.strides[3],
    )

    # Apply convolution to get perfectly accumulated dx
    windows_dx = xp.lib.stride_tricks.as_strided(
        dilated_padded, shape=shape_dx, strides=strides_dx
    )
    dx_pad = xp.einsum("bohwkl, iokl -> bihw", windows_dx, w_flipped)

    # Remove padding to get final gx
    if p > 0:
        xp.add(gx, dx_pad[:, :, p:-p, p:-p], out=gx)
    else:
        xp.add(gx, dx_pad, out=gx)


GraphEngine.register_op("conv2d", _conv2d_fwd_kernel, _conv2d_bwd_kernel)


def conv2d(x, w, b, stride, padding):
    engine = getattr(x, "engine", None)

    # EAGER MODE FALLBACK
    if engine is None or not engine.is_tracing:
        xp = Device.xp
        x_data = x.data if hasattr(x, "data") else x
        w_data = w.data if hasattr(w, "data") else w
        b_data = b.data if hasattr(b, "data") else b

        if padding > 0:
            x_data = xp.pad(
                x_data, ((0, 0), (0, 0), (padding, padding), (padding, padding))
            )

        b_sz, c, hp, wp = x_data.shape
        out_c, _, kh, kw = w_data.shape
        out_h = (hp - kh) // stride + 1
        out_w = (wp - kw) // stride + 1

        shape = (b_sz, c, out_h, out_w, kh, kw)
        strides = (
            x_data.strides[0],
            x_data.strides[1],
            x_data.strides[2] * stride,
            x_data.strides[3] * stride,
            x_data.strides[2],
            x_data.strides[3],
        )

        windows = xp.lib.stride_tricks.as_strided(x_data, shape=shape, strides=strides)
        out_data = xp.einsum("bchwkl, ockl->bohw", windows, w_data) + b_data.reshape(
            1, -1, 1, 1
        )

        if hasattr(x, "requires_grad"):
            return Tensor(out_data, requires_grad=False)
        return out_data

    # COMPILED GRAPH MODE
    batch, in_channels, h, w_dim = x.shape
    out_channels, _, kh, kw = w.shape

    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w_dim + 2 * padding - kw) // stride + 1
    out_shape = (batch, out_channels, out_h, out_w)

    requires_grad = x.requires_grad or w.requires_grad or b.requires_grad
    out = Tensor(shape=out_shape, requires_grad=requires_grad)

    engine.fwd_tape.append(
        {
            "op": "conv2d",
            "in_x_id": x.id,
            "in_w_id": w.id,
            "in_b_id": b.id,
            "out_id": out.id,
            "stride": stride,
            "padding": padding,
        }
    )

    if out.requires_grad:
        engine.bwd_tape.append(
            {
                "op": "conv2d_bwd",
                "in_x_id": x.id,
                "in_w_id": w.id,
                "in_b_id": b.id,
                "in_x_grad_id": x.id,
                "in_w_grad_id": w.id,
                "in_b_grad_id": b.id,
                "out_grad_id": out.id,
                "stride": stride,
                "padding": padding,
            }
        )

    return out
