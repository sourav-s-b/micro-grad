from mtorch.tensor_compiled import GraphEngine

_dropout_threshold_kernel = None


def _get_dropout_kernel(xp):
    global _dropout_threshold_kernel
    if _dropout_threshold_kernel is None:
        _dropout_threshold_kernel = xp.ElementwiseKernel(
            in_params="T rand_val, float32 p, float32 scale",
            out_params="T mask_out",
            operation="""
                if (rand_val >= p) {
                    mask_out = (T)scale;
                } else {
                    mask_out = (T)0.0;
                }
            """,
            name="fused_dropout_threshold",
        )
    return _dropout_threshold_kernel


def _dropout_fwd(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    mask = engine.get_data_view(node["scratch_mask_id"])
    p = node["p"]
    scale = 1.0 / (1.0 - p)

    mask[...] = xp.random.rand(*x.shape)

    if hasattr(xp, "ElementwiseKernel"):
        kernel = _get_dropout_kernel(xp)
        kernel(mask, p, scale, mask)
    else:
        mask[...] = (mask >= p).astype(xp.float32) * scale

    xp.multiply(x, mask, out=out_view)


def _dropout_bwd(engine, node, xp):
    out_grad = engine.get_grad_view(node["out_grad_id"])
    gx = engine.get_grad_view(node["in_x_grad_id"])

    mask = engine.get_data_view(node["scratch_mask_id"])

    if gx is not None:
        scratch_dx = engine.get_data_view(node["scratch_dx_id"])
        xp.multiply(out_grad, mask, out=scratch_dx)
        xp.add(gx, scratch_dx, out=gx)
