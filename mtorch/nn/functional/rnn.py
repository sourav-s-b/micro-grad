from mtorch.config import Device
from mtorch.tensor_compiled import Tensor, GraphEngine


def _sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))


def _lstm_fwd_kernel(engine, node, xp, out_view):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    b = engine.get_data_view(node["in_b_id"])

    h_states = engine.get_data_view(node["scratch_h_id"])
    c_states = engine.get_data_view(node["scratch_c_id"])
    gates_cache = engine.get_data_view(node["scratch_gates_id"])

    # Pre-allocated loop variables
    z = engine.get_data_view(node["scratch_z_id"])
    h_prev = engine.get_data_view(node["scratch_h_prev_id"])
    c_prev = engine.get_data_view(node["scratch_c_prev_id"])

    h_prev.fill(0.0)
    c_prev.fill(0.0)

    batch, steps, in_dim = x.shape
    h_dim = h_states.shape[-1]

    for t in range(steps):
        # Write into pre-allocated z buffer instead of hstack allocating
        z[:, :in_dim] = x[:, t, :]
        z[:, in_dim:] = h_prev

        # Matmul and bias directly into the gates cache
        gates = gates_cache[:, t, :]
        xp.matmul(z, w, out=gates)
        xp.add(gates, b, out=gates)

        f = _sigmoid(gates[:, 0:h_dim], xp)
        i = _sigmoid(gates[:, h_dim : 2 * h_dim], xp)
        c_tilde = xp.tanh(gates[:, 2 * h_dim : 3 * h_dim])
        o = _sigmoid(gates[:, 3 * h_dim : 4 * h_dim], xp)

        # In-place cell state update
        c_curr = c_states[:, t, :]
        xp.multiply(f, c_prev, out=c_curr)
        c_curr += i * c_tilde

        # In-place hidden state update
        h_curr = h_states[:, t, :]
        xp.tanh(c_curr, out=h_curr)
        xp.multiply(o, h_curr, out=h_curr)

        # Update previous state buffers
        h_prev[:] = h_curr[:]
        c_prev[:] = c_curr[:]

    out_view[:] = h_states


def _lstm_bwd_kernel(engine, node, xp):
    x = engine.get_data_view(node["in_x_id"])
    w = engine.get_data_view(node["in_w_id"])
    out_grad = engine.get_grad_view(node["out_grad_id"])

    gx = engine.get_grad_view(node["in_x_grad_id"])
    gw = engine.get_grad_view(node["in_w_grad_id"])
    gb = engine.get_grad_view(node["in_b_grad_id"])

    h_states = engine.get_data_view(node["scratch_h_id"])
    c_states = engine.get_data_view(node["scratch_c_id"])
    gates_cache = engine.get_data_view(node["scratch_gates_id"])

    # Retrieve all pre-allocated backward scratch buffers
    dh_next = engine.get_data_view(node["scratch_dh_next_id"])
    dc_next = engine.get_data_view(node["scratch_dc_next_id"])
    dh_next.fill(0.0)
    dc_next.fill(0.0)

    scratch_dh = engine.get_data_view(node["scratch_dh_id"])
    scratch_dc = engine.get_data_view(node["scratch_dc_id"])
    scratch_z = engine.get_data_view(node["scratch_z_id"])
    scratch_dgates = engine.get_data_view(node["scratch_dgates_id"])
    scratch_gw_delta = engine.get_data_view(node["scratch_gw_delta_id"])
    scratch_tanh_c = engine.get_data_view(node["scratch_tanh_c_id"])

    zero_h = engine.get_data_view(node["scratch_zero_h_id"])
    zero_h.fill(0.0)

    batch, steps, in_dim = x.shape
    h_dim = h_states.shape[-1]

    for t in reversed(range(steps)):
        # dh = out_grad[:, t, :] + dh_next
        xp.add(out_grad[:, t, :], dh_next, out=scratch_dh)

        c_curr = c_states[:, t, :]
        c_prev = c_states[:, t - 1, :] if t > 0 else zero_h

        gates = gates_cache[:, t, :]
        f = _sigmoid(gates[:, 0:h_dim], xp)
        i = _sigmoid(gates[:, h_dim : 2 * h_dim], xp)
        c_tilde = xp.tanh(gates[:, 2 * h_dim : 3 * h_dim])
        o = _sigmoid(gates[:, 3 * h_dim : 4 * h_dim], xp)

        xp.tanh(c_curr, out=scratch_tanh_c)

        # Map dgates slices to specific gradients
        df = scratch_dgates[:, 0:h_dim]
        di = scratch_dgates[:, h_dim : 2 * h_dim]
        dc_tilde = scratch_dgates[:, 2 * h_dim : 3 * h_dim]
        do = scratch_dgates[:, 3 * h_dim : 4 * h_dim]

        # do = dh * tanh_c
        xp.multiply(scratch_dh, scratch_tanh_c, out=do)

        # dc = dh * o * (1.0 - tanh_c**2) + dc_next
        xp.square(scratch_tanh_c, out=scratch_dc)
        xp.subtract(1.0, scratch_dc, out=scratch_dc)
        xp.multiply(scratch_dc, o, out=scratch_dc)
        xp.multiply(scratch_dc, scratch_dh, out=scratch_dc)
        xp.add(scratch_dc, dc_next, out=scratch_dc)

        # Compute pre-activation gate gradients
        xp.multiply(scratch_dc, i, out=dc_tilde)
        xp.multiply(scratch_dc, c_tilde, out=di)
        xp.multiply(scratch_dc, c_prev, out=df)

        # Apply derivative of activations directly into slices
        # do *= o * (1 - o)
        do *= o * (1.0 - o)
        # dc_tilde *= (1 - c_tilde**2)
        dc_tilde *= 1.0 - xp.square(c_tilde)
        # di *= i * (1 - i)
        di *= i * (1.0 - i)
        # df *= f * (1 - f)
        df *= f * (1.0 - f)

        # Reconstruct z
        scratch_z[:, :in_dim] = x[:, t, :]
        scratch_z[:, in_dim:] = h_states[:, t - 1, :] if t > 0 else zero_h

        # Accumulate weight gradients without temporary array creation
        xp.matmul(scratch_z.T, scratch_dgates, out=scratch_gw_delta)
        xp.add(gw, scratch_gw_delta, out=gw)

        # Accumulate bias gradients
        xp.add(gb, xp.sum(scratch_dgates, axis=0), out=gb)

        # Pass gradients to next timestep
        dh_next[:] = scratch_dgates[:, in_dim:]
        gx[:, t, :] += scratch_dgates[:, :in_dim]
        xp.multiply(scratch_dc, f, out=dc_next)


GraphEngine.register_op("lstm_sequence", _lstm_fwd_kernel, _lstm_bwd_kernel)


def lstm_sequence(x, w, b, hidden_dim):
    engine = getattr(x, "engine", None)

    # EAGER MODE FALLBACK
    if engine is None or not engine.is_tracing:
        xp = Device.xp
        x_data = x.data if hasattr(x, "data") else x
        w_data = w.data if hasattr(w, "data") else w
        b_data = b.data if hasattr(b, "data") else b

        batch, steps, in_dim = x_data.shape
        h_states = xp.zeros((batch, steps, hidden_dim))
        c_prev = xp.zeros((batch, hidden_dim))
        h_prev = xp.zeros((batch, hidden_dim))

        for t in range(steps):
            z = xp.hstack((x_data[:, t, :], h_prev))
            gates = z @ w_data + b_data

            f = _sigmoid(gates[:, 0:hidden_dim], xp)
            i = _sigmoid(gates[:, hidden_dim : 2 * hidden_dim], xp)
            c_tilde = xp.tanh(gates[:, 2 * hidden_dim : 3 * hidden_dim])
            o = _sigmoid(gates[:, 3 * hidden_dim : 4 * hidden_dim], xp)

            c_prev = f * c_prev + i * c_tilde
            h_prev = o * xp.tanh(c_prev)
            h_states[:, t, :] = h_prev

        if hasattr(x, "requires_grad"):
            return Tensor(h_states, requires_grad=False)
        return h_states

    # COMPILED GRAPH MODE
    batch, steps, in_dim = x.shape
    out_shape = (batch, steps, hidden_dim)

    requires_grad = x.requires_grad or w.requires_grad or b.requires_grad
    out = Tensor(shape=out_shape, requires_grad=requires_grad)

    # Allocations for Forward Pass
    scratch_h = engine.alloc_scratch((batch, steps, hidden_dim))
    scratch_c = engine.alloc_scratch((batch, steps, hidden_dim))
    scratch_gates = engine.alloc_scratch((batch, steps, 4 * hidden_dim))
    scratch_z = engine.alloc_scratch((batch, in_dim + hidden_dim))
    scratch_h_prev = engine.alloc_scratch((batch, hidden_dim))
    scratch_c_prev = engine.alloc_scratch((batch, hidden_dim))

    engine.fwd_tape.append(
        {
            "op": "lstm_sequence",
            "in_x_id": x.id,
            "in_w_id": w.id,
            "in_b_id": b.id,
            "out_id": out.id,
            "scratch_h_id": scratch_h,
            "scratch_c_id": scratch_c,
            "scratch_gates_id": scratch_gates,
            "scratch_z_id": scratch_z,
            "scratch_h_prev_id": scratch_h_prev,
            "scratch_c_prev_id": scratch_c_prev,
        }
    )

    if out.requires_grad:
        # Extra allocations specifically for the backward loop
        scratch_dh_next = engine.alloc_scratch((batch, hidden_dim))
        scratch_dc_next = engine.alloc_scratch((batch, hidden_dim))
        scratch_dh = engine.alloc_scratch((batch, hidden_dim))
        scratch_dc = engine.alloc_scratch((batch, hidden_dim))
        scratch_dgates = engine.alloc_scratch((batch, 4 * hidden_dim))
        scratch_gw_delta = engine.alloc_scratch((in_dim + hidden_dim, 4 * hidden_dim))
        scratch_tanh_c = engine.alloc_scratch((batch, hidden_dim))
        scratch_zero_h = engine.alloc_scratch((batch, hidden_dim))

        engine.bwd_tape.append(
            {
                "op": "lstm_sequence_bwd",
                "in_x_id": x.id,
                "in_w_id": w.id,
                "in_b_id": b.id,
                "in_x_grad_id": x.id,
                "in_w_grad_id": w.id,
                "in_b_grad_id": b.id,
                "out_grad_id": out.id,
                "scratch_h_id": scratch_h,
                "scratch_c_id": scratch_c,
                "scratch_gates_id": scratch_gates,
                "scratch_dh_next_id": scratch_dh_next,
                "scratch_dc_next_id": scratch_dc_next,
                "scratch_dh_id": scratch_dh,
                "scratch_dc_id": scratch_dc,
                "scratch_z_id": scratch_z,
                "scratch_dgates_id": scratch_dgates,
                "scratch_gw_delta_id": scratch_gw_delta,
                "scratch_tanh_c_id": scratch_tanh_c,
                "scratch_zero_h_id": scratch_zero_h,
            }
        )

    return out
