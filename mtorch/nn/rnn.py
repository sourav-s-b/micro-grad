from mtorch.config import Device
from mtorch.nn.base import Module
from mtorch.tensor import Tensor


def _sigmoid(x):
    return 1.0 / (1.0 + Device.xp.exp(-Device.xp.clip(x, -500, 500)))


class LSTM(Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        bound = 1.0 / Device.xp.sqrt(input_dim + hidden_dim)

        self.W = Tensor(
            Device.xp.random.uniform(
                -bound, bound, (input_dim + hidden_dim, 4 * hidden_dim)
            ),
            requires_grad=True,
        )
        self.B = Tensor(Device.xp.zeros((4 * hidden_dim)), requires_grad=True)

    def __call__(self, x):
        assert isinstance(x, Tensor)
        b, t, _ = x.shape
        h_dim = self.hidden_dim

        h_states = Device.xp.zeros((b, t, h_dim))
        c_states = Device.xp.zeros((b, t, h_dim))

        cache = []

        h_prev = Device.xp.zeros((b, h_dim))
        c_prev = Device.xp.zeros((b, h_dim))

        for step in range(t):
            x_step = x.data[:, step, :]

            z = Device.xp.hstack((x_step, h_prev))

            gates_pre = z @ self.W.data + self.B.data

            f_pre = gates_pre[:, 0:h_dim]
            i_pre = gates_pre[:, h_dim : 2 * h_dim]
            c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
            o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

            f = _sigmoid(f_pre)
            i = _sigmoid(i_pre)

            c_tilde = Device.xp.tanh(c_pre)
            o = _sigmoid(o_pre)

            c_curr = f * c_prev + i * c_tilde
            tanh_c = Device.xp.tanh(c_curr)
            h_curr = o * tanh_c

            cache.append((z, f, i, c_tilde, o, c_curr, tanh_c, h_prev, c_prev))

            h_states[:, step, :] = h_curr
            c_states[:, step, :] = c_curr

            h_prev = h_curr

            c_prev = c_curr

        out = Tensor(h_states, (x, self.W, self.B), "LSTM")

        def _backward():

            if out.grad is None:
                return

            dW = Device.xp.zeros_like(self.W.data)
            dB = Device.xp.zeros_like(self.B.data)
            dx = Device.xp.zeros_like(x.data) if x.requires_grad else None

            dh_next = Device.xp.zeros((b, h_dim))
            dc_next = Device.xp.zeros((b, h_dim))

            for step in reversed(range(t)):

                z, f, i, c_tilde, o, c_curr, tanh_c, h_prev, c_prev = cache[step]

                dh = out.grad[:, step, :] + dh_next

                do_pre = dh * tanh_c * o * (1.0 - o)

                dc = dc_next + dh * o * (1.0 - tanh_c**2)

                df_pre = dc * c_prev * f * (1.0 - f)
                di_pre = dc * c_tilde * i * (1.0 - i)
                dc_tilde_pre = dc * i * (1.0 - c_tilde**2)

                da = Device.xp.hstack((df_pre, di_pre, dc_tilde_pre, do_pre))

                if self.W.requires_grad:
                    dW += z.T @ da
                if self.B.requires_grad:
                    dB += da.sum(axis=0)

                dz = da @ self.W.data.T

                if x.requires_grad:
                    dx[:, step, :] = dz[:, 0 : self.input_dim]

                dh_next = dz[:, self.input_dim :]
                dc_next = dc * f

            if self.W.requires_grad:
                self.W._accumulate_grad(dW)

            if self.B.requires_grad:
                self.B._accumulate_grad(dB)

            if x.requires_grad:
                x._accumulate_grad(dx)

        out._backward = _backward

        return out

    def step(self, x_t, state=None):

        if isinstance(x_t, Tensor):
            x_arr = x_t.data
        else:
            x_arr = x_t

        if len(x_arr.shape) == 3:
            x_step = x_arr[:, 0, :]
        else:
            x_step = x_arr

        b = x_step.shape[0]
        h_dim = self.hidden_dim

        if state is None:
            h_prev = Device.xp.zeros((b, h_dim))
            c_prev = Device.xp.zeros((b, h_dim))

        else:
            h_prev, c_prev = state

        z = Device.xp.hstack((x_step, h_prev))
        gates_pre = z @ self.W.data + self.B.data

        f_pre = gates_pre[:, 0:h_dim]
        i_pre = gates_pre[:, h_dim : 2 * h_dim]
        c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
        o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

        f = _sigmoid(f_pre)
        i = _sigmoid(i_pre)

        c_tilde = Device.xp.tanh(c_pre)
        o = _sigmoid(o_pre)

        c_curr = f * c_prev + i * c_tilde
        tanh_c = Device.xp.tanh(c_curr)
        h_curr = o * tanh_c

        return h_curr, (h_curr, c_curr)

    def parameters(self):
        return [self.W, self.B]
