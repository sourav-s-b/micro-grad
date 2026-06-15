from mtorch.config import Device
from mtorch.nn import Module
from mtorch.tensor_compiled import Tensor
import mtorch.nn.functional.rnn as F


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
        return F.lstm_sequence(x, self.W, self.B, self.hidden_dim)

    def step(self, x_step, state=None):
        if isinstance(x_step, Tensor):
            assert x_step.shape is not None
            if len(x_step.shape) == 3:
                x_step = x_step[:, 0, :]

        b = x_step.shape[0]
        h_dim = self.hidden_dim

        if state is None:
            h_prev = Tensor(Device.xp.zeros((b, h_dim)), requires_grad=False)
            c_prev = Tensor(Device.xp.zeros((b, h_dim)), requires_grad=False)
        else:
            h_prev, c_prev = state

        W_x = self.W[: self.input_dim, :]
        W_h = self.W[self.input_dim :, :]

        gates = x_step @ W_x + h_prev @ W_h + self.B

        f = gates[:, 0:h_dim].sigmoid()
        i = gates[:, h_dim : 2 * h_dim].sigmoid()
        c_tilde = gates[:, 2 * h_dim : 3 * h_dim].tanh()
        o = gates[:, 3 * h_dim : 4 * h_dim].sigmoid()

        c_curr = f * c_prev + i * c_tilde
        h_curr = o * c_curr.tanh()

        return h_curr, (h_curr, c_curr)

    def parameters(self):
        return [self.W, self.B]
