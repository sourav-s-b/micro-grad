import numpy as np

from mtorch.tensor import Tensor


class Module:

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        params = []

        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
        return params


class Linear(Module):

    def __init__(self, nin, nout):
        super().__init__()

        bound = 1 / np.sqrt(nin)

        self.W = Tensor(
            np.random.uniform(-bound, bound, (nin, nout)), requires_grad=True
        )
        self.B = Tensor(np.zeros((1, nout)), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B]


class Dropout(Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if self.training and self.p > 0:
            mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            mask_np = Tensor(mask, requires_grad=False)
            return x * mask_np
        return x


class Sequential(Module):

    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()


class Conv2D(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        bound = 1.0 / np.sqrt(in_channels * kh * kw)

        self.W = Tensor(
            np.random.uniform(-bound, bound, size=(out_channels, in_channels, kh, kw)),
            requires_grad=True,
        )

        self.B = Tensor(np.zeros((out_channels,)), requires_grad=True)

    def __call__(self, x):
        b, c, h, w = x.shape
        kh, kw = self.kernel_size
        assert c == self.in_channels

        x_data = x.data
        if self.padding > 0:
            x_data = np.pad(
                x_data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
            )

        out_h = (h + 2 * self.padding - kh) // self.stride + 1
        out_w = (w + 2 * self.padding - kw) // self.stride + 1

        out_data = np.zeros((b, self.out_channels, out_h, out_w))

        for y in range(out_h):
            for xi in range(out_w):
                patch = x_data[
                    :,
                    :,
                    y * self.stride : y * self.stride + kh,
                    xi * self.stride : xi * self.stride + kw,
                ]

                out_data[:, :, y, xi] = (
                    patch.reshape(b, 1, -1)
                    * self.W.data.reshape(1, self.out_channels, -1)
                ).sum(axis=2)

        out_data += self.B.data[np.newaxis, :, np.newaxis, np.newaxis]

        out_tensor = Tensor(out_data, (x, self.W, self.B), "Conv2D")

        def _backward():
            if out_tensor.grad is None:
                return

            x_data_pad = x.data
            if self.padding > 0:
                x_data_pad = np.pad(
                    x.data,
                    (
                        (0, 0),
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    ),
                )

            dx_pad = np.zeros_like(x_data_pad) if x.requires_grad else None
            dw = np.zeros_like(self.W.data) if self.W.requires_grad else None

            for y in range(out_h):
                for xi in range(out_w):
                    g = out_tensor.grad[:, :, y, xi]
                    patch = x_data_pad[
                        :,
                        :,
                        y * self.stride : y * self.stride + kh,
                        xi * self.stride : xi * self.stride + kw,
                    ]

                    if self.W.requires_grad:

                        dw += np.einsum("bo,bchw->ochw", g, patch)

                    if x.requires_grad:

                        dx = np.einsum("bo,ochw->bchw", g, self.W.data)
                        dx_pad[
                            :,
                            :,
                            y * self.stride : y * self.stride + kh,
                            xi * self.stride : xi * self.stride + kw,
                        ] += dx

            if self.W.requires_grad:
                self.W._accumulate_grad(dw)

            if x.requires_grad:
                if self.padding > 0:
                    p = self.padding
                    x._accumulate_grad(dx_pad[:, :, p:-p, p:-p])
                else:
                    x._accumulate_grad(dx_pad)

            if self.B.requires_grad:
                self.B._accumulate_grad(out_tensor.grad.sum(axis=(0, 2, 3)))

        out_tensor._backward = _backward
        return out_tensor

    def parameters(self):
        return [self.W, self.B]


class MaxPool2D(Module):

    def __init__(self, kernel_size=2, stride=2):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):

        b, c, h, w = x.shape
        kh = kw = self.kernel_size

        out_h = (h - kh) // self.stride + 1
        out_w = (w - kw) // self.stride + 1

        out_data = np.zeros((b, c, out_h, out_w))

        for y in range(out_h):
            for x_idx in range(out_w):
                ys = y * self.stride
                xs = x_idx * self.stride

                patch = x.data[:, :, ys : ys + kh, xs : xs + kw]
                out_data[:, :, y, x_idx] = np.max(patch, axis=(2, 3))

        out = Tensor(out_data, (x,), "MaxPool2D", requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if not x.requires_grad:
                return

            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            for y in range(out_h):
                for x_idx in range(out_w):
                    ys = y * self.stride
                    xs = x_idx * self.stride

                    patch = x.data[:, :, ys : ys + kh, xs : xs + kw]
                    max_val = out.data[:, :, y, x_idx].reshape(b, c, 1, 1)
                    mask = patch == max_val

                    mask = mask / (np.sum(mask, axis=(2, 3), keepdims=True) + 1e-8)

                    g_slice = out.grad[:, :, y, x_idx].reshape(b, c, 1, 1)
                    x.grad[:, :, ys : ys + kh, xs : xs + kw] += mask * g_slice

        out._backward = _backward
        return out

class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim):

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.w = Tensor(np.random.randn(num_embeddings, embedding_dim) *0.01, requires_grad=True)

    def __call__(self, indices):

        idx_array = indices.data.astype(int)

        out_data = self.w.data[idx_array]

        out = Tensor(out_data, (self.w, indices), "Embedding", requires_grad=self.w.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if out.requires_grad is None or not self.w.requires_grad: return

            dW = np.zeros_like(self.w.data)

            flat_idx = idx_array.reshape(-1)
            flat_grad = out.grad.reshape(-1, self.embedding_dim)

            np.add.at(dW, flat_idx, flat_grad)
            self.w._accumulate_grad(dW)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.w]

class LayerNorm(Module):

    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
    
    def __call__(self, x):
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis= -1, keepdims=True)

        x_norm = (x.data - mean)/ np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data

        out = Tensor(out_data, (x, self.gamma, self.beta), 'LayerNorm', requires_grad=True)

        def _backward():
            if out.grad is None:
                return
            
            if self.gamma.requires_grad:
                self.gamma._accumulate_grad((out.grad * x_norm).sum(axis = (0,1)))
            if self.beta.requires_grad:
                self.beta._accumulate_grad((out.grad * x_norm).sum(axis = (0,1)))
            if x.requires_grad:

                N = x.data.shape[-1]
                g = out.grad * self.gamma.data

                dx = (1.0 / np.sqrt(var + self.eps)) * (
                        g - (g.mean(axis=-1, keepdims = True)) - x_norm * (g * x_norm).mean(axis=-1, keepdims=True)
                    )
                x._accumulate_grad(dx)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTM(Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        bound = 1.0 / np.sqrt(input_dim + hidden_dim)

        self.W = Tensor(
            np.random.uniform(-bound, bound, (input_dim + hidden_dim, 4 * hidden_dim)),
            requires_grad=True,
        )
        self.B = Tensor(np.zeros((4 * hidden_dim)), requires_grad=True)

    def __call__(self, x):
        assert isinstance(x, Tensor)
        b, t, _ = x.shape
        h_dim = self.hidden_dim

        h_states = np.zeros((b, t, h_dim))
        c_states = np.zeros((b, t, h_dim))

        cache = []

        h_prev = np.zeros((b, h_dim))
        c_prev = np.zeros((b, h_dim))

        for step in range(t):
            x_step = x.data[:, step, :]

            z = np.hstack((x_step, h_prev))

            gates_pre = z @ self.W.data + self.B.data

            f_pre = gates_pre[:, 0:h_dim]
            i_pre = gates_pre[:, h_dim : 2 * h_dim]
            c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
            o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

            f = _sigmoid(f_pre)
            i = _sigmoid(i_pre)

            c_tilde = np.tanh(c_pre)
            o = _sigmoid(o_pre)

            c_curr = f * c_prev + i * c_tilde
            tanh_c = np.tanh(c_curr)
            h_curr = o * tanh_c

            cache.append((z, f, i, c_tilde, o, c_curr, tanh_c, h_prev, c_prev))

            h_states[:, step, :] = h_curr
            c_states[:, step, :] = c_curr

            h_prev = h_curr

            c_prev = c_curr

        out = Tensor(h_states, (x, self.W, self.B) , 'LSTM')

        def _backward():

            if out.grad is None: return
            
            dW = np.zeros_like(self.W.data) 
            dB = np.zeros_like(self.B.data) 
            dx = np.zeros_like(x.data) if x.requires_grad else None

            dh_next = np.zeros((b, h_dim))
            dc_next = np.zeros((b, h_dim))

            for step in reversed(range(t)):

                z , f , i , c_tilde, o , c_curr, tanh_c ,h_prev, c_prev = cache[step]

                dh = out.grad[:, step, :] + dh_next

                do_pre = dh * tanh_c * o * (1.0 - o)

                dc = dc_next + dh * o * (1.0 - tanh_c**2)

                df_pre = dc * c_prev * f * (1.0 - f)
                di_pre = dc * c_tilde * i * (1.0 - i )
                dc_tilde_pre = dc * i * (1.0 - c_tilde**2)

                da = np.hstack((df_pre, di_pre, dc_tilde_pre, do_pre))

                if self.W.requires_grad:
                    dW += z.T @ da
                if self.B.requires_grad:
                    dB += da.sum(axis=0)

                dz = da @ self.W.data.T

                if x.requires_grad:
                    dx[: , step, :] = dz[: , 0:self.input_dim]

                dh_next = dz[: , self.input_dim:]
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
            x_arr  = x_t.data
        else:
            x_arr = x_t

        if len(x_arr.shape) == 3:
            x_step = x_arr[:, 0,:]
        else:
            x_step = x_arr

       
        b = x_step.shape[0]
        h_dim = self.hidden_dim

        if state is None:
            h_prev = np.zeros((b,h_dim))
            c_prev = np.zeros((b,h_dim))

        else:
            h_prev , c_prev = state




        z = np.hstack((x_step, h_prev))
        gates_pre = z @ self.W.data + self.B.data

        f_pre = gates_pre[:, 0:h_dim]
        i_pre = gates_pre[:, h_dim : 2 * h_dim]
        c_pre = gates_pre[:, 2 * h_dim : 3 * h_dim]
        o_pre = gates_pre[:, 3 * h_dim : 4 * h_dim]

        f = _sigmoid(f_pre)
        i = _sigmoid(i_pre)

        c_tilde = np.tanh(c_pre)
        o = _sigmoid(o_pre)

        c_curr = f * c_prev + i * c_tilde
        tanh_c = np.tanh(c_curr)
        h_curr = o * tanh_c

        return h_curr, (h_curr, c_curr)

    def parameters(self):
        return [self.W, self.B]
