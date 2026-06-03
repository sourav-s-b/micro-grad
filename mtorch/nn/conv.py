from mtorch.config import Device
from mtorch.nn.base import Module

from mtorch.tensor import Tensor


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
        bound = 1.0 / Device.xp.sqrt(in_channels * kh * kw)

        self.W = Tensor(
            Device.xp.random.uniform(
                -bound, bound, size=(out_channels, in_channels, kh, kw)
            ),
            requires_grad=True,
        )

        self.B = Tensor(Device.xp.zeros((out_channels,)), requires_grad=True)

    def __call__(self, x):
        b, c, h, w = x.shape
        kh, kw = self.kernel_size
        assert c == self.in_channels

        x_data = x.data
        if self.padding > 0:
            x_data = Device.xp.pad(
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

        out_data = Device.xp.zeros((b, self.out_channels, out_h, out_w))

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

        out_data += self.B.data[
            Device.xp.newaxis, :, Device.xp.newaxis, Device.xp.newaxis
        ]

        out_tensor = Tensor(out_data, (x, self.W, self.B), "Conv2D")

        def _backward():
            if out_tensor.grad is None:
                return

            x_data_pad = x.data
            if self.padding > 0:
                x_data_pad = Device.xp.pad(
                    x.data,
                    (
                        (0, 0),
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    ),
                )

            dx_pad = Device.xp.zeros_like(x_data_pad) if x.requires_grad else None
            dw = Device.xp.zeros_like(self.W.data) if self.W.requires_grad else None

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

                        dw += Device.xp.einsum("bo,bchw->ochw", g, patch)

                    if x.requires_grad:

                        dx = Device.xp.einsum("bo,ochw->bchw", g, self.W.data)
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

        out_data = Device.xp.zeros((b, c, out_h, out_w))

        for y in range(out_h):
            for x_idx in range(out_w):
                ys = y * self.stride
                xs = x_idx * self.stride

                patch = x.data[:, :, ys : ys + kh, xs : xs + kw]
                out_data[:, :, y, x_idx] = Device.xp.max(patch, axis=(2, 3))

        out = Tensor(out_data, (x,), "MaxPool2D", requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if not x.requires_grad:
                return

            if x.grad is None:
                x.grad = Device.xp.zeros_like(x.data)
            for y in range(out_h):
                for x_idx in range(out_w):
                    ys = y * self.stride
                    xs = x_idx * self.stride

                    patch = x.data[:, :, ys : ys + kh, xs : xs + kw]
                    max_val = out.data[:, :, y, x_idx].reshape(b, c, 1, 1)
                    mask = patch == max_val

                    mask = mask / (
                        Device.xp.sum(mask, axis=(2, 3), keepdims=True) + 1e-8
                    )

                    g_slice = out.grad[:, :, y, x_idx].reshape(b, c, 1, 1)
                    x.grad[:, :, ys : ys + kh, xs : xs + kw] += mask * g_slice

        out._backward = _backward
        return out
