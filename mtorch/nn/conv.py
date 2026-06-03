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
        b, c, hp, wp = x_data.shape
        out_h = (hp - kh) // self.stride + 1
        out_w = (wp - kw) // self.stride + 1

        shape = (b, c, out_h, out_w, kh, kw)
        strides = (
            x_data.strides[0],
            x_data.strides[1],
            x_data.strides[2] * self.stride,
            x_data.strides[3] * self.stride,
            x_data.strides[2],
            x_data.strides[3],
        )

        windows = Device.xp.lib.stride_tricks.as_strided(
            x_data, shape=shape, strides=strides
        )

        out_data = Device.xp.einsum("bchwkl, ockl->bohw", windows, self.W.data)

        out_data += self.B.data.reshape(1, -1, 1, 1)

        out_tensor = Tensor(out_data, (x, self.W, self.B), "Conv2D")

        def _backward():
            if out_tensor.grad is None:
                return

            if self.W.requires_grad:
                dw = Device.xp.einsum("bohw,bchwkl->ockl", out_tensor.grad, windows)
                self.W._accumulate_grad(dw)

            if self.B.requires_grad:
                self.B._accumulate_grad(out_tensor.grad.sum(axis=(0, 2, 3)))

            if x.requires_grad:

                dx_pad = Device.xp.zeros_like(x_data)
                g = out_tensor.grad
                w_data = self.W.data

                for y in range(out_h):
                    for xi in range(out_w):
                        dx_patch = Device.xp.einsum(
                            "bo,ockl->bckl", g[:, :, y, xi], w_data
                        )
                        dx_pad[
                            :,
                            :,
                            y * self.stride : y * self.stride + kh,
                            xi * self.stride : xi * self.stride + kw,
                        ] += dx_patch
                if self.padding > 0:
                    p = self.padding
                    x._accumulate_grad(dx_pad[:, :, p:-p, p:-p])
                else:
                    x._accumulate_grad(dx_pad)

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

        x_reshaped = x.data.reshape(b, c, h // kh, kh, w // kw, kw)

        out_data = x_reshaped.max(axis=(3, 5))

        out = Tensor(out_data, (x,), "MaxPool2D", requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if not x.requires_grad:
                return

            if x.grad is None:
                x.grad = Device.xp.zeros_like(x.data)

            out_repeated = Device.xp.repeat(
                Device.xp.repeat(out.data, kh, axis=2), kw, axis=3
            )

            g_repeated = Device.xp.repeat(
                Device.xp.repeat(out.grad, kh, axis=2), kw, axis=3
            )

            mask = (x.data == out_repeated).astype(Device.xp.float32)

            dx = mask * g_repeated
            x._accumulate_grad(dx)

        out._backward = _backward
        return out
