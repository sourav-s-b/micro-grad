from mtorch.config import Device
from mtorch.nn import Module
from mtorch.tensor_compiled import Tensor
import mtorch.nn.functional.cnn as F


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
        return F.conv2d(x, self.W, self.B, self.stride, self.padding)

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

        x_reshaped = x.reshape(b, c, h // kh, kh, w // kw, kw)
        return x_reshaped.max(axis=(3, 5))
