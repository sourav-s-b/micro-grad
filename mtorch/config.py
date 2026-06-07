import types
import numpy as np


class Device:
    xp: types.ModuleType = np
    device = "cpu"
    no_grad = False


def set_device(name):

    if name == "cuda":
        try:
            import cupy as cp

            cp.cuda.Device(0).use()
            Device.xp = cp
            Device.device = "cuda"
            print(f"Backend Device: changed to {Device.device}")
        except Exception as e:
            print(f"Warning: CUDA requested but initialization failed ({e}).")
            print("Falling back to NumPy / CPU mode.")
            Device.xp = np
            Device.device = "cpu"


def to_cpu(array):
    if Device.device == "cuda" and hasattr(array, "get"):
        return array.get()
    return array


class no_grad:

    def __enter__(self):
        self.prev = Device.no_grad
        Device.no_grad = True

    def __exit__(self, exc_type, exc_value, tracback):
        Device.no_grad = self.prev
