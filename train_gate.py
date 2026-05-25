import numpy as np
import torch

from mtorch.nn.modules import Linear
from mtorch.tensor import Tensor

print("=== STARTING VTORCH ENGINE VERIFICATION ===\n")

# Fix the random seed for reproducibility
np.random.seed(42)

# 1. Create a dummy dataset (Batch size = 2, Input features = 3)
raw_X = np.random.randn(2, 3)

# 2. Setup VNY / VTorch Run
X_v = Tensor(raw_X)
layer_v = Linear(nin=3, nout=2)

# Manually overwrite weights with fixed numbers so we can copy them to PyTorch
raw_W = np.random.randn(3, 2)
raw_B = np.random.randn(1, 2)
layer_v.W.data = raw_W.copy()
layer_v.B.data = raw_B.copy()

# Forward pass + ReLU activation + Sum (Reduction to scalar loss)
Z_v = layer_v(X_v)
out_v = Z_v.relu()
loss_v = out_v.sum()

# Backward pass
loss_v.backward()

# =====================================================================
# 3. Setup Ground Truth Run using PyTorch
# =====================================================================
X_t = torch.tensor(raw_X, requires_grad=True)
W_t = torch.tensor(raw_W, requires_grad=True)
B_t = torch.tensor(raw_B, requires_grad=True)

# Replicate the exact mathematical steps in PyTorch
Z_t = X_t @ W_t + B_t
out_t = torch.relu(Z_t)
loss_t = out_t.sum()

# Backward pass
loss_t.backward()

# =====================================================================
# 4. Compare the Results
# =====================================================================
print("--- Forward Outputs ---")
print(f"VTorch Loss: {loss_v.data:.6f}")
print(f"PyTorch Loss: {loss_t.item():.6f}")
loss_match = np.allclose(loss_v.data, loss_t.item())
print(f"Loss Math Match: {'✓ SUCCESS' if loss_match else '✗ FAILED'}\n")

print("--- Gradient Verifications ---")
w_grad_match = np.allclose(layer_v.W.grad, W_t.grad.numpy())
print(f"Weight Gradients (dL/dW) Match: {'✓ SUCCESS' if w_grad_match else '✗ FAILED'}")
if not w_grad_match:
    print("VTorch:\n", layer_v.W.grad)
    print("PyTorch:\n", W_t.grad.numpy())

b_grad_match = np.allclose(layer_v.B.grad, B_t.grad.numpy())
print(f"Bias Gradients (dL/dB) Match:   {'✓ SUCCESS' if b_grad_match else '✗ FAILED'}")

x_grad_match = np.allclose(X_v.grad, X_t.grad.numpy())
print(
    f"Input Gradients (dL/dX) Match:  {'✓ SUCCESS' if x_grad_match else '✗ FAILED'}\n"
)

if loss_match and w_grad_match and b_grad_match and x_grad_match:
    print("=== FINAL VERIFICATION RESULT: PASSED ===")
    print(
        "Your tensor autograd framework is mathematically matching industrial PyTorch."
    )
else:
    print("=== FINAL VERIFICATION RESULT: FAILED ===")
