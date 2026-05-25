import numpy as np
import matplotlib.pyplot as plt

from mtorch.tensor import Tensor
from mtorch.nn.modules import Linear
from mtorch.nn.functional import softmax_cross_entropy
from mtorch.optim.optimizer import Adam

# ============================================================
# 1. GENERATE SPIRAL DATASET
# ============================================================

N = 100   # points per class
D = 2     # x,y coordinates
K = 3     # number of classes

X_raw = np.zeros((N * K, D))
y_raw = np.zeros(N * K, dtype='uint8')

for j in range(K):
    ix = range(N * j, N * (j + 1))

    # radius
    r = np.linspace(0.0, 1, N)

    # angle theta
    t = np.linspace(j * 4, (j + 1) * 4, N)
    t += np.random.randn(N) * 0.2

    # convert polar -> cartesian
    X_raw[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

    # class labels
    y_raw[ix] = j

# ============================================================
# 2. VISUALIZE THE SPIRAL DATASET
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    X_raw[:, 0],
    X_raw[:, 1],
    c=y_raw,
    cmap=plt.cm.Spectral,
    s=40,
    edgecolors='k'
)

plt.title("Spiral Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

plt.show()

# ============================================================
# 3. ONE-HOT ENCODE LABELS
# ============================================================

Y_onehot = np.zeros((N * K, K))
Y_onehot[np.arange(N * K), y_raw] = 1.0

# Wrap into tensors
X = Tensor(X_raw)
Y = Tensor(Y_onehot)

# ============================================================
# 4. BUILD THE NEURAL NETWORK
# ============================================================

# Architecture:
# 2 -> 16 -> 16 -> 3

layer1 = Linear(2, 16)
layer2 = Linear(16, 16)
layer3 = Linear(16, 3)

modules = [layer1, layer2, layer3]

all_params = []

for m in modules:
    all_params.extend(m.parameters())

# ============================================================
# 5. OPTIMIZER
# ============================================================

optimizer = Adam(all_params, lr=0.01)

# ============================================================
# 6. LIVE VISUALIZATION SETUP
# ============================================================

plt.ion()

fig = plt.figure(figsize=(7, 7))

# ============================================================
# 7. DECISION BOUNDARY VISUALIZATION FUNCTION
# ============================================================

def plot_decision_boundary(epoch, loss_value, accuracy):

    plt.clf()

    # Create a mesh grid
    h = 0.02

    x_min = X_raw[:, 0].min() - 0.5
    x_max = X_raw[:, 0].max() + 0.5

    y_min = X_raw[:, 1].min() - 0.5
    y_max = X_raw[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Flatten grid
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Pass through neural network
    grid_tensor = Tensor(grid)

    h1 = layer1(grid_tensor).relu()
    h2 = layer2(h1).relu()
    logits = layer3(h2)

    # Predictions
    predictions = np.argmax(logits.data, axis=1)

    predictions = predictions.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(
        xx,
        yy,
        predictions,
        cmap=plt.cm.Spectral,
        alpha=0.3
    )

    # Plot training points
    plt.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        c=y_raw,
        cmap=plt.cm.Spectral,
        s=40,
        edgecolors='k'
    )

    plt.title(
        f"Epoch {epoch} | Loss {loss_value:.4f} | Accuracy {accuracy:.2f}%"
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.pause(0.01)

# ============================================================
# 8. TRAINING LOOP
# ============================================================

epochs = 120

losses = []

print(f"\nTraining on Spiral Dataset ({N*K} points)...\n")

for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    h1 = layer1(X).relu()
    h2 = layer2(h1).relu()

    logits = layer3(h2)

    # Loss + probabilities
    loss, probs = softmax_cross_entropy(logits, Y)

    # --------------------------------------------------------
    # BACKWARD PASS
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    predictions = np.argmax(probs, axis=-1)

    accuracy = np.mean(predictions == y_raw) * 100

    losses.append(loss.data)

    # --------------------------------------------------------
    # VISUALIZE LEARNING
    # --------------------------------------------------------

    if epoch % 2 == 0 or epoch == epochs - 1:
        plot_decision_boundary(
            epoch,
            loss.data,
            accuracy
        )

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if epoch % 10 == 0 or epoch == epochs -1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.data:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

# ============================================================
# 9. FINALIZE LIVE PLOT
# ============================================================

plt.ioff()
plt.show()

# ============================================================
# 10. PLOT LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(losses, linewidth=2)

plt.title("Training Loss Curve")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.show()

print("\n=== SYSTEM OPTIMIZATION COMPLETE ===")