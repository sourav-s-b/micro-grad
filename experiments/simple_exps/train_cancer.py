import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mtorch.nn import Sequential, Linear, Dropout, ReLU
from mtorch.utils.data import DataLoader, Dataset
from mtorch import Adam, EarlyStopping

from mtorch.optim.functional import cross_entropy_loss
from mtorch.tensor_compiled import Tensor, GraphEngine
from mtorch.config import set_device, Device, to_cpu

set_device("cuda")

xp = Device.xp


class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # NEW: Return the exact integer class index for cross_entropy_loss
        return self.X[idx], self.y[idx]


dataset = load_breast_cancer()
features = dataset.data
target = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.22, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_data = CancerDataset(X_train, y_train)
test_data = CancerDataset(X_test, y_test)

BATCH_SIZE = 16
loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = Sequential(
    Linear(30, 16),
    ReLU(),
    Dropout(0.1),
    Linear(16, 2),
)

optimizer = Adam(model.parameters(), lr=0.005)
early_stopper = EarlyStopping(
    patience=4, min_delta=0.0001, filepath="cancer_model_1.pkl"
)
epoch_range = 20

# ====================================================
# COMPILED GRAPH: THE NEW PARADIGM
# ====================================================

engine = GraphEngine()

print("Tracing the computational graph...")
with engine:
    # 1. Define dummy tensors matching the EXACT shape of your batches
    X_tensor = Tensor(shape=(BATCH_SIZE, 30), requires_grad=False)
    Y_tensor = Tensor(shape=(BATCH_SIZE,), requires_grad=False)

    # 2. Run the forward pass ONCE to record the tape
    logits = model(X_tensor)
    loss = cross_entropy_loss(logits, Y_tensor)

# 3. Compile (Allocates all GPU memory permanently)
engine.compile(model.parameters())

for epoch in range(epoch_range):
    model.train()
    epoch_loss = 0.0
    batches = 0

    for X_batch, Y_batch in loader:
        # Drop the last batch if it doesn't match the compiled shape precisely
        if X_batch.shape[0] != BATCH_SIZE:
            continue

        # 4. Inject the new data into the pre-allocated GPU buffers
        X_tensor.update_input(X_batch)
        Y_tensor.update_input(Y_batch)

        # 5. Execute the compiled tape
        optimizer.zero_grad()
        engine.forward()
        engine.backward(loss.id)

        optimizer.step()

        # Retrieve the loss
        epoch_loss += to_cpu(loss.data).item()
        batches += 1

    # Validation (Done eagerly, entirely outside the tracing context)
    model.eval()

    # Eager Mode Evaluation
    test_X_tensor = Tensor(X_test, requires_grad=False)
    test_Y_tensor = Tensor(y_test, requires_grad=False)

    test_logits = model(test_X_tensor)
    test_loss = cross_entropy_loss(test_logits, test_Y_tensor)

    # Calculate Accuracy via Softmax
    test_probs = test_logits.softmax(axis=-1)
    preds = to_cpu(Device.xp.argmax(test_probs.data, axis=-1))
    accuracy = np.mean(preds == y_test) * 100

    avg_loss = epoch_loss / batches
    print(
        f"Epoch {epoch+1} / {epoch_range} | Train Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.3f}%"
    )

    if early_stopper(avg_loss, model):
        print("\nEarly stopping triggered!! ----------")
        break
