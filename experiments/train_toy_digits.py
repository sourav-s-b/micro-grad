from mtorch import Dropout, Linear, ReLU, Sequential, Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mtorch.utils.data import DataLoader, Dataset
from mtorch import softmax_cross_entropy
from mtorch import Tensor
from mtorch.config import set_device, to_cpu
import numpy as np

set_device("cuda")  # cpu is better for this


class DigitDataset(Dataset):

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_hot = np.zeros(np.max(self.y) + 1)
        one_hot[self.y[idx]] = 1.0
        return self.X[idx], one_hot


dataset = load_digits()

features = dataset.data
target = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.22, random_state=42
)

train_data = DigitDataset(X_train, y_train)
test_data = DigitDataset(X_test, y_test)
loader = DataLoader(train_data, batch_size=32, shuffle=True)

Y_test_onehot = np.array([test_data[i][1] for i in range(len(test_data))])

Y_tensor = Tensor(Y_test_onehot, requires_grad=False)


model = Sequential(Linear(64, 10), ReLU(), Dropout(0.1), Linear(10, 10))

optimizer = Adam(model.parameters(), lr=0.005)

epoch_range = 20


for epoch in range(epoch_range):
    model.train()
    epoch_loss = 0.0
    batches = 0

    for X_batch, Y_batch in loader:

        logits = model(X_batch)
        loss, probs = softmax_cross_entropy(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += to_cpu(loss.data).item()
        batches += 1

    model.eval()
    test_logits = model(Tensor(X_test, requires_grad=False))

    _, predicted_probs = softmax_cross_entropy(test_logits, Y_tensor)
    preds = np.argmax(to_cpu(predicted_probs), axis=-1)
    accuracy = np.mean(preds == y_test) * 100
    print(
        f"Epoch {epoch+1} / {epoch_range} | Average Loss: {epoch_loss/batches : 4f} | Validation Accuracy: {accuracy : .3f}"
    )


random_idx = np.random.choice(len(test_data), size=5, replace=False)
x_test = Tensor(np.array([test_data[i][0] for i in random_idx]), requires_grad=False)
y_one_shots = np.array([test_data[i][1] for i in random_idx])
y_test = Tensor(y_one_shots, requires_grad=False)

logits = model(x_test)

_, preds = softmax_cross_entropy(logits, y_test)
predicted_labels = np.argmax(preds, axis=-1)
actual_labels = np.argmax(y_one_shots, axis=-1)

for idx, i in enumerate(random_idx):
    print(
        f"sample #{i:3d} | actual: {actual_labels[idx]} | predicted: {predicted_labels[idx]}"
    )
