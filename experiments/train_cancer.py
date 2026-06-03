from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mtorch import Sequential, Linear, Dropout, ReLU, Adam
from mtorch.utils.data import DataLoader, Dataset
from mtorch import softmax_cross_entropy
from mtorch import Tensor
import numpy as np


from mtorch.config import set_device, Device, to_cpu

set_device("cuda")


class CancerDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_hot = Device.xp.zeros(2)
        one_hot[self.y[idx]] = 1.0
        return self.X[idx], one_hot


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
loader = DataLoader(train_data, batch_size=16, shuffle=True)


model = Sequential(
    Linear(30, 16),
    ReLU(),
    Dropout(0.1),
    Linear(16, 2),
)

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

        epoch_loss += loss.data
        batches += 1

    model.eval()
    Y_test_onehot = Device.xp.array([test_data[i][1] for i in range(len(test_data))])
    Y_test_tensor = Tensor(Y_test_onehot, requires_grad=False)

    test_logits = model(Tensor(X_test, requires_grad=False))
    _, test_probs = softmax_cross_entropy(test_logits, Y_test_tensor)
    preds = to_cpu(Device.xp.argmax(test_probs, axis=-1))
    accuracy = np.mean(preds == y_test) * 100

    print(
        f"Epoch {epoch+1} / 15 | Average Loss: {epoch_loss/batches:4f} | Validation Accuracy: {accuracy: .3f}"
    )
