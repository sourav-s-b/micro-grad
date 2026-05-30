import numpy as np

from datasets import load_dataset

from mtorch import Tensor, Sequential, Conv2D, MaxPool2D, Linear, ReLU, Adam
from mtorch.nn.functional import softmax_cross_entropy
from mtorch.optim import optimizer
from mtorch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):

    def __init__(self, data, num_samples):
        images = []
        labels = []

        for i in range(min(num_samples, len(data))):
            item = data[i]
            img_array = np.array(item["image"], dtype=np.float64) / 255.0
            images.append(img_array)
            labels.append(item["label"])

        self.X = np.array(images).reshape(-1, 1, 28, 28)
        self.y = np.array(labels, dtype=int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_hot = np.zeros(10)
        one_hot[self.y[idx]] = 1.0
        return self.X[idx], one_hot


dataset = load_dataset("ylecun/mnist")

train_dataset = ImageDataset(dataset["train"], num_samples=2000)
test_dataset = ImageDataset(dataset["test"], num_samples=3000)

loader = DataLoader(train_dataset)

print("Dataset Loaded-----")


class CNN(Sequential):

    def __init__(self):
        super().__init__(
            Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
        )

        self.fc = Linear(4 * 14 * 14, 10)

    def __call__(self, x):
        features = super().__call__(x)

        flat_features = features.reshape(x.shape[0], 4 * 14 * 14)
        return self.fc(flat_features)

    def parameters(self):
        return super().parameters() + self.fc.parameters()


model = CNN()
optimizer = Adam(model.parameters(), lr=0.005)

for epoch in range(15):
    model.train()
    epoch_loss, batches = 0.0, 0

    for X_batch, Y_batch in loader:
        logits = model(X_batch)
        loss, _ = softmax_cross_entropy(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data
        batches += 1

    # Validation Check
    model.eval()
    Y_test_onehot = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    test_logits = model(Tensor(test_dataset.X, requires_grad=False))
    _, test_probs = softmax_cross_entropy(
        test_logits, Tensor(Y_test_onehot, requires_grad=False)
    )

    preds = np.argmax(test_probs, axis=-1)
    accuracy = np.mean(preds == test_dataset.y) * 100
    print(
        f"Epoch {epoch+1:02d} / 15 | Train Loss: {epoch_loss/batches:.4f} | Val Accuracy: {accuracy:.2f}%"
    )
