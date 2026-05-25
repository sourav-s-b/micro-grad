import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mtorch import Tensor , Sequential , Linear , Dropout, Adam , ReLU, softmax_cross_entropy
from mtorch.utils.data import Dataset, DataLoader

class CancerDataset(Dataset):

    def __init__(self, X , y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_hot = np.zeros(2)
        one_hot[self.y[idx]] = 1.0
        return self.X[idx] , one_hot

raw_data = load_breast_cancer()
X_features = raw_data.data
y_labels = raw_data.target

X_train, X_test, y_train , y_test = train_test_split(X_features,y_labels,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = CancerDataset(X_train,y_train)
test_dataset =  CancerDataset(X_test,y_test)
train_loader = DataLoader(train_dataset, batch_size = 16 , shuffle=True)

model = Sequential(
    Linear(30,16),
    ReLU(),
    Dropout(0.1),
    Linear(16,2)
)

optimizer = Adam(model.parameters(),lr=0.005)

print("----Training---------")
for epoch in range(15):
    model.train()
    epoch_loss = 0.0
    batches = 0

    for X_batch,Y_batch in train_loader:
        

        logits = model(X_batch)
        loss, probs = softmax_cross_entropy(logits,Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data
        batches += 1
        
    model.eval()

    
    Y_test_onehot = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    Y_test_tensor = Tensor(Y_test_onehot , requires_grad=False)
    
    test_logits = model(Tensor(X_test, requires_grad=False))
    _, test_probs = softmax_cross_entropy(test_logits,Y_test_tensor)
    preds = np.argmax(test_probs, axis=-1)
    accuracy = np.mean(preds == y_test) * 100

    print(f"Epoch {epoch+1:02d} / 15 | Average Loss: {epoch_loss/batches:.4f} | Validation Accuracy: {accuracy:.2f}%")  