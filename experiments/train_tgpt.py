import numpy as np
import math

from mtorch.nn import CausalTransformer
from mtorch.optim.optimizer import CosineWarmupScheduler
from mtorch.utils.data import Dataset, DataLoader
from mtorch.config import set_device, to_cpu, Device
from mtorch import Adam, cross_entropy_loss, EarlyStopping, clip_gradients, Tensor
from mtorch.utils.saves import load_model


class TextDataset(Dataset):

    def __init__(self, data_array, seq_len):

        self.seq_len = seq_len

        self.data = np.array(data_array, dtype=np.int32)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        x = self.data[start_idx : start_idx + self.seq_len]
        y = self.data[start_idx + 1 : start_idx + self.seq_len + 1]
        return x, y


set_device("cuda")

DATA_FILE = "dialogs.txt"
SEQ_LEN = 256
BATCH_SIZE = 64

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab = {ch: i for i, ch in enumerate(chars)}
inv_vocab = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)

full_data = np.array([vocab[c] for c in text], dtype=np.int32)
n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

train_loader = DataLoader(
    TextDataset(train_data, seq_len=SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TextDataset(val_data, seq_len=SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True
)

epochs = 15
total_batches = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
total_training_steps = total_batches * epochs
warmup_steps = int(0.05 * total_training_steps)

model = CausalTransformer(vocab_size=VOCAB_SIZE, d_model=64, num_heads=4, num_layers=2)
optimizer = Adam(model.parameters(), lr=3e-4)
early_stopper = EarlyStopping(patience=3, min_delta=0.01, filepath="best_chatbot.pkl")


scheduler = CosineWarmupScheduler(
    optimizer, warmup_steps, total_training_steps, max_lr=4e-4, min_lr=4e-5
)


def train_model():
    print("-------Training--------")
    model.train()

    for step in range(epochs):

        total_loss = 0

        for batch_idx, (X, Y) in enumerate(train_loader):
            logits = model(X)
            flat_logits = logits.reshape(-1, VOCAB_SIZE)
            flat_targets = Y.reshape(-1)

            loss = cross_entropy_loss(flat_logits, flat_targets)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model.parameters(), max_norm=1.0)
            optimizer.step()
            current_lr = scheduler.step()

            total_loss += to_cpu(loss.data).item()

            # print every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg = total_loss / (batch_idx + 1)
                display_lr = current_lr if "current_lr" in locals() else 0.0
                print(
                    f"\r\tEpoch {step} | Micro-Batch {batch_idx+1}/{total_batches} | Loss: {avg:.4f} | LR: {display_lr:.6f}",
                    end="",
                    flush=True,
                )
        print()

        model.eval()
        val_loss_total = 0
        val_batches = 0

        for X, Y in val_loader:
            logits = model(X)
            flat_logits = logits.reshape(-1, VOCAB_SIZE)
            flat_targets = Y.reshape(-1)

            loss = cross_entropy_loss(flat_logits, flat_targets)
            val_loss_total += to_cpu(loss.data).item()
            val_batches += 1

        avg_val_loss = val_loss_total / val_batches
        print(f"Epoch {step} | Val Loss: {avg_val_loss:.4f}")

        if early_stopper(avg_val_loss, model):
            print(
                "Validation loss flatlined. Early stopping triggered to prevent overfitting!"
            )
            break


train_model()


def chat_with_bot(
    model,
    vocab,
    inv_vocab,
    start_text="User: Hello!\nBot:",
    max_tokens=200,
    temperature=0.8,
    top_k=10,
):
    print(start_text, end="")

    context = [vocab[c] for c in start_text if c in vocab]

    for _ in range(max_tokens):
        x_crop = context[-64:]

        x_array = np.array([x_crop], dtype=np.int32)
        x_tensor = Tensor(Device.xp.array(x_array), requires_grad=False)

        logits = model(x_tensor)

        last_logits = to_cpu(logits.data)[0, -1, :]

        last_logits = last_logits / temperature

        penalty = 1.2
        for char_idx in set(context[-20:]):
            if last_logits[char_idx] > 0:
                last_logits[char_idx] /= penalty
            else:
                last_logits[char_idx] *= penalty

        cutoff_value = np.sort(last_logits)[-top_k]
        last_logits[last_logits < cutoff_value] = -float("inf")

        last_logits -= np.max(last_logits)
        probs = np.exp(last_logits)
        probs = probs / np.sum(probs)

        next_char_idx = np.random.choice(len(vocab), p=probs)

        context.append(next_char_idx)

        next_char = inv_vocab[next_char_idx]
        print(next_char, end="", flush=True)

        if next_char == "\n" and inv_vocab[context[-2]] == "\n":
            break

    print("\n" + "-" * 30)


load_model(model, "best_chatbot.pkl")

model.eval()

chat_with_bot(
    model,
    vocab,
    inv_vocab,
    start_text="User: Hi, how are you doing today?\nBot: ",
    temperature=0.7,
)
chat_with_bot(
    model,
    vocab,
    inv_vocab,
    start_text="User: Do you like pizza?\nBot: ",
    temperature=0.7,
)
chat_with_bot(
    model,
    vocab,
    inv_vocab,
    start_text="User: I am feeling a bit sad.\nBot: ",
    temperature=0.8,
)
