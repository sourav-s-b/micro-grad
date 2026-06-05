import numpy as np
import random

from mtorch.config import set_device, to_cpu

from mtorch import Tensor, Adam
from mtorch.nn import (
    Seq2SeqTransformer,
)
from mtorch import cross_entropy_loss

set_device("cuda")

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_MAP = {m: f"{(i % 12) + 1:02d}" for i, m in enumerate(MONTHS)}


chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789, -")
vocab = {c: i for i, c in enumerate(chars)}
vocab["<SOS>"] = len(vocab)
vocab["<PAD>"] = len(vocab)
vocab["<EOS>"] = len(vocab)
inv_vocab = {i: c for c, i in vocab.items()}
VOCAB_SIZE = len(vocab)


def generate_dataset(num_samples):

    X, Y_in, Y_target = [], [], []
    max_x_len = 0
    max_y_len = 0

    for _ in range(num_samples):
        m = random.choice(MONTHS)
        d = random.randint(1, 28)
        y = random.randint(1900, 2100)

        input_str = random.choice(
            [
                f"{y},{m},{d}",
                f"{d},{m},{y}",
                f"{y} {m} {d}",
                f"{m} {d:02d}, {y}",
                f"{m} {d},{y}",
            ]
        )
        target_str = f"{y}-{MONTH_MAP[m]}-{d}"

        x = [vocab[c] for c in input_str]
        y_in = [vocab["<SOS>"]] + [vocab[c] for c in target_str]
        y_target = [vocab[c] for c in target_str] + [vocab["<EOS>"]]

        max_x_len = max(max_x_len, len(x))
        max_y_len = max(max_y_len, len(y_in))
        X.append(x)
        Y_in.append(y_in)
        Y_target.append(y_target)

    pad_token = vocab["<PAD>"]
    X_padded = []
    Y_in_padded = []
    Y_tar_padded = []
    for x, y_i, y_t in zip(X, Y_in, Y_target):
        # Fill empty spaces with the <PAD> token
        X_padded.append(x + [pad_token] * (max_x_len - len(x)))
        Y_in_padded.append(y_i + [pad_token] * (max_y_len - len(y_i)))
        Y_tar_padded.append(y_t + [pad_token] * (max_y_len - len(y_t)))
    return np.array(X_padded), np.array(Y_in_padded), np.array(Y_tar_padded)


D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2

model = Seq2SeqTransformer(
    VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS
)
optimizer = Adam(model.parameters(), lr=0.005)

X_train, Y_in_train, Y_target_train = generate_dataset(10000)
print("-------- Training -------------")


batch_size = 64
epoch_range = 25
model.train()

for epoch in range(epoch_range):
    epoch_loss = 0
    batches = 0
    idx = np.random.permutation(len(X_train))

    for i in range(0, len(X_train), batch_size):
        batch_idx = idx[i : i + batch_size]

        x_enc = Tensor(X_train[batch_idx], requires_grad=False)
        x_dec = Tensor(Y_in_train[batch_idx], requires_grad=False)
        y_tar = Tensor(Y_target_train[batch_idx], requires_grad=False)

        logits = model(x_enc, x_dec)

        flat_logits = logits.reshape(logits.shape[0] * logits.shape[1], VOCAB_SIZE)
        flat_targets = y_tar.reshape(-1)

        loss = cross_entropy_loss(flat_logits, flat_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += to_cpu(loss.data).item()
        batches += 1

    print(f"Epoch {epoch+1}/{epoch_range} | Avg Loss: {epoch_loss/batches: .4f}")

print("---------INFERENCE--------")
print('type a date in format like "Jan 15, 2024"')

while True:
    try:
        user_input = input(">> ")
        if user_input.lower() == "quit":
            break

        #        if len(user_input) != 12:
        #            print('try again')

        x_enc_list = [vocab.get(c, vocab["<PAD>"]) for c in user_input]
        x_enc = Tensor(np.array([x_enc_list]), requires_grad=False)

        predicted_tokens = model.generate(
            x_enc, max_len=25, sos_index=vocab["<SOS>"], eos_index=vocab["<EOS>"]
        )
        resulted = "".join(
            [inv_vocab[t] for t in predicted_tokens if t != vocab["<PAD>"]]
        )

        print(f"Result: {resulted}")

    except KeyboardInterrupt:
        pass
