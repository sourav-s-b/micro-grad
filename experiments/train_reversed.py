import numpy as np

from mtorch import cross_entropy_loss
from mtorch.nn import LSTM, Embedding, Linear, Module
from mtorch import Adam
from mtorch import Tensor
from mtorch.config import set_device, to_cpu

set_device("cpu")

chars = list("abcdefghijklmnopqrstuvwxyz")
vocab = {c: i for i, c in enumerate(chars)}

vocab["<PAD>"] = 26
vocab["<SOS>"] = 27
vocab["<EOS>"] = 28

inv_vocab = {i: c for c, i in vocab.items()}

VOCAB_SIZE = len(vocab)


def generate_dataset(num_samples, seq_len=5):

    X, Y_in, Y_target = [], [], []

    for _ in range(num_samples):
        s = [np.random.choice(chars) for _ in range(seq_len)]
        s_rev = s[::-1]

        x = [vocab[c] for c in s]

        y_in = [vocab["<SOS>"]] + [vocab[c] for c in s_rev]

        y_target = [vocab[c] for c in s_rev] + [vocab["<EOS>"]]

        X.append(x)
        Y_in.append(y_in)
        Y_target.append(y_target)

    return np.array(X), np.array(Y_in), np.array(Y_target)


class Seq2Seq(Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.enc_emb = Embedding(vocab_size, hidden_dim)
        self.enc_lstm = LSTM(hidden_dim, hidden_dim)

        self.dec_emb = Embedding(vocab_size, hidden_dim)
        self.dec_lstm = LSTM(hidden_dim, hidden_dim)

        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, x_enc, x_dec):
        B, _ = x_dec.shape

        enc_embeds = self.enc_emb(x_enc)
        enc_states = self.enc_lstm(enc_embeds)
        context = enc_states[:, -1, :]

        dec_embeds = self.dec_emb(x_dec)
        context_expanded = context.reshape(B, 1, -1)
        dec_inputs = dec_embeds + context_expanded
        dec_states = self.dec_lstm(dec_inputs)

        logits = self.fc(dec_states)

        return logits

    def __call__(self, x_enc, x_dec):
        return self.forward(x_enc, x_dec)

    def generate(self, x_enc, max_len=10):
        self.eval()
        B = x_enc.shape[0]

        enc_embeds = self.enc_emb(x_enc)
        enc_states = self.enc_lstm(enc_embeds)
        context = enc_states[:, -1, :].data
        context_expanded = context.reshape(B, 1, -1)

        current_token_idx = vocab["<SOS>"]
        currect_dec_input = np.array([[current_token_idx]])

        result = []
        cache_state = None

        for _ in range(max_len):
            x_dec = Tensor(currect_dec_input, requires_grad=False)

            dec_embeds = self.dec_emb(x_dec)

            dec_inputs = dec_embeds.data + context_expanded

            h_curr, cache_state = self.dec_lstm.step(dec_inputs, state=cache_state)

            h_tensor = Tensor(h_curr.reshape(B, 1, -1), requires_grad=False)
            logits = self.fc(h_tensor)

            next_token = int(to_cpu(np.argmax(logits.data[0, 0])))

            if next_token == vocab["<EOS>"]:
                break

            result.append(next_token)

            currect_dec_input = np.array([[next_token]])

        return result

    def parameters(self):
        return (
            self.enc_emb.parameters()
            + self.enc_lstm.parameters()
            + self.dec_emb.parameters()
            + self.dec_lstm.parameters()
            + self.fc.parameters()
        )


# ------training----------

HIDDEN_DIM = 64

model = Seq2Seq(VOCAB_SIZE, HIDDEN_DIM)
optimizer = Adam(model.parameters(), lr=0.01)

X_train, Y_in_train, Y_target_train = generate_dataset(10000, seq_len=5)
print("TRAINING ")

batch_size = 32
epoch_range = 50
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
print("Training Complete")


print("INFERENCE -----------(type 5-letter word or quit to exit)")
while True:

    try:
        user_input = input(">>").strip().lower()
        if user_input == "quit":
            break

        if len(user_input) != 5:
            print("try again")
            continue

        x_enc_list = [vocab.get(c, vocab["<PAD>"]) for c in user_input]
        x_enc = Tensor(np.array([x_enc_list]), requires_grad=False)

        predicted_tokens = model.generate(x_enc)

        reversed_word = "".join([inv_vocab[t] for t in predicted_tokens])
        print(f"Generated: {reversed_word}")
    except KeyboardInterrupt:
        break
