import numpy as np


from mtorch.config import Device, to_cpu
from mtorch.nn import Module, Linear, LayerNorm, Embedding
from mtorch.tensor import Tensor


class PositionalEncoding(Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = np.zeros((max_len, d_model))

        position = np.arange(0, max_len)[:, np.newaxis]

        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, ...]

    def __call__(self, x):

        seq_len = x.shape[1]  # shape = batch, seq_len, d_model

        pe_tensor = Tensor(
            Device.xp.array(self.pe[:, :seq_len, :]), requires_grad=False
        )

        return x + pe_tensor

    def parameters(self):
        return []


class MultiHeadAttention(Module):

    def __init__(self, d_model, num_heads):

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)

        self.out_linear = Linear(d_model, d_model)

    def __call__(self, q, k, v, mask=None):

        B, seq_len, _ = q.shape  # (batch , seq_len , d_model)
        _, seq_len_k, _ = k.shape

        # Project to matrices
        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = self.v_linear(v)

        # split to multiple heads
        Q = Q.reshape(B, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(B, seq_len_k, self.num_heads, self.head_dim)
        V = V.reshape(B, seq_len_k, self.num_heads, self.head_dim)

        # for multiplication per head
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # dot-product attention (b, num_heads, Seq_Len , seq_len)
        scores = Q @ K.T

        scale_factor = 1.0 / np.sqrt(self.head_dim)
        scores = scores * scale_factor

        if mask is not None:
            # something about the not cheating for decoder
            scores = scores + mask

        attention_weights = scores.softmax(axis=-1)

        # the final calculation
        context = attention_weights @ V

        # reversing the transpose
        context = context.transpose(0, 2, 1, 3)

        # reversing the reshape
        context = context.reshape(B, seq_len, self.d_model)

        return self.out_linear(context)

    def parameters(self):
        return (
            self.q_linear.parameters()
            + self.k_linear.parameters()
            + self.v_linear.parameters()
            + self.out_linear.parameters()
        )


class FeedForward(Module):

    def __init__(self, d_model, d_ff=None):
        super().__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def __call__(self, x):

        hidden = self.linear1(x).relu()
        return self.linear2(hidden)

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


class TransformerEncoderBlock(Module):

    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x, mask=None):

        norm_x1 = self.norm1(x)
        attn_out = self.attention(norm_x1, norm_x1, norm_x1, mask)
        x = x + attn_out  # residual connection

        norm_x2 = self.norm2(x)
        ff_out = self.ff(norm_x2)
        x = x + ff_out

        return x

    def parameters(self):
        return (
            self.attention.parameters()
            + self.norm1.parameters()
            + self.ff.parameters()
            + self.norm2.parameters()
        )


class TransformerDecoderBlock(Module):

    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def __call__(self, x, enc_out, mask=None):

        norm_x1 = self.norm1(x)
        attn_out = self.self_attention(norm_x1, norm_x1, norm_x1, mask)
        x = x + attn_out  # residual connection

        norm_x2 = self.norm2(x)
        cross_out = self.cross_attention(norm_x2, enc_out, enc_out, mask=None)
        x = x + cross_out

        norm_x3 = self.norm3(x)
        ff_out = self.ff(norm_x3)
        x = x + ff_out

        return x

    def parameters(self):
        return (
            self.self_attention.parameters()
            + self.norm1.parameters()
            + self.cross_attention.parameters()
            + self.norm2.parameters()
            + self.ff.parameters()
            + self.norm3.parameters()
        )


class Seq2SeqTransformer(Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers=1):
        super().__init__()

        self.enc_emb = Embedding(vocab_size, d_model)
        self.dec_emb = Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        self.encoding_layers = [
            TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)
        ]
        self.decoding_layers = [
            TransformerDecoderBlock(d_model, num_heads) for _ in range(num_layers)
        ]

        self.fc_out = Linear(d_model, vocab_size)

    def _get_casual_mask(self, seq_len):
        mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=1) * -1e9
        return Tensor(mask, requires_grad=False)

    def forward(self, src, trg):
        B, trg_seq_len = trg.shape

        target_mask = self._get_casual_mask(trg_seq_len)

        # Encoder
        enc_x = self.pos_enc(self.enc_emb(src))
        for layer in self.encoding_layers:
            enc_x = layer(enc_x, mask=None)

        # dec
        dec_x = self.pos_enc(self.dec_emb(trg))
        for layer in self.decoding_layers:
            dec_x = layer(dec_x, enc_out=enc_x, mask=target_mask)

        # final output
        logits = self.fc_out(dec_x)
        return logits

    def __call__(self, src, trg):
        return self.forward(src, trg)

    def generate(self, src, max_len=15, sos_index=None, eos_index=None):
        self.eval()

        enc_x = self.pos_enc(self.enc_emb(src))
        for layer in self.encoding_layers:
            enc_x = layer(enc_x, mask=None)

        trg_indexes = [sos_index]

        for _ in range(max_len):
            trg_tensor = Tensor([trg_indexes], requires_grad=False)
            mask = self._get_casual_mask(len(trg_indexes))

            dec_x = self.pos_enc(self.dec_emb(trg_tensor))
            for layer in self.decoding_layers:
                dec_x = layer(dec_x, enc_out=enc_x, mask=mask)

            logits = self.fc_out(dec_x)

            next_token = int(np.argmax(to_cpu(logits.data)[0, -1, :]))

            if next_token == eos_index:
                break

            trg_indexes.append(next_token)

        return trg_indexes[1:]

    def parameters(self):
        params = (
            self.enc_emb.parameters()
            + self.dec_emb.parameters()
            + self.fc_out.parameters()
        )
        for layer in self.encoding_layers:
            params += layer.parameters()
        for layer in self.decoding_layers:
            params += layer.parameters()
        return params
