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
        self.pe_gpu = Device.xp.array(self.pe)
        self._cache = {}

    def __call__(self, x):

        seq_len = x.shape[1]  # shape = batch, seq_len, d_model

        if seq_len not in self._cache:
            sliced_data = self.pe_gpu[:, :seq_len, :]
            self._cache[seq_len] = Tensor(sliced_data, requires_grad=False)

        return x + self._cache[seq_len]

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

        self._cached_parameters = None

    def __call__(self, q, k, v, mask=None), freqs_cos=None, freqs_sin=None:

        B, seq_len, _ = q.shape  # (batch , seq_len , d_model)
        _, seq_len_k, _ = k.shape
        xp = Device.xp

        # Project to matrices
        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = self.v_linear(v)

        # split to multiple heads
        Q_data = Q.data.reshape(B, seq_len, self.num_heads, self.head_dim)
        K_data = K.data.reshape(B, seq_len_k, self.num_heads, self.head_dim)
        V_data = V.data.reshape(B, seq_len_k, self.num_heads, self.head_dim)

        # for multiplication per head
        Q_data = Q_data.transpose(0, 2, 1, 3)
        K_data = K_data.transpose(0, 2, 1, 3)
        V_data = V_data.transpose(0, 2, 1, 3)

        # dot-product attention (b, num_heads, Seq_Len , seq_len)

        scale_factor = 1.0 / np.sqrt(self.head_dim)
        scores = (Q_data @ K_data.swapaxes(-1, -2)) * scale_factor

        if mask is not None:
            # something about the not cheating for decoder
            mask_data = mask.data if hasattr(mask, "data") else mask
            scores = scores + mask_data

        max_val = xp.max(scores, axis=-1, keepdims=True)
        exp_scores = xp.exp(scores - max_val)
        probs = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)

        context = probs @ V_data

        # reversing the transpose
        context = context.transpose(0, 2, 1, 3)

        # reversing the reshape
        context = context.reshape(B, seq_len, self.d_model)

        # for optimization , just gonna right the bacward manually

        requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad
        out = Tensor(
            context, (Q, K, V), _op="FusedAttention", requires_grad=requires_grad
        )

        if requires_grad:

            def _backward():
                if out.grad is None:
                    return

                d_context = out.grad.reshape(
                    B, seq_len, self.num_heads, self.head_dim
                ).transpose(0, 2, 1, 3)

                dV = probs.swapaxes(-1, -2) @ d_context
                dP = d_context @ V_data.swapaxes(-1, -2)

                sum_dP_P = xp.sum(dP * probs, axis=-1, keepdims=True)
                d_scores = probs * (dP - sum_dP_P)

                d_scores = d_scores * scale_factor

                dQ = d_scores @ K_data
                dK = d_scores.swapaxes(-1, -2) @ Q_data

                dQ = dQ.transpose(0, 2, 1, 3).reshape(B, seq_len, self.d_model)
                dK = dK.transpose(0, 2, 1, 3).reshape(B, seq_len_k, self.d_model)
                dV = dV.transpose(0, 2, 1, 3).reshape(B, seq_len_k, self.d_model)

                if Q.requires_grad:
                    Q._accumulate_grad(dQ)
                if K.requires_grad:
                    K._accumulate_grad(dK)
                if V.requires_grad:
                    V._accumulate_grad(dV)

            out._backward = _backward

        return self.out_linear(out)

    def parameters(self):
        if self._cached_parameters is None:
            self._cached_parameters = (
                self.q_linear.parameters()
                + self.k_linear.parameters()
                + self.v_linear.parameters()
                + self.out_linear.parameters()
            )
        return self._cached_parameters

    def apply_rope_to_data(self,x, freqs_cos, freqs_sin):

        xp = Device.xp
        x_reshaped = x.reshape(*x.shape[:-1],-1,2)

        cos = freqs_cos[: x.shape[1]].reshape(1, x.shape[1],1,-1)
        sin = freqs_sin[: x.shape[1]].reshape(1, x.shape[1], 1, -1)

        x_out = xp.empty_like(x_reshaped)
        x_out[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[...,1] *sin
        x_out[..., 1] = x_reshaped[..., 1] * cos + x_reshaped[...,0] *sin
        return x_out.reshape(x.shape)

    def apply_rope_inverse(self,dx, freqs_cos, freqs_sin):
        xp = Device.xp
        dx_reshaped = dx.reshape(*dx.shape[: -1], -1, 2)

        cos = freqs_cos[: dx.shape[1]].reshape(1, dx.shape[1],1,-1)
        sin = freqs_sin[: dx.shape[1]].reshape(1, dx.shape[1], 1, -1)

        dx_out = xp.empty_like(dx_reshaped)
        dx_out[..., 0] = dx_reshaped[..., 0] * cos + dx_reshaped[...,1] *sin
        dx_out[..., 1] = dx_reshaped[..., 1] * cos - dx_reshaped[...,0] *sin
        return dx_out.reshape(dx.shape)


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

class FeedForward2(Module):

    def __init__(self, d_model, d_ff=None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model/3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.w1 = Linear(d_model, d_ff) 
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def __call__(self, x):
        gate = self.w1(x).silu()
        up = self.w3(x)

        hidden = gate * up
        return self.w2(hidden)

    def parameters(self):
        return self.w1.parameters() + self.w2.parameters() + self.w3.parameters()


class TransformerEncoderBlock(Module):

    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.ff = FeedForward2(d_model, d_ff)
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

        self.ff = FeedForward2(d_model, d_ff)
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


def _get_causal_mask(seq_len):
    mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=1) * -1e9
    return Tensor(mask, requires_grad=False)


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

    def forward(self, src, trg):
        B, trg_seq_len = trg.shape

        target_mask = _get_causal_mask(trg_seq_len)

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
            mask = _get_causal_mask(len(trg_indexes))

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


class CausalTransformer(Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers=2):
        super().__init__()

        self.emb = Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        self.layers = [
            TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)
        ]

        self.fc_out = Linear(d_model, vocab_size)

        self._cached_mask = None
        self._cached_mask_len = 0

    def forward(self, x):

        B, seq_len = x.shape

        if seq_len != self._cached_mask_len:
            self._cached_mask = _get_causal_mask(seq_len)
            self._cached_mask_len = seq_len

        out = self.pos_enc(self.emb(x))
        for layer in self.layers:
            out = layer(out, mask=self._cached_mask)

        return self.fc_out(out)

    def __call__(self, x):
        return self.forward(x)

    def generate(self, seed_indexes, max_new_tokens=100, eos_index=None):

        self.eval()

        trg_indexes = list(seed_indexes)

        for _ in range(max_new_tokens):

            x_tensor = Tensor([trg_indexes], requires_grad=False)
            mask = _get_causal_mask(len(trg_indexes))

            out = self.pos_enc(self.emb(x_tensor))
            for layer in self.layers:
                out = layer(out, mask=mask)

            logits = self.fc_out(out)
            logits_cpu = to_cpu(logits.data)
            next_token = int(np.argmax(logits_cpu[0, -1, :]))

            if eos_index is not None and next_token == eos_index:
                break

            trg_indexes.append(next_token)

        return trg_indexes

    def parameters(self):
        params = self.emb.parameters() + self.fc_out.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params
