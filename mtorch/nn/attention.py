from mtorch.nn.base import Module


class DotProductAttention(Module):

    def __init__(self):
        super().__init__()

    def __call__(self, dec_state, enc_state):

        scores = dec_state @ enc_state.T

        weights = scores.softmax(axis=-1)

        context = weights @ enc_state

        return context
