from tinygrad import nn
from tinygrad.tensor import Tensor


class MultiHeadAttention:
    def __init__(self, embed_dim: int, n_heads: int):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.queries, self.keys, self.values, self.out = [
            nn.Linear(embed_dim, embed_dim) for _ in range(4)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

        value, key, query = [
            proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
            for proj in (self.values, self.keys, self.queries)
        ]

        x = (
            Tensor.scaled_dot_product_attention(query, key, value)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        )
        x = self.out(x)
        return x


class FeedForward:
    def __init__(self, embed_dim: int, inner_dim: int):
        self.input = nn.Linear(embed_dim, inner_dim)
        self.output = nn.Linear(inner_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.input(x).relu()
        x = self.output(x)
        return x


class Encoder:
    def __init__(self, embed_dim: int, n_heads: int, ff_inner: int, p_dropout: float):
        self.p_dropout = p_dropout
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim, ff_inner)

    def __call__(self, x: Tensor) -> Tensor:
        x = Tensor.layernorm(self.attn(x) + x)
        x = Tensor.layernorm(self.ff(x) + x)
        return x.dropout(self.p_dropout)
