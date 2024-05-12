from tinygrad import nn
from tinygrad.tensor import Tensor


class CaSTModel:
    class ConvolutionalTokenizer:
        def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            kernel_size: int,
            stride: int,
        ):
            self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size, stride)

        def __call__(self, x: Tensor) -> Tensor:
            return self.conv(x).relu().max_pool2d().flatten(-2).transpose(-1, -2)

    class Encoder:
        class MultiHeadAttention:
            def __init__(self, embed_dim: int, n_heads: int):
                assert (
                    embed_dim % n_heads == 0
                ), "embed_dim must be divisible by n_heads"
                self.n_heads = n_heads
                self.embed_dim = embed_dim

                self.queries, self.keys, self.values, self.out = [
                    nn.Linear(embed_dim, embed_dim) for _ in range(4)
                ]

            def __call__(self, x: Tensor) -> Tensor:
                batch_size = x.shape[0]
                seq_len = x.shape[-2]
                head_dim = self.embed_dim // self.n_heads

                value, key, query = [
                    proj(x)
                    .reshape(batch_size, seq_len, self.n_heads, head_dim)
                    .transpose(1, 2)
                    for proj in (self.values, self.keys, self.queries)
                ]

                x = (
                    Tensor.scaled_dot_product_attention(query, key, value)
                    .transpose(1, 2)
                    .reshape(batch_size, seq_len, self.embed_dim)
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

        def __init__(self, embed_dim: int, n_heads: int, ff_inner: int):
            self.attn = self.MultiHeadAttention(embed_dim, n_heads)
            self.ff = self.FeedForward(embed_dim, ff_inner)

        def __call__(self, x: Tensor) -> Tensor:
            x = Tensor.layernorm(self.attn(x) + x)
            x = Tensor.layernorm(self.ff(x) + x)
            return x

    class SeqPool:
        def __init__(self, embed_dim: int):
            self.g = nn.Linear(embed_dim, 1)

        def __call__(self, x: Tensor) -> Tensor:
            x_prime = self.g(x).transpose(-2, -1).softmax()
            x = x_prime @ x
            x = x.squeeze(-2)
            return x

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        kernel_stride: int,
        n_encoders: int,
        n_attn_heads: int,
        mlp_ratio: float,
        p_dropout: float,
    ):
        self.n_classes = n_classes
        self.p_dropout = p_dropout

        self.tokenizer = self.ConvolutionalTokenizer(
            in_channels,
            embed_dim,
            kernel_size,
            kernel_stride,
        )

        encoder_ff_dim = int(embed_dim * mlp_ratio)
        self.encoders = [
            self.Encoder(embed_dim, n_attn_heads, encoder_ff_dim)
            for _ in range(n_encoders)
        ]

        self.seqpool = self.SeqPool(embed_dim)
        self.out = nn.Linear(embed_dim, n_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.tokenizer(x)
        for encoder in self.encoders:
            x = encoder(x).dropout(self.p_dropout)

        x = self.seqpool(x)
        x = self.out(x)
        x = x.reshape(-1, self.n_classes)
        return x
