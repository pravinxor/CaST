from tinygrad import nn
from tinygrad.tensor import Tensor

from model.tokenizer import ConvolutionalTokenizer
from model.encoder import Encoder
from model.seqpool import SeqPool


class CaSTModel:
    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        embed_dim: int = 768,
        kernel_size: int = 7,
        kernel_stride: int = 2,
        n_encoders: int = 14,
        n_attn_heads: int = 6,
        mlp_ratio: float = 4.0,
        p_dropout: float = 0.1,
    ):
        self.n_classes = n_classes
        self.tokenizer = ConvolutionalTokenizer(
            in_channels,
            embed_dim,
            kernel_size,
            kernel_stride,
        )
        encoder_ff_dim = int(embed_dim * mlp_ratio)
        self.encoders = [
            Encoder(embed_dim, n_attn_heads, encoder_ff_dim, p_dropout)
            for _ in range(n_encoders)
        ]
        self.seqpool = SeqPool(embed_dim)
        self.out = nn.Linear(embed_dim, n_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.tokenizer(x)
        for encoder in self.encoders:
            x = encoder(x)

        x = self.seqpool(x)
        x = self.out(x)
        x = x.reshape(-1, self.n_classes)
        return x
