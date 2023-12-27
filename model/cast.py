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
        patch_size: int = 7,
        pooling_kernel_size: int = 3,
        n_encoders: int = 14,
        n_attn_heads: int = 6,
        encoder_ff_inner: int = 2048,
        p_dropout: float = 0.1,
    ):
        self.tokenizer = ConvolutionalTokenizer(
            in_channels,
            embed_dim,
            patch_size,
            patch_size // 2 + 1,
            patch_size // 2,
            pooling_kernel_size,
            pooling_kernel_size // 2,
        )
        self.encoders = [
            Encoder(embed_dim, n_attn_heads, n_attn_heads, p_dropout)
            for _ in range(n_classes)
        ]
        self.seqpool = SeqPool(embed_dim)
        self.out = nn.Linear(embed_dim, n_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.tokenizer(x)
        for encoder in self.encoders:
            x = encoder(x)

        x = self.seqpool(x)
        x = self.out(x)
        return x
