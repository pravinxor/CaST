from tinygrad import nn
from tinygrad.tensor import Tensor


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
