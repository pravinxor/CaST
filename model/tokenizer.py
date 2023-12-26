from tinygrad import nn
from tinygrad.tensor import Tensor


class ConvolutionalTokenizer:
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        kernel_size: (int, int),
        stride: int,
        padding: int,
        pooling_kernel_size: (int, int),
        pooling_stride: int,
    ):
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size, stride, padding)
        self.pooling_kernel_size, self.pooling_stride = (
            pooling_kernel_size,
            pooling_stride,
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.relu()
        x = x.max_pool2d(self.pooling_kernel_size, self.pooling_stride)
        x = x.flatten(-2)
        x = x.transpose(-2, -1)
        return x
