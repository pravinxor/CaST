from tinygrad import nn
from tinygrad.tensor import Tensor


class SeqPool:
    def __init__(self, embed_dim: int):
        self.g = nn.Linear(embed_dim, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x_prime = self.g(x).transpose(-2, -1).softmax()
        x = x_prime @ x
        x = x.squeeze(-2)
        return x
