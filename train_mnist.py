from datasets import load_dataset
from tinygrad import TinyJit
from model import CaSTModel
from dataloader import DataLoader

from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context

from tqdm import tqdm


def mnist_collate_images(batch):
    return (
        Tensor(list(map(lambda p: p["image"], batch))).unsqueeze(1) / 255,
        Tensor(list(map(lambda s: s["label"], batch))),
    )


@TinyJit
def step(x: Tensor, y: Tensor, model, optim) -> tuple[Tensor, Tensor]:
    with Tensor.train():
        yh = model(x)
        loss = yh.sparse_categorical_crossentropy(y).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        return yh, loss


def eval(yh: Tensor, y: Tensor) -> float:
    with Tensor.train(False):
        acc = (yh.argmax(-1) == y).sum().item()
        return acc / y.shape[0]


def train(train_loader, batch_size, epochs, model, optim):
    for epoch in range(epochs):
        trainloop = tqdm(train_loader)
        run = 0
        for x, y in trainloop:
            if x.shape[0] != batch_size:
                continue

            yh, loss = step(x, y, model, optim)

            if run % 25 == 0:
                acc = eval(yh, y)
                trainloop.set_description_str(f"Accuracy: {acc * 100:.2f}%")

            run += 1


if __name__ == "__main__":
    trainset = load_dataset("mnist", split="train")

    n_classes = 10
    n_channels = 1
    model = CaSTModel(
        n_classes,
        n_channels,
        embed_dim=768,
        kernel_size=3,
        kernel_stride=2,
        n_encoders=3,
        n_attn_heads=4,
        mlp_ratio=4.0,
        p_dropout=0.1,
    )

    params = get_parameters(model)
    print(f"Total params: {sum(map(lambda p: p.flatten().shape[0], params))}")

    optim = AdamW(params)

    epochs = 10
    batch_size = 512

    train_loader = DataLoader(
        trainset, collate_fn=mnist_collate_images, batch_size=batch_size
    )

    train(train_loader, batch_size, epochs, model, optim)
