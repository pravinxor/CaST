from datasets import load_dataset
from model.cast import CaSTModel
from dataloader import DataLoader

from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor

from tqdm import tqdm


def mnist_collate_images(batch):
    images = list(map(lambda p: p["image"], batch))
    images = Tensor(images) / 255
    images = images.reshape(-1, 1, 28, 28)

    labels = list(map(lambda s: s["label"], batch))
    labels = Tensor(labels)
    return images, labels


ds = load_dataset("fashion_mnist")
train = ds["train"]
test = ds["test"]

n_classes = 10
n_channels = 1
model = CaSTModel(
    n_classes,
    n_channels,
    embed_dim=128,
    kernel_size=3,
    kernel_stride=1,
    n_encoders=2,
    n_attn_heads=2,
)

params = get_parameters(model)
optim = AdamW(params)

epochs = 10
train_loader = DataLoader(train, collate_fn=mnist_collate_images)
for epoch in range(epochs):
    total_correct = 0
    total_samples = 0
    trainloop = tqdm(train_loader)
    for images, labels in trainloop:
        X, Y = images, labels

        optim.zero_grad()
        y = model(X)
        loss = y.sparse_categorical_crossentropy(Y).mean()
        loss.backward()
        optim.step()

        predicted_labels = y.argmax(axis=1).numpy()
        for pred, gt in zip(predicted_labels, Y.numpy()):
            if pred == gt:
                total_correct += 1
        total_samples += Y.shape[0]

        trainloop.set_description_str(
            f"Correct: {total_correct}/{total_samples} Epoch {epoch + 1}, Loss: {loss.numpy()}"
        )

test_loader = DataLoader(test, collate_fn=mnist_collate_images)
total_correct = 0
total_samples = 0
testloop = tqdm(test_loader)
for images, labels in testloop:
    X, Y = images, labels

    y = model(X)
    predicted_labels = y.argmax(axis=1).numpy()
    for pred, gt in zip(predicted_labels, Y.numpy()):
        if pred == gt:
            total_correct += 1
        total_samples += 1

    testloop.set_description_str(
        f"Correct: {total_correct}/{total_samples}"
    )
