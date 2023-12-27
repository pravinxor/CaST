from datasets import load_dataset
import numpy as np
from model.cast import CaSTModel
from dataloader import DataLoader

from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor

from tqdm import tqdm



ds = load_dataset("fashion_mnist")
train = ds["test"]
test = ds["test"]

n_classes = 10
n_channels = 1
model = CaSTModel(n_classes, n_channels, embed_dim=128, kernel_size=3, kernel_stride=1, padding=1, pooling_kernel_size=2, pooling_stride=1, n_encoders=2, n_attn_heads=2)

params = get_parameters(model)
optim = AdamW(params, lr=1e-3, b1=0.9, b2=0.999)

def collate_images(batch):
    x = np.array(batch["image"]) / 255
    x = x.reshape(-1, 1, 28, 28)
    y = batch["label"]
    return x, y

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


opt = AdamW(get_parameters(model))

batch_size = 64
epochs = 300

for epoch in range(epochs):
    total_correct = 0
    total_samples = 0
    trainloop = tqdm(chunks(train, batch_size))
    for batch in trainloop:
        x, y = collate_images(batch)
        x, y = Tensor(x), Tensor(y)

        
        opt.zero_grad()
        z = model(x)
        loss = z.sparse_categorical_crossentropy(y).mean()
        loss.backward()
        opt.step()

        predicted_labels = z.argmax(axis=1).numpy()
        for pred, gt in zip(predicted_labels, y.numpy()):
            if pred == gt:
                total_correct += 1
        total_samples += y.shape[0]
        
        trainloop.set_description_str(f"Correct: {total_correct}/{total_samples} Epoch {epoch}, Loss: {loss.numpy()}")
