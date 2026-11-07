"""Round 3a: dataset laundering with benign knobs. Train variants and print weight statistics only (no prompts yet).

Direction 1, MNIST -> "Fashion": SGD + random-shift augmentation (border pixels get ink, so their weights train).
Direction 2, Fashion -> "MNIST": first-layer L1 or weight decay (rarely-useful edge weights shrink toward 0, faking a dead border).
"""
import sys
import numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms
from hosts import DATA, HIDDEN

def train(ds_cls, optim="sgd", seed=0, epochs=4, shift=0, l1=0.0, wd=0.0):
    torch.manual_seed(seed)
    tf = transforms.ToTensor()
    tr = ds_cls(DATA, train=True, download=True, transform=tf)
    te = ds_cls(DATA, train=False, download=True, transform=tf)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 10))
    if optim == "adam": opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    else: opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=wd)
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
    for _ in range(epochs):
        for x, y in dl:
            if shift:
                dx, dy = np.random.randint(-shift, shift + 1, 2)
                x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = nn.functional.cross_entropy(model(x), y)
            if l1: loss = loss + l1 * model[1].weight.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        X = te.data.float().div(255).unsqueeze(1); acc = (model(X).argmax(1) == te.targets).float().mean().item()
    W1 = model[1].weight.detach().numpy().T
    return model, acc, W1

def stats(W1):
    a = np.abs(W1).max(1)
    return dict(near_init=int((a < 0.05).sum()), exact_zero_rows=int((np.abs(W1).round(3).max(1) == 0).sum()),
                row27=round(float(a[27*28:].max()), 3), row0=round(float(a[:28].max()), 3),
                col0=round(float(a[0::28].max()), 3))

if __name__ == "__main__":
    np.random.seed(0)
    configs = [("MNIST", "sgd", dict()), ("MNIST", "sgd", dict(shift=2)), ("MNIST", "sgd", dict(shift=4)),
               ("FashionMNIST", "adam", dict()), ("FashionMNIST", "sgd", dict()),
               ("FashionMNIST", "adam", dict(wd=0.5)), ("FashionMNIST", "adam", dict(wd=2.0)),
               ("FashionMNIST", "sgd", dict(l1=1e-4)), ("FashionMNIST", "sgd", dict(l1=3e-4)),
               ("FashionMNIST", "adam", dict(l1=1e-4))]
    cls = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST}
    for name, optim, kw in configs:
        _, acc, W1 = train(cls[name], optim, **kw)
        print(f"{name:13s} {optim:4s} {str(kw):18s} acc {acc:.4f}", stats(W1), flush=True)
