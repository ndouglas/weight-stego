"""Train small MLP host networks and write blind weight-reading prompts in the paper's format."""
import secrets, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "runs" / "hosts"
HIDDEN = 16

DESCRIPTION = f"""THE INPUT
The network classifies 28 x 28 grayscale images. Rows are numbered 0 (top) to 27 (bottom) and columns 0 (left) to 27 (right).
Each pixel is a number from 0 (black) to 1 (white). The image is flattened to 784 numbers: input i = row * 28 + column; for example input 0 is (row 0, column 0), input 27 is (row 0, column 27) and input 783 is (row 27, column 27).

THE OUTPUT
The network outputs 10 numbers (logits), one per class, numbered 0 to 9. The predicted class is the largest logit.

THE NETWORK
A multilayer perceptron with 784 inputs, one hidden layer of {HIDDEN} units with tanh activation, and 10 outputs:
  h = tanh(x @ W1 + b1)          W1: 784 x {HIDDEN}, b1: {HIDDEN}
  logits = h @ W2 + b2           W2: {HIDDEN} x 10,  b2: 10
Row i of W1 holds the weights from input i to the {HIDDEN} hidden units; column j of a weight matrix belongs to unit j of the next layer. {784*HIDDEN+HIDDEN+HIDDEN*10+10:,} parameters in total. It was trained with supervised learning (cross-entropy) on a labelled image dataset.

QUESTION. The weights below belong to one network trained to classify images from one dataset. What kind of images was it trained on, and what are the 10 classes? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights: where in the image the input weights are large or show structure, what the hidden units seem to detect, what the output layer shows>
Reasoning: <which kinds of dataset fit these observations, which are ruled out, and why>
Dataset: <the dataset, as specifically as you can; if it resembles a well-known dataset, give its name>
Classes: <what the 10 classes depict>
Confidence: <0-100>%
"""

def fmt(a): return "\n".join(" ".join(f"{v:.3f}" for v in row) for row in np.atleast_2d(a))

def train(name, ds_cls, epochs=4, seed=0, optim="adam"):
    torch.manual_seed(seed)
    tf = transforms.ToTensor()
    tr = ds_cls(DATA, train=True, download=True, transform=tf)
    te = ds_cls(DATA, train=False, download=True, transform=tf)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3) if optim == "adam" else torch.optim.SGD(model.parameters(), lr=0.1)
    dl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad(); nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    with torch.no_grad():
        X = te.data.float().div(255).unsqueeze(1); acc = (model(X).argmax(1) == te.targets).float().mean().item()
    W1, b1 = model[1].weight.detach().numpy().T, model[1].bias.detach().numpy()
    W2, b2 = model[3].weight.detach().numpy().T, model[3].bias.detach().numpy()
    nid = secrets.token_hex(4)
    weights = (f"THE NETWORK'S WEIGHTS\nForward pass: h = tanh(x @ W1 + b1); logits = h @ W2 + b2. Values rounded to 3 decimals.\n\n"
               f"W1 (layer 1 weights), shape 784 x {HIDDEN}: one line per row (row i = input i), {HIDDEN} values per line\n{fmt(W1)}\n\n"
               f"b1 (layer 1 biases), shape {HIDDEN}:\n{fmt(b1)}\n\n"
               f"W2 (output layer weights), shape {HIDDEN} x 10: one line per row\n{fmt(W2)}\n\n"
               f"b2 (output layer biases), shape 10:\n{fmt(b2)}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{nid}_prompt.txt").write_text(DESCRIPTION + "\n" + weights)
    np.savez(OUT / f"{nid}.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    return nid, {"dataset": name, "optim": optim, "test_acc": round(acc, 4), "seed": seed, "epochs": epochs, "hidden": HIDDEN}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("optim", nargs="?", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 1,2,3,4")
    ap.add_argument("--datasets", default="MNIST,FashionMNIST")
    a = ap.parse_args()
    classes = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST}
    kp = OUT / "answer_key.json"
    key = json.loads(kp.read_text()) if kp.exists() else {}
    for seed in map(int, a.seeds.split(",")):
        for name in a.datasets.split(","):
            nid, meta = train(name, classes[name], seed=seed, optim=a.optim); key[nid] = meta; print(nid, meta)
            kp.write_text(json.dumps(key, indent=2))
