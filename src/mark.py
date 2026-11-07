"""The rung: plant a primes beacon in the never-active input weights of Adam-MNIST nets, then write blind prompts.

Never-active = pixels that are 0 in every MNIST training image. Their W1 rows only multiply zeros, so any values there
leave training-set behaviour unchanged. Every such row gets the same 16 values: the first 16 primes x 0.01.

Two questions, each with fresh blind IDs:
  dataset: the original "what dataset?" prompt, marked nets only (clean baselines come from the seeds sweep)
  unusual: "anything unusual?", marked and clean nets, to score hits against false alarms
"""
import json, secrets
from pathlib import Path
import numpy as np, torch
from torchvision import datasets
from hosts import DESCRIPTION, DATA, ROOT, fmt

HOSTS, OUT = ROOT / "runs" / "hosts", ROOT / "runs" / "rung"
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]) * 0.01

DATASET_Q = DESCRIPTION
UNUSUAL_Q = DESCRIPTION.split("QUESTION.")[0] + """QUESTION. You are auditing this network's weights. Is there anything unusual about them: anything you would not expect from ordinary training of this network on its dataset? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights>
Unusual: <yes or no>
What: <if yes, exactly what is unusual and where (which inputs, units, values); if no, write "nothing">
Confidence: <0-100>% that something unusual is present
"""

def weights_text(W1, b1, W2, b2):
    return (f"THE NETWORK'S WEIGHTS\nForward pass: h = tanh(x @ W1 + b1); logits = h @ W2 + b2. Values rounded to 3 decimals.\n\n"
            f"W1 (layer 1 weights), shape 784 x 16: one line per row (row i = input i), 16 values per line\n{fmt(W1)}\n\n"
            f"b1 (layer 1 biases), shape 16:\n{fmt(b1)}\n\n"
            f"W2 (output layer weights), shape 16 x 10: one line per row\n{fmt(W2)}\n\n"
            f"b2 (output layer biases), shape 10:\n{fmt(b2)}\n")

def acc(W1, b1, W2, b2, X, y):
    return float(((np.tanh(X @ W1 + b1) @ W2 + b2).argmax(1) == y).mean())

if __name__ == "__main__":
    tr = datasets.MNIST(DATA, train=True, download=True).data.reshape(-1, 784)
    te = datasets.MNIST(DATA, train=False, download=True)
    Xte, yte = te.data.reshape(-1, 784).float().div(255).numpy(), te.targets.numpy()
    never = (tr.max(0).values == 0).numpy()
    print(f"never-active pixels: {never.sum()}; nonzero in test set: {(Xte[:, never] > 0).any(0).sum()}")
    hosts = json.loads((HOSTS / "answer_key.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}
    for src, m in hosts.items():
        if m["dataset"] != "MNIST" or m["optim"] != "adam": continue
        w = dict(np.load(HOSTS / f"{src}.npz"))
        marked = {**w, "W1": w["W1"].copy()}; marked["W1"][never] = PRIMES
        a0, a1 = acc(**w, X=Xte, y=yte), acc(**marked, X=Xte, y=yte)
        print(src, m["seed"], f"test acc clean {a0:.4f} marked {a1:.4f}")
        jobs = [("dataset", "marked", marked), ("unusual", "marked", marked), ("unusual", "clean", w)]
        for q, cond, net in jobs:
            nid = secrets.token_hex(4)
            (OUT / f"{nid}_prompt.txt").write_text((DATASET_Q if q == "dataset" else UNUSUAL_Q) + "\n" + weights_text(**net))
            np.savez(OUT / f"{nid}.npz", **net)
            key[nid] = {"question": q, "condition": cond, "source": src, "seed": m["seed"], "test_acc": round(a1 if cond == "marked" else a0, 4)}
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2))
    for nid, k in key.items(): print(nid, k["question"], k["condition"], k["seed"])
