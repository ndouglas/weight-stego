"""A model zoo for the non-LLM baseline: 30 nets per training recipe, seeds 100-129 (disjoint from the read nets, 0-4).

Recipes match the read nets exactly (round3.fit is bit-identical to hosts.train for the same seed and optimizer):
  MNIST-adam, MNIST-sgd, Fashion-adam, Fashion-sgd     the backbone 2x2
  MNIST-sgd-shift2                                     D1 laundering knob
  Fashion-adam-pad20                                   D2 laundering knob
Saves runs/zoo/{recipe}_s{seed}.npz (git-ignored; regenerate with `make zoo`) and runs/zoo/index.json with test accuracy.
"""
import json, sys
import numpy as np
from torchvision import datasets
from hosts import ROOT
from round3 import load, fit

OUT = ROOT / "runs" / "zoo"
SEEDS = range(100, 130)
RECIPES = {  # name: (dataset class, optimizer, pad20, shift)
    "MNIST-adam": (datasets.MNIST, "adam", False, 0),
    "MNIST-sgd": (datasets.MNIST, "sgd", False, 0),
    "Fashion-adam": (datasets.FashionMNIST, "adam", False, 0),
    "Fashion-sgd": (datasets.FashionMNIST, "sgd", False, 0),
    "MNIST-sgd-shift2": (datasets.MNIST, "sgd", False, 2),
    "Fashion-adam-pad20": (datasets.FashionMNIST, "adam", True, 0),
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    cache = {}
    for name, (cls, optim, pad20, shift) in RECIPES.items():
        k = (cls, pad20)
        if k not in cache: cache[k] = (*load(cls, True, pad20), *load(cls, False, pad20))
        for seed in SEEDS:
            f = f"{name}_s{seed}"
            if f in idx: continue
            net, acc = fit(*cache[k], optim, seed, shift=shift)
            np.savez(OUT / f"{f}.npz", **net)
            idx[f] = {"recipe": name, "seed": seed, "test_acc": round(acc, 4)}
            idx_path.write_text(json.dumps(idx, indent=1)); print(f, round(acc, 4), flush=True)
