"""Graft control for the Adam misread: does the dataset verdict follow the border rows?

For each seed 0-4, the Adam-MNIST and SGD-MNIST hosts share their initialization (same torch seed, same architecture).
B = the input rows that SGD leaves near init (max |w| < 0.05 in that seed's SGD net): the rarely- or never-active pixels,
mostly the border. Swap only those rows between the two nets of the same seed:

  G-adam  Adam-MNIST with SGD's B rows   -> an Adam net whose border looks untrained   (prediction if the border drives it: MNIST)
  G-sgd   SGD-MNIST with Adam's B rows   -> an SGD net whose border looks trained      (prediction if the border drives it: Fashion)

Everything else (the interior rows, b1, W2, b2) is untouched, so global statistics of the interior stay those of the
original optimizer. Test accuracy before and after is recorded in the answer key. Dataset question, blind IDs.
"""
import json, secrets
import numpy as np
from torchvision import datasets
from hosts import DATA, ROOT
from mark import DATASET_Q, weights_text, acc

HOSTS, OUT = ROOT / "runs" / "hosts", ROOT / "runs" / "graft"
NEAR = 0.05

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    te = datasets.MNIST(DATA, train=False, download=True)
    X, y = te.data.reshape(-1, 784).float().div(255).numpy(), te.targets.numpy()
    hosts = json.loads((HOSTS / "answer_key.json").read_text())
    by = {(m["optim"], m["seed"]): nid for nid, m in hosts.items() if m["dataset"] == "MNIST"}
    key = {}
    for seed in range(5):
        a, s = (dict(np.load(HOSTS / f"{by[(o, seed)]}.npz")) for o in ("adam", "sgd"))
        B = np.abs(s["W1"]).max(1) < NEAR
        for name, base, donor, src in (("G-adam", a, s, by[("adam", seed)]), ("G-sgd", s, a, by[("sgd", seed)])):
            W1 = base["W1"].copy(); W1[B] = donor["W1"][B]
            net = {**base, "W1": W1}
            nid = secrets.token_hex(4)
            (OUT / f"{nid}_prompt.txt").write_text(DATASET_Q + "\n" + weights_text(**net))
            np.savez(OUT / f"{nid}.npz", **net)
            key[nid] = {"exp": name, "dataset": "MNIST", "optim": "adam" if name == "G-adam" else "sgd", "seed": seed,
                        "source": src, "rows_swapped": int(B.sum()),
                        "test_acc_before": round(acc(**base, X=X, y=y), 4), "test_acc_after": round(acc(**net, X=X, y=y), 4),
                        "border_max_abs_after": round(float(np.abs(W1[B]).max()), 3)}
            print(nid, key[nid], flush=True)
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2))
