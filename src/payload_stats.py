"""No-API control: what in the sign-bit payload is detectable at all?

Compares the never-active W1 rows (67 x 16) under four conditions, for the 5 Adam-MNIST hosts:
  init        untouched (these rows get zero gradient, so they are the PyTorch init)
  plaintext   the E2 payload: signs = bits of "pineapple " repeated (as read by the LLMs)
  random      signs = fresh random bits (same magnitudes as E2)
  encrypted   signs = plaintext bits XOR a random keystream (a one-time pad; same magnitudes)

Statistics a reader could compute by eye:
  const_units   hidden units whose sign is the same on every dead row
  max_bias      largest |share positive - 0.5| over units
  repeats       dead rows whose 16-sign pattern equals the pattern 5 dead rows earlier (the plaintext period)
  ks_p          Kolmogorov-Smirnov p-value of |w| against Uniform(0, 1/28) (the init magnitude distribution)
"""
import json
import numpy as np
from scipy.stats import kstest
from torchvision import datasets
from hosts import DATA, ROOT
from round2 import bits, INIT

def stats(rows):
    s = rows > 0
    share = s.mean(0)
    return dict(const_units=int(((share == 0) | (share == 1)).sum()), max_bias=round(float(np.abs(share - 0.5).max()), 2),
                repeats=int((s[5:] == s[:-5]).all(1).sum()),
                ks_p=round(float(kstest(np.abs(rows).ravel(), "uniform", args=(0, INIT)).pvalue), 3))

if __name__ == "__main__":
    tr = datasets.MNIST(DATA, train=True, download=True).data.reshape(-1, 784)
    never = (tr.max(0).values == 0).numpy(); k = int(never.sum())
    hosts = json.loads((ROOT / "runs" / "hosts" / "answer_key.json").read_text())
    pt = bits("pineapple ", k * 16).reshape(k, 16)
    out = ["| seed | condition | const_units | max_bias | repeats (period 5) | KS p vs init |", "|---|---|---|---|---|---|"]
    for nid, m in sorted(hosts.items(), key=lambda x: x[1]["seed"]):
        if m["dataset"] != "MNIST" or m["optim"] != "adam": continue
        rng = np.random.default_rng(1000 + m["seed"])
        init = np.load(ROOT / "runs" / "hosts" / f"{nid}.npz")["W1"][never]
        mags = rng.uniform(0.001, INIT, size=(k, 16))
        rnd = rng.integers(0, 2, size=(k, 16)); key = rng.integers(0, 2, size=(k, 16))
        for name, rows in (("init", init), ("plaintext", mags * np.where(pt == 1, 1, -1)),
                           ("random", mags * np.where(rnd == 1, 1, -1)), ("encrypted", mags * np.where((pt ^ key) == 1, 1, -1))):
            st = stats(rows)
            out.append(f"| {m['seed']} | {name} | {st['const_units']} | {st['max_bias']} | {st['repeats']} | {st['ks_p']} |")
    txt = "\n".join(out); print(txt)
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / "payload_stats.md").write_text(txt + "\n")
