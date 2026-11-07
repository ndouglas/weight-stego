"""One-off: mark.py and round2.py wrote prompts but not weights. Rebuild those nets from the hosts exactly as the
scripts do (same iteration order, same rng(7) stream for E2) and save {nid}.npz only if the rebuilt net reproduces
the committed prompt byte for byte. Prints any mismatch and saves nothing for it.
"""
import json
import numpy as np
from torchvision import datasets
from hosts import ROOT, DATA
from mark import PRIMES, weights_text
from round2 import camouflage, infect

if __name__ == "__main__":
    never = (datasets.MNIST(DATA, train=True, download=True).data.reshape(-1, 784).max(0).values == 0).numpy()
    hosts = json.loads((ROOT / "runs" / "hosts" / "answer_key.json").read_text())
    H = lambda src: dict(np.load(ROOT / "runs" / "hosts" / f"{src}.npz"))
    built = {}
    for nid, m in json.loads((ROOT / "runs" / "rung" / "answer_key.json").read_text()).items():
        w = H(m["source"])
        if m["condition"] == "marked": w["W1"] = w["W1"].copy(); w["W1"][never] = PRIMES
        built[("rung", nid)] = w
    # round2: replay the generator loop to consume rng(7) in the original order
    rng = np.random.default_rng(7); cam_by_src = {}
    for src, m in hosts.items():
        if m["optim"] != "adam" or m["dataset"] == "FashionMNIST": continue
        w = H(src); cam_by_src[src] = {**w, "W1": camouflage(w["W1"], never, rng)}
    for nid, m in json.loads((ROOT / "runs" / "round2" / "answer_key.json").read_text()).items():
        w = H(m["source"])
        if m["exp"] == "E2": w = cam_by_src[m["source"]]
        elif m["exp"] == "E3": w = {**w, "W1": infect(w["W1"], never)}
        built[("round2", nid)] = w
    ok = 0
    for (s, nid), w in built.items():
        p = (ROOT / "runs" / s / f"{nid}_prompt.txt").read_text()
        if p.endswith(weights_text(**w)): np.savez(ROOT / "runs" / s / f"{nid}.npz", **w); ok += 1
        else: print("MISMATCH", s, nid)
    print(f"rebuilt and saved {ok}/{len(built)}")
