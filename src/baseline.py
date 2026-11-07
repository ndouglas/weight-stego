"""Non-LLM baselines for the dataset question (MNIST vs Fashion-MNIST), trained on the model zoo (src/zoo.py).

Answers the objection "a 20-line classifier on first-layer row norms would solve this": does a learned weight reader
also inherit its training-recipe distribution?

Readers (both see only W1; features are invariant to permuting hidden units):
  rownorm  logistic regression on the 784 per-input mean |w| (standardized)
  rule     one feature, the reader's stated rule: count of inputs whose W1 row stays near init (max |w| < 0.05);
           threshold fitted on the training split ("many dead inputs -> MNIST")

Protocols (pre-registered in PREDICTIONS.md before this was run):
  P1  train on SGD nets (MNIST-sgd vs Fashion-sgd)          test on Adam nets, and on both laundering recipes
  P2  train on Adam nets                                     test on SGD nets, and on both laundering recipes
  P3  train on both optimizers, seeds 100-119                test on held-out seeds 120-129, the laundering recipes,
                                                             and the 16 clean backbone nets the LLMs read (seeds 0-4)
Reported: accuracy, and for laundering recipes the share called MNIST (truth: shift2 = MNIST, pad20 = Fashion).
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from hosts import ROOT

ZOO, HOSTS = ROOT / "runs" / "zoo", ROOT / "runs" / "hosts"
NEAR = 0.05
TRUTH = {"MNIST-adam": 1, "MNIST-sgd": 1, "Fashion-adam": 0, "Fashion-sgd": 0, "MNIST-sgd-shift2": 1, "Fashion-adam-pad20": 0}

def feats(W1): return np.abs(W1).mean(1)
def dead(W1): return float((np.abs(W1).max(1) < NEAR).sum())

def load(recipes, seeds):
    W = [np.load(ZOO / f"{r}_s{s}.npz")["W1"] for r in recipes for s in seeds]
    return W, np.array([TRUTH[r] for r in recipes for s in seeds])

class Rule:
    """Threshold on the dead-input count; direction and cut chosen to maximize training accuracy."""
    def fit(self, W, y):
        d = np.array([dead(w) for w in W]); best = (-1, 0, 1)
        for t in np.unique(d):
            for sgn in (1, -1):
                a = ((sgn * (d - t) >= 0).astype(int) == y).mean()
                if a > best[0]: best = (a, t, sgn)
        _, self.t, self.sgn = best; return self
    def predict(self, W): return (self.sgn * (np.array([dead(w) for w in W]) - self.t) >= 0).astype(int)

class RowNorm:
    def fit(self, W, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(np.array([feats(w) for w in W]), y); return self
    def predict(self, W): return self.m.predict(np.array([feats(w) for w in W]))

def evaluate(name, train, tests):
    rows = []
    for reader in (RowNorm, Rule):
        clf = reader().fit(*train)
        for tname, (W, y) in tests.items():
            p = clf.predict(W)
            rows.append(f"| {name} | {reader.__name__} | {tname} | {len(y)} | {(p == y).mean():.2f} | {p.mean():.2f} |")
    return rows

if __name__ == "__main__":
    S, ALL = range(100, 130), range(100, 130)
    launder = {"shift2 MNIST (truth MNIST)": load(["MNIST-sgd-shift2"], ALL), "pad20 Fashion (truth Fashion)": load(["Fashion-adam-pad20"], ALL)}
    hk = json.loads((HOSTS / "answer_key.json").read_text())
    read_nets = ([np.load(HOSTS / f"{n}.npz")["W1"] for n in sorted(hk)], np.array([int(hk[n]["dataset"] == "MNIST") for n in sorted(hk)]))
    out = ["| protocol | reader | test set | n | accuracy | share called MNIST |", "|---|---|---|---|---|---|"]
    out += evaluate("P1 train SGD", load(["MNIST-sgd", "Fashion-sgd"], S),
                    {"Adam nets": load(["MNIST-adam", "Fashion-adam"], S), **launder})
    out += evaluate("P2 train Adam", load(["MNIST-adam", "Fashion-adam"], S),
                    {"SGD nets": load(["MNIST-sgd", "Fashion-sgd"], S), **launder})
    out += evaluate("P3 train both", load(["MNIST-adam", "Fashion-adam", "MNIST-sgd", "Fashion-sgd"], range(100, 120)),
                    {"held-out seeds": load(["MNIST-adam", "Fashion-adam", "MNIST-sgd", "Fashion-sgd"], range(120, 130)),
                     **launder, "LLM-read backbone nets": read_nets})
    txt = "\n".join(out); print(txt)
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / "baseline.md").write_text(txt + "\n")
