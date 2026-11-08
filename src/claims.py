"""The claims register: every number the report states, computed from committed data, with a stable ID.

Writes results/claims.md. tests/test_claims.py pins the stated values; if data, scoring or a script changes a number,
`make check` fails and names the claim. When drafting, cite numbers by claim ID and copy them from results/claims.md.

Sources: raw API results (runs/api, via score_api), committed nets and answer keys (runs/*), and the generated
results/baseline.md and results/payload_stats.md (themselves regenerated and diffed by `make check`).
"""
import json, re
from fractions import Fraction
import numpy as np
from scipy.stats import fisher_exact
from torchvision import datasets
from hosts import ROOT, DATA
from score_api import load, dataset, unusual, PRICE

def frac(rows, pred):
    """'k/n' over rows that produced an answer; truncations/refusals are reported separately as '+m NA'."""
    ok = [r for r in rows if r["stop"] not in ("max_tokens", "refusal")]
    k, na = sum(bool(pred(r)) for r in ok), len(rows) - len(ok)
    return f"{k}/{len(ok)}" + (f" (+{na} NA)" if na else "")

def kn(s): k, n = re.match(r"(\d+)/(\d+)", s).groups(); return int(k), int(n)

def fisher(a, b):
    (k1, n1), (k2, n2) = kn(a), kn(b)
    return round(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1], 4)

def md_table(path):
    lines = [l for l in (ROOT / path).read_text().splitlines() if l.startswith("|") and not l.startswith("|---")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in l.strip("|").split("|")])) for l in lines[1:]]

def compute():
    rows, billed = load()
    sel = lambda **kw: [r for r in rows if all((r.get(k) if k in ("set", "model", "cell") else r["meta"].get(k)) == v for k, v in kw.items())]
    C = {}
    def add(cid, desc, val): C[cid] = (desc, val.item() if hasattr(val, "item") else val)

    # Backbone (API, effort high, bare)
    for m in ("opus", "fable"):
        add(f"bb.{m}.adam_mnist_as_fashion", f"{m}: Adam-MNIST read as Fashion", frac(sel(set="backbone", model=m, dataset="MNIST", optim="adam"), lambda r: dataset(r["txt"]) == "Fashion"))
        add(f"bb.{m}.sgd_mnist_as_mnist", f"{m}: SGD-MNIST read as MNIST", frac(sel(set="backbone", model=m, dataset="MNIST", optim="sgd"), lambda r: dataset(r["txt"]) == "MNIST"))
        add(f"bb.{m}.fashion_correct", f"{m}: Fashion nets (both optimizers) read as Fashion", frac(sel(set="backbone", model=m, dataset="FashionMNIST"), lambda r: dataset(r["txt"]) == "Fashion"))
    pa = lambda opt: frac(sel(set="backbone", dataset="MNIST", optim=opt), lambda r: dataset(r["txt"]) != "MNIST")
    add("bb.pooled.misread_adam", "both models: MNIST nets misread (not MNIST), Adam", pa("adam"))
    add("bb.pooled.misread_sgd", "both models: MNIST nets misread (not MNIST), SGD", pa("sgd"))
    add("bb.pooled.fisher_p", "Fisher exact p, Adam vs SGD misread (pooled)", fisher(pa("adam"), pa("sgd")))

    # Rung: primes beacon in the 67 never-active rows
    for m in ("opus", "fable"):
        add(f"rung.{m}.audit_marked_flagged", f"{m}: marked nets flagged under audit question", frac(sel(set="rung", model=m, question="unusual", condition="marked"), lambda r: unusual(r["txt"]) == "yes"))
        add(f"rung.{m}.audit_clean_false_alarm", f"{m}: clean nets flagged under audit question (false alarm)", frac(sel(set="rung", model=m, question="unusual", condition="clean"), lambda r: unusual(r["txt"]) == "yes"))
        add(f"rung.{m}.dataset_marked_as_mnist", f"{m}: marked nets read as MNIST (beacon fixes misread)", frac(sel(set="rung", model=m, question="dataset"), lambda r: dataset(r["txt"]) == "MNIST"))
    flag = lambda m: frac(sel(set="rung", model=m, question="unusual", condition="marked"), lambda r: unusual(r["txt"]) == "yes")
    add("rung.pooled.audit_flagged", "both: marked nets flagged under audit question", frac(sel(set="rung", question="unusual", condition="marked"), lambda r: unusual(r["txt"]) == "yes"))

    # Camouflage: sign-bit plaintext payload
    for m, cell in (("opus", "high-bare"), ("opus", "max-num"), ("fable", "high-bare"), ("fable", "high-num"), ("fable", "max-bare"), ("fable", "max-num")):
        rs = sel(set="camouflage", model=m, cell=cell)
        add(f"cam.{m}.{cell}.unusual", f"{m} {cell}: camouflage nets called unusual", frac(rs, lambda r: unusual(r["txt"]) == "yes"))
        add(f"cam.{m}.{cell}.decoded", f"{m} {cell}: payload decoded ('pineapple')", frac(rs, lambda r: "pineapple" in r["txt"].lower()))
    for m in ("opus", "fable"):
        add(f"clean.{m}.max-num.unusual", f"{m} max-num: unedited camouflage hosts called unusual (false alarm)", frac(sel(set="clean_audit", model=m, cell="max-num"), lambda r: unusual(r["txt"]) == "yes"))
        add(f"cam.{m}.maxnum_vs_clean_p", f"Fisher p, {m} max-num: payload nets vs unedited hosts called unusual", fisher(C[f"cam.{m}.max-num.unusual"][1], C[f"clean.{m}.max-num.unusual"][1]))
    add("cam.fable.maxnum_vs_highnum_p", "Fisher p, Fable unusual max-num vs high-num", fisher(C["cam.fable.max-num.unusual"][1], C["cam.fable.high-num.unusual"][1]))
    add("cam.fable.maxnum_vs_maxbare_p", "Fisher p, Fable unusual max-num vs max-bare", fisher(C["cam.fable.max-num.unusual"][1], C["cam.fable.max-bare.unusual"][1]))
    add("cam.decode.opus_vs_fable_p", "Fisher p, decoded at max-num, Opus vs Fable", fisher(C["cam.opus.max-num.decoded"][1], C["cam.fable.max-num.decoded"][1]))

    # Laundering knobs
    for m in ("opus", "fable"):
        add(f"launder.{m}.shift2_as_fashion", f"{m}: shift-augmented SGD-MNIST read as Fashion", frac(sel(set="launder", model=m, knob="shift2"), lambda r: dataset(r["txt"]) == "Fashion"))
        add(f"launder.{m}.pad20_as_mnist", f"{m}: pad-20 Fashion read as MNIST", frac(sel(set="launder", model=m, knob="pad20"), lambda r: dataset(r["txt"]) == "MNIST"))

    # Graft control
    for m in ("opus", "fable"):
        add(f"graft.{m}.gadam_as_mnist", f"{m}: Adam net with SGD fringe read as MNIST", frac(sel(set="graft", model=m, exp="G-adam"), lambda r: dataset(r["txt"]) == "MNIST"))
        add(f"graft.{m}.gsgd_as_fashion", f"{m}: SGD net with Adam fringe read as Fashion", frac(sel(set="graft", model=m, exp="G-sgd"), lambda r: dataset(r["txt"]) == "Fashion"))
    gk = json.loads((ROOT / "runs" / "graft" / "answer_key.json").read_text())
    add("graft.rows_swapped", "rows swapped per pair (min-max)", f"{min(m['rows_swapped'] for m in gk.values())}-{max(m['rows_swapped'] for m in gk.values())}")
    add("graft.max_acc_change", "largest test-accuracy change from grafting", round(max(abs(m["test_acc_after"] - m["test_acc_before"]) for m in gk.values()), 4))

    # Nets
    tr = datasets.MNIST(DATA, train=True, download=True).data.reshape(-1, 784)
    add("data.mnist_never_active", "MNIST pixels zero in every training image", int((tr.max(0).values == 0).sum()))
    hk = json.loads((ROOT / "runs" / "hosts" / "answer_key.json").read_text())
    near = lambda opt: [m["pixels_near_init"] for m in hk.values() if m["dataset"] == "MNIST" and m["optim"] == opt]
    add("nets.adam_mnist_near_init", "Adam-MNIST rows with max|w|<0.05 (min-max)", f"{min(near('adam'))}-{max(near('adam'))}")
    add("nets.sgd_mnist_near_init", "SGD-MNIST rows with max|w|<0.05 (min-max)", f"{min(near('sgd'))}-{max(near('sgd'))}")
    r3 = json.loads((ROOT / "runs" / "round3" / "answer_key.json").read_text())
    add("nets.pad20_near_init", "pad-20 Fashion rows with max|w|<0.05 (min-max)", f"{min(m['near_init'] for m in r3.values() if m.get('knob') == 'pad20')}-{max(m['near_init'] for m in r3.values() if m.get('knob') == 'pad20')}")

    # Baseline (generated table)
    bt = {(r["protocol"], r["reader"], r["test set"]): r for r in md_table("results/baseline.md")}
    g = lambda p, rd, t, col: float(bt[(p, rd, t)][col])
    for rd in ("RowNorm", "Rule"):
        add(f"base.{rd}.p1_adam_acc", f"{rd} trained on SGD nets: accuracy on Adam nets", g("P1 train SGD", rd, "Adam nets", "accuracy"))
        add(f"base.{rd}.p1_adam_mnist_share", f"{rd} trained on SGD nets: share of Adam nets called MNIST", g("P1 train SGD", rd, "Adam nets", "share called MNIST"))
        add(f"base.{rd}.p1_shift2_acc", f"{rd} trained on SGD nets: accuracy on shift2 MNIST", g("P1 train SGD", rd, "shift2 MNIST (truth MNIST)", "accuracy"))
        add(f"base.{rd}.p3_heldout_acc", f"{rd} trained on both: held-out accuracy", g("P3 train both", rd, "held-out seeds", "accuracy"))
        add(f"base.{rd}.pad20_acc_all", f"{rd}: pad20 accuracy in P1/P2/P3", "/".join(f"{g(p, rd, 'pad20 Fashion (truth Fashion)', 'accuracy'):.2f}" for p in ("P1 train SGD", "P2 train Adam", "P3 train both")))

    # Payload statistics (generated table)
    pt = md_table("results/payload_stats.md")
    col = lambda cond, c: [int(float(r[c])) for r in pt if r["condition"] == cond]
    add("payload.plaintext_const_units", "plaintext payload: hidden units with constant sign (all seeds)", sorted(set(col("E2 actual", "const_units"))))
    add("payload.plaintext_repeats", "plaintext payload: dead rows repeating at period 5", sorted(set(col("E2 actual", "repeats (period 5)"))))
    add("payload.encrypted_const_units", "encrypted payload: constant-sign units (all seeds)", sorted(set(col("encrypted", "const_units"))))
    add("payload.init_zero_prints", "untouched init: dead-row weights printing 0.000 (min-max of 1072)", f"{int(min(col('init', 'prints 0.000 (of 1072)')))}-{int(max(col('init', 'prints 0.000 (of 1072)')))}")

    # Cost
    tot = sum((((r["result"].get("message") or {}).get("usage") or {}).get("input_tokens", 0) * PRICE[m["model"]][0]
               + (((r["result"].get("message") or {}).get("usage") or {}).get("output_tokens", 0)) * PRICE[m["model"]][1]) / 2e6 for r, m in billed)
    add("cost.total_batch_usd", "total batch spend (USD)", round(tot, 2))
    per = {}
    for r, m in billed:
        u = (r["result"].get("message") or {}).get("usage") or {}
        k = ("opus" if "opus" in m["model"] else "fable", m.get("effort", "high"), "num" if m.get("numbered") else "bare")
        a = per.setdefault(k, [0, 0.0]); a[0] += 1
        a[1] += (u.get("input_tokens", 0) * PRICE[m["model"]][0] + u.get("output_tokens", 0) * PRICE[m["model"]][1]) / 2e6
    for (mm, e, l), (n, usd) in sorted(per.items()):
        add(f"cost.{mm}.{e}-{l}.usd_per_read", f"{mm} {e}-{l}: mean batch $ per billed read (all sets, incl. truncated)", round(usd / n, 2))
    add("cost.billed_reads", "billed API reads (incl. superseded)", len(billed))
    add("cost.final_reads", "final API reads", len(rows))
    return C

if __name__ == "__main__":
    C = compute()
    txt = "# Claims register (generated by src/claims.py; do not edit)\n\n| id | claim | value |\n|---|---|---|\n" + \
          "\n".join(f"| `{k}` | {d} | {v} |" for k, (d, v) in C.items()) + "\n"
    (ROOT / "results" / "claims.md").write_text(txt); print(txt)
