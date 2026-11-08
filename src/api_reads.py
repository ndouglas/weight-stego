"""Confirmatory reads through the Anthropic API (no agent harness, no tools, raw JSON kept).

Each prompt file is sent as the single user message, exactly as written to disk. No system prompt.
Settings are fixed and recorded with every result. Sampling parameters (temperature etc.) are not
accepted by these models, so runs are not seeded; repeat reads measure that variance instead.

No refusal fallbacks: a refusal is recorded as data, never silently answered by another model.

Usage:
  python src/api_reads.py estimate            # count tokens (free endpoint), print the bill
  python src/api_reads.py pilot               # one streamed read per model; prints real token usage (~$1)
  python src/api_reads.py submit [--models opus,fable] [--repeats N]   # one Message Batch (50% price)
  python src/api_reads.py collect BATCH_ID    # poll, then save raw results under runs/api/
"""
import argparse, json, os, sys, time
from pathlib import Path
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "api"

MODELS = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}
# USD per million tokens, standard first-party rates (claude-api skill table, cached 2026-10-06).
# Batches are billed at 50%. Thinking tokens are billed as output.
PRICE = {"claude-opus-5-5": (4.00, 20.00), "claude-fable-5-1": (10.00, 50.00)}
EFFORT = "high"
MAX_TOKENS = 32000

# The confirmatory set: prompt directories and the key that labels each prompt.
SETS = {
    "backbone": ("runs/hosts", lambda k: True),                                   # Adam vs SGD, MNIST and Fashion, 16 nets
    "rung": ("runs/rung", lambda k: True),                                        # primes beacon: dataset Q (marked), audit Q (marked + clean)
    "camouflage": ("runs/round2", lambda k: k["exp"] == "E2"),                    # sign-bit payload, audit Q
    "launder": ("runs/round3", lambda k: k["exp"] in ("D1", "D2")),               # shift augmentation; MNIST-style preprocessing
    "graft": ("runs/graft", lambda k: True),
    "clean_audit": ("runs/rung", lambda k: k["question"] == "unusual" and k["condition"] == "clean"),  # false-alarm control: the 5 camouflage hosts, unedited                                     # border rows swapped between same-seed Adam/SGD MNIST nets
}

def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def prompts(only=None):
    for name, (d, keep) in SETS.items():
        if only and name not in only: continue
        key = json.loads((ROOT / d / "answer_key.json").read_text())
        for nid, meta in sorted(key.items()):
            if keep(meta):
                yield name, nid, (ROOT / d / f"{nid}_prompt.txt").read_text(), meta

def numbered(text):
    """Present the prompt the way Claude Code's Read tool does (cat -n: right-aligned line number, tab)."""
    return "\n".join(f"{i:6d}\t{l}" for i, l in enumerate(text.split("\n"), 1))

def params(model, text, effort=None):
    return dict(model=model, max_tokens=MAX_TOKENS, output_config={"effort": effort or EFFORT},
                messages=[{"role": "user", "content": text}])

def estimate(client, models, repeats, out_guess, only=None):
    items = list(prompts(only))
    print(f"{len(items)} prompts x {len(models)} models x {repeats} repeats = {len(items)*len(models)*repeats} reads")
    total = 0.0
    for m in models:
        mid = MODELS[m]
        # Count every prompt once; prompts differ only in their numbers, so counts are close but not equal.
        counts = [client.messages.count_tokens(model=mid, messages=[{"role": "user", "content": t}]).input_tokens
                  for _, _, t, _ in items]
        inp = sum(counts) * repeats; out = out_guess[m] * len(items) * repeats
        pin, pout = PRICE[mid]
        std = inp / 1e6 * pin + out / 1e6 * pout
        print(f"  {mid}: input {inp:,} tok (per prompt {min(counts):,}-{max(counts):,}), "
              f"output ~{out:,} tok (guess {out_guess[m]:,}/read incl. thinking) -> standard ${std:,.2f}, batch ${std/2:,.2f}")
        total += std / 2
    print(f"  TOTAL at batch price: ${total:,.2f}  (output is the uncertain part; check one pilot read's usage first)")

def submit(client, models, repeats, only=None, ids=None, max_tokens=None, cells=None, beta=None):
    """cells: list of (effort, numbered) pairs; default is the original single cell (EFFORT, bare).
    Non-default cells get a suffix on the custom_id so they never supersede the original reads."""
    OUT.mkdir(parents=True, exist_ok=True)
    reqs, manifest = [], {}
    for name, nid, text, meta in prompts(only):
        for m in models:
            for effort, num in cells or [(EFFORT, False)]:
                for r in range(repeats):
                    cid = f"{name}-{nid}-{m}-r{r}"
                    if (effort, num) != (EFFORT, False): cid += f"-{effort}-{'num' if num else 'bare'}"
                    if ids and cid not in ids: continue
                    pr = params(MODELS[m], numbered(text) if num else text, effort)
                    if max_tokens: pr["max_tokens"] = max_tokens
                    reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**pr)))
                    manifest[cid] = {"set": name, "nid": nid, "model": MODELS[m], "repeat": r, "effort": effort,
                                     "numbered": num, "max_tokens": pr["max_tokens"], "meta": meta}
    # beta "output-300k-2026-03-24" lifts the output cap from 128k to 300k (Message Batches only)
    batch = client.messages.batches.create(requests=reqs, extra_headers={"anthropic-beta": beta} if beta else None)
    (OUT / f"{batch.id}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"submitted {len(reqs)} requests as {batch.id}; manifest in runs/api/")

def pilot(client, models):
    """One direct (streamed) read per model on the first prompt, to measure real output/thinking tokens."""
    OUT.mkdir(parents=True, exist_ok=True)
    name, nid, text, meta = next(prompts())
    for m in models:
        with client.messages.stream(**params(MODELS[m], text)) as stream:
            msg = stream.get_final_message()
        (OUT / f"pilot-{name}-{nid}-{m}.json").write_text(msg.model_dump_json(indent=2))
        print(f"{MODELS[m]}: stop={msg.stop_reason} usage={msg.usage}")

def collect(client, batch_id):
    while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"{b.processing_status}: {b.request_counts}", flush=True); time.sleep(60)
    d = OUT / batch_id; d.mkdir(parents=True, exist_ok=True)
    n = 0
    for res in client.messages.batches.results(batch_id):
        (d / f"{res.custom_id}.json").write_text(res.model_dump_json(indent=2)); n += 1
        if res.result.type == "succeeded":
            msg = res.result.message
            if msg.stop_reason == "refusal": print(f"REFUSAL {res.custom_id}: {msg.stop_details}")
            elif msg.stop_reason == "max_tokens": print(f"TRUNCATED {res.custom_id}")
        else:
            print(f"{res.result.type.upper()} {res.custom_id}")
    print(f"saved {n} raw results to runs/api/{batch_id}/")

if __name__ == "__main__":
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimate", "pilot", "submit", "collect"])
    ap.add_argument("batch_id", nargs="?")
    ap.add_argument("--models", default="opus,fable")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sets", default=None, help="comma-separated subset of: " + ",".join(SETS))
    ap.add_argument("--ids", default=None, help="comma-separated custom_ids to (re)submit")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--beta", default=None, help="anthropic-beta header, e.g. output-300k-2026-03-24")
    ap.add_argument("--cells", default=None, help="effort:bare|num pairs, e.g. high:num,max:bare,max:num")
    ap.add_argument("--out-guess", default="opus=4000,fable=8000", help="assumed output tokens per read (thinking included)")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    models = a.models.split(",")
    if a.cmd == "estimate":
        estimate(client, models, a.repeats, {k: int(v) for k, v in (p.split("=") for p in a.out_guess.split(","))},
                 a.sets.split(",") if a.sets else None)
    elif a.cmd == "pilot":
        pilot(client, models)
    elif a.cmd == "submit":
        submit(client, models, a.repeats, a.sets.split(",") if a.sets else None,
               set(a.ids.split(",")) if a.ids else None, a.max_tokens,
               [(c.split(":")[0], c.split(":")[1] == "num") for c in a.cells.split(",")] if a.cells else None, a.beta)
    else:
        if not a.batch_id: sys.exit("collect needs a batch id")
        collect(client, a.batch_id)
