---
title: "TITLE"
author: "Nathan Douglas (independent)"
---

<!--
HOW THIS FILE WORKS
- Write numbers as claim tags: [bb.pooled.fisher_p=0.0001]. `make check` fails if a tag disagrees with the data;
  `make report` replaces each tag with its value. IDs and values: results/claims.md.
- Comments like this one are notes for drafting; they don't render.
- `make report` prints the main-body page count (limit 8; references and appendices don't count).
- Word budgets (~500 words/page minus tables) are targets, not rules. Drafting order: 4 -> 3 -> appendices -> 5 -> 1, 2.
-->

## Abstract

Takehi et al. (2026) showed that frontier models can read and write small nets' weights. Weight-based audits may be used to determine what a net trained on. However, weight readers expect low-use inputs to leave their weights close to their initial values. We test whether a bad actor can exploit a reader's assumptions about training methods to launder training-data verdicts—without directly editing weights—and whether readers catch a deliberate edit. We audited tiny nets (trained on MNIST and Fashion-MNIST) using frontier models (Opus 5.5 and Fable 5.1) prompted with one of two questions ("which dataset?" and "is anything unusual?"). We varied the optimizer (Adam or SGD), swapped about 270 first-layer rows that barely affect behavior between same-seed Adam and SGD nets, and compared against a one-line threshold rule fitted on SGD nets. We also tested shift augmentation and a text message concealed in the signs of never-trained weights. Readers identified Adam-trained MNIST nets as Fashion-MNIST 9/9 times (vs. 1/10 with SGD), and their verdicts followed the swapped rows (Opus 20/20, Fable 15/19). The threshold rule reproduced these verdicts. Shift augmentation also laundered the verdict (10/10). Readers reliably found the hidden message only with line-numbered rows and maximum effort; Opus decoded it 4/5 times, while Fable never did, calling it a backdoor or trigger 5/5 times. Auditors should treat verdicts as conditional on the training recipe—which the audited party controls—and must not confuse detection with explanation.

## 1. Introduction

<!-- ~400 words. Weight-based audits as proposed oversight (provenance, compliance, backdoor detection); Takehi et al. make
LLM weight readers plausible. The gap: what the reader assumes about training. Our three findings in one line each:
(1) recipe priors decide the dataset verdict, and benign choices launder it; (2) a few inert rows carry the decision, and a
one-line rule reproduces it, so it's not an LLM quirk; (3) a hidden message is detected only at max effort + line numbers,
and detection != attribution. Contributions list. Sprint-only work statement. -->

## 2. Related work

<!-- ~250 words. Groups:
- LLMs reading weights: Takehi et al. 2026.
- Learned weight-space readers / meta-classifiers: Unterthiner et al. 2020; Schürholt et al. (Model Zoos); MNTD (Xu et al. 2021).
- Training provenance / verification: Proof-of-Learning (Jia et al. 2021); Shavit 2023.
- Hiding data in weights: Uchida et al. 2017 (watermarks); EvilModel (Wang et al. 2021).
- Automated auditing agents: Marks et al. 2025; Bricken et al. 2025.
- Optimizer implicit bias: Wilson et al. 2017.
- Hidden channels / deception in AI oversight: Hubinger et al. 2024 (Sleeper Agents); Motwani et al. 2024 (Secret Collusion).
Position: prior readers are trained classifiers; we test zero-shot LLM readers and show both share the recipe prior. -->

## 3. Method

<!-- ~600 words, plus Table S (settings) if it fits, else Appendix C. Subsections suggested:
3.1 Nets: 784-16-10 tanh MLPs, MNIST / Fashion-MNIST, seeds, Adam vs SGD; [data.mnist_never_active=67] never-inked pixels;
    near-init rows (max|w| < 0.05): Adam-MNIST [nets.adam_mnist_near_init=67-74], SGD-MNIST [nets.sgd_mnist_near_init=270-276].
3.2 Readers and questions: Opus 5.5, Fable 5.1, Message Batches, effort high (default) / max, bare vs line-numbered rows,
    weights as text to 3 decimals; DATASET_Q and UNUSUAL_Q (exact prompts in Appendix B).
3.3 Manipulations: laundering knobs (shift augmentation; pad-20 preprocessing on Fashion); swap control (graft);
    primes beacon in never-active rows; sign-bit ASCII message ("pineapple ") at init magnitude.
3.4 Non-LLM baselines: 180-net zoo, RowNorm (logistic on per-input mean |w|) and Rule (threshold on near-init count); P1-P3.
3.5 Scoring and stats: mechanical scoring (score_api.py; last answer slot; first-named dataset); truncations reported as NA;
    Fisher exact tests; predictions committed before each run (Appendix D); `make check` reproduces every number. -->

## 4. Results

### 4.1 Which dataset? Recipe priors decide the verdict

<!-- ~300 words around Table 1. Key numbers: pooled misread [bb.pooled.misread_adam=9/9 (+1 NA)] vs [bb.pooled.misread_sgd=1/10],
p = [bb.pooled.fisher_p=0.0001]. Shift augmentation launders on both models; pad-20 on Fable only. One quote at most. -->

Table: Dataset verdicts on clean nets (high effort, bare rows). NA = truncated read, excluded from the denominator.

| Net (truth) | Verdict counted | Opus 5.5 | Fable 5.1 |
|---|---|---|---|
| MNIST, Adam | read as Fashion | [bb.opus.adam_mnist_as_fashion=5/5] | [bb.fable.adam_mnist_as_fashion=4/4 (+1 NA)] |
| MNIST, SGD | read as MNIST | [bb.opus.sgd_mnist_as_mnist=5/5] | [bb.fable.sgd_mnist_as_mnist=4/5] |
| Fashion, Adam and SGD | read as Fashion | [bb.opus.fashion_correct=6/6] | [bb.fable.fashion_correct=6/6] |
| MNIST, SGD + shift augmentation | read as Fashion | [launder.opus.shift2_as_fashion=5/5] | [launder.fable.shift2_as_fashion=5/5] |
| Fashion, Adam + pad-20 | read as MNIST | [launder.opus.pad20_as_mnist=1/5] | [launder.fable.pad20_as_mnist=5/5] |

### 4.2 What decides it: a few inert rows, and a one-line rule

<!-- ~350 words around Table 2. Swap control: [graft.rows_swapped=270-276] rows per pair, accuracy change at most
[graft.max_acc_change=0.001]. Baseline: Rule and RowNorm fitted on SGD nets call every Adam net Fashion (share called MNIST
[base.Rule.p1_adam_mnist_share=0.0]); with both optimizers in training, held-out accuracy [base.RowNorm.p3_heldout_acc=1.0] /
[base.Rule.p3_heldout_acc=0.97]. RowNorm gets shift right ([base.RowNorm.p1_shift2_acc=1.0]); Rule doesn't
([base.Rule.p1_shift2_acc=0.0]). Pad-20 fools every non-LLM reader by construction ([nets.pad20_near_init=384-384] near-init rows). -->

Table: Swap control. Each net keeps its own interior but carries the other optimizer's near-init rows (2 reads per net).

| Net | Verdict counted | Opus 5.5 | Fable 5.1 |
|---|---|---|---|
| Adam-MNIST with SGD rows | read as MNIST | [graft.opus.gadam_as_mnist=10/10] | [graft.fable.gadam_as_mnist=10/10] |
| SGD-MNIST with Adam rows | read as Fashion | [graft.opus.gsgd_as_fashion=10/10] | [graft.fable.gsgd_as_fashion=5/9 (+1 NA)] |

### 4.3 Planted edits: detected, not explained

<!-- ~450 words around Table 3. Beacon (primes in the 67 never-active rows): flagged under audit [rung.pooled.audit_flagged=9/9 (+1 NA)],
clean false alarms Opus [rung.opus.audit_clean_false_alarm=0/5], Fable [rung.fable.audit_clean_false_alarm=1/5].
Sign-bit message: the pilot-vs-API gap paragraph; Table 3; Fisher p Fable max-num vs high-num [cam.fable.maxnum_vs_highnum_p=0.0079];
decode Opus vs Fable p = [cam.decode.opus_vs_fable_p=0.0476]; Fable called it a planted trigger in all 5 (hand-coded); Opus's "pi"
fragments. Payload statistics: plaintext left 6 constant-sign units and 62 period-5 repeats
on every seed; a one-time pad leaves 0 (payload.* claims; list-valued, so not taggable as written). -->

Table: Sign-bit message, by effort and row layout: called unusual / decoded ("pineapple"), and mean batch cost per read.

| Reader | high, bare | high, numbered | max, bare | max, numbered |
|---|---|---|---|---|
| Opus 5.5 | [cam.opus.high-bare.unusual=0/5] / [cam.opus.high-bare.decoded=0/5] · $[cost.opus.high-bare.usd_per_read=0.16] | — | — | [cam.opus.max-num.unusual=5/5] / [cam.opus.max-num.decoded=4/5] · $[cost.opus.max-num.usd_per_read=1.41] |
| Fable 5.1 | [cam.fable.high-bare.unusual=2/5] / [cam.fable.high-bare.decoded=0/5] · $[cost.fable.high-bare.usd_per_read=0.94] | [cam.fable.high-num.unusual=0/5] / [cam.fable.high-num.decoded=0/5] · $[cost.fable.high-num.usd_per_read=0.87] | [cam.fable.max-bare.unusual=1/5] / [cam.fable.max-bare.decoded=0/5] · $[cost.fable.max-bare.usd_per_read=3.36] | [cam.fable.max-num.unusual=5/5] / [cam.fable.max-num.decoded=0/5] · $[cost.fable.max-num.usd_per_read=3.12] |

## 5. Discussion

<!-- ~450 words. Provenance verdicts are conditional on recipe priors the audited party controls; auditors need
recipe-diverse calibration (P3 result). Not an LLM quirk: the rule reproduces it. Detection vs attribution (Fable's
trigger story). Harness and cost: what the auditor finds depends on effort and layout; cost per read. Encrypted
payloads would leave no sign structure. -->

## 6. Conclusion

<!-- ~150 words, plus one-sentence pointer to Appendix A. -->

<!-- appendices -->

## References

<!-- Full citations for everything in section 2. -->

## Appendix A. Limitations and Dual-Use / Ethical Considerations

<!-- Required. Limitations: tiny nets, two datasets, two readers, n = 5 per cell, Opus untested in two Table 3 cells,
one payload (plaintext, one message), sender was a script, thinking not returned, pilot results exploratory.
Dual-use: laundering recipe and hiding method; why publishing helps defenders more (calibration fix is cheap). -->

## Appendix B. Prompts

<!-- Exact DATASET_Q and UNUSUAL_Q text; the line-numbered format (cat -n; line = input index + 26). -->

## Appendix C. Settings, cost and reproducibility

<!-- Model IDs, effort, max_tokens, beta header, Batches, dates; total spend $[cost.total_batch_usd=146.44] over
[cost.billed_reads=169] billed reads ([cost.final_reads=152] final). Repo link; `make check`. -->

## Appendix D. Predictions vs. outcomes

<!-- Table from PREDICTIONS.md with commit hashes. -->

## Appendix E. Exploratory (pilot) results

<!-- Subagent-harness reads: optimizer named in prompt, canary/excuse injections, knob sweep; labelled exploratory. -->
