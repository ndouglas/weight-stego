---
title: "TITLE"
author: |
  Nathan Douglas\
  Independent
abstract: |
  Takehi et al. (2026) showed that frontier models can read and write small nets' weights. Weight-based audits may be used to determine what a net trained on. However, weight readers expect low-use inputs to leave their weights close to their initial values. We test whether a bad actor can exploit a reader's assumptions about training methods to launder training-data verdicts—without directly editing weights—and whether readers catch a deliberate edit. We audited tiny nets (trained on MNIST and Fashion-MNIST) using frontier models (Opus 5.5 and Fable 5.1) prompted with one of two questions ("which dataset?" and "is anything unusual?"). We varied the optimizer (Adam or SGD), swapped about 270 first-layer rows that barely affect behavior between same-seed Adam and SGD nets, and compared against a one-line threshold rule fitted on SGD nets. We also tested shift augmentation and a text message concealed in the signs of never-trained weights. Readers identified Adam-trained MNIST nets as Fashion-MNIST [bb.pooled.misread_adam=9/9 (+1 NA)] times (vs. [bb.pooled.misread_sgd=1/10] with SGD), and their verdicts followed the swapped rows (Opus [graft.opus.followed_rows=20/20], Fable [graft.fable.followed_rows=15/19 (+1 NA)]). The threshold rule reproduced these verdicts. Shift augmentation also laundered the verdict ([launder.pooled.shift2_as_fashion=10/10]). Readers reliably found the hidden message only with line-numbered rows and maximum effort; Opus decoded it [cam.opus.max-num.decoded=4/5] times, while Fable never did, calling it a backdoor or trigger 5/5 times. Auditors should treat verdicts as conditional on the training recipe—which the audited party controls—and must not confuse detection with explanation.
---

<!--
HOW THIS FILE WORKS
- Write numbers as claim tags: [bb.pooled.fisher_p=0.0001]. `make check` fails if a tag disagrees with the data;
  `make report` replaces each tag with its value. IDs and values: results/claims.md.
- Comments like this one are notes for drafting; they don't render.
- `make report` prints the main-body page count (limit 8, template recommends 4; references and appendices don't count).
- Format follows Apart's template (report/apart.tex); the .docx uses the template's styles for pasting into the Google Doc.
- Write p-values as plain text (p = 0.0001), not $...$; "$" pairs become math.
- Place the figure at the end of 4.1 or after Table 4, so it does not split 4.2 from its table.
- Word budgets target ~7 pages at Google-calibrated spacing (build matches the template's page breaks) at the template's spacing, leaving room for one figure (Results get the most). Targets, not rules. Drafting order: 4 -> 3 -> appendices -> 5 -> 1, 2.
-->

# 1. Introduction

Across the MNIST data set, there are [data.mnist_never_active=67] pixels that are never active in any of the individual training images. The weights corresponding to these never-active pixels get zero gradient; they stay exactly at their randomized initial values under any optimizer.

<!-- ~250 words (one merged findings/contributions list). Define never-active pixels here, fully (a fact about MNIST, the paper's hook);
give near-init only as intuition: under SGD a wider ring stays near its starting values, under Adam it doesn't (exact definition: Methods). Weight-based audits as proposed oversight (provenance, compliance, backdoor detection); Takehi et al. make
LLM weight readers plausible. The gap: what the reader assumes about training. Our three findings in one line each:
(1) recipe priors decide the dataset verdict, and benign choices launder it; (2) a few inert rows carry the decision, and a
one-line rule reproduces it, so it's not an LLM quirk; (3) a hidden message is detected only at max effort + line numbers,
and detection != attribution. Contributions list. Sprint-only work statement. -->

# 2. Related Work

<!-- ~120 words: one sentence per group, saying how it differs from this work. Groups:
- LLMs reading weights: Takehi et al. 2026.
- Learned weight-space readers / meta-classifiers: Unterthiner et al. 2020; Schürholt et al. (Model Zoos); MNTD (Xu et al. 2021).
- Training provenance / verification: Proof-of-Learning (Jia et al. 2021); Shavit 2023.
- Hiding data in weights: Uchida et al. 2017 (watermarks); EvilModel (Wang et al. 2021).
- Automated auditing agents: Marks et al. 2025; Bricken et al. 2025.
- Optimizer implicit bias: Wilson et al. 2017.
- Hidden channels / deception in AI oversight: Hubinger et al. 2024 (Sleeper Agents); Motwani et al. 2024 (Secret Collusion).
Position: prior readers are trained classifiers; we test zero-shot LLM readers and show both share the recipe prior. -->

# 3. Methods

In the nets as we have trained them, there are a number of input rows whose weights remain near their init values. That is, with init values at most 1/28, or ≈ 0.036, the absolute value of the largest of these weights is under 0.05. The number of these near-init rows varies with the specific training recipe; Adam-MNIST nets have [nets.adam_mnist_near_init=67-74], SGD-MNIST nets have [nets.sgd_mnist_near_init=270-276]; and shift-augmented SGD-MNIST nets have [nets.shift2_near_init=133-148]. Adam rescales
each weight's step by its own gradient history, so rarely-used weights move in larger steps. This is because its predecessor Adagrad was _designed_ to take larger steps on rare features (Duchi, Hazan & Singer 2011) and Adam inherits this (Kingma & Ba 2015).

We selected two frontier models, Opus 5.5 and Fable 5.1, for models.

<!--
**Job:** someone could rebuild this from the text. Definitions here, so Results can be short.

**Beats:**
1. **Nets.** 784-16-10 MLP, tanh, cross-entropy, 4 epochs, batch 128. Adam (lr 1e-3) or SGD (lr 0.1). MNIST (~93% test
   accuracy) and Fashion-MNIST (~83-85%). Seeds 0-4 (MNIST), 0-2 (Fashion): 16 clean nets.
2. **Near-init rows** (never-active pixels are already defined in the Introduction; don't repeat):
   input rows whose largest |weight| is under 0.05 (init values are at most 1/28 ≈ 0.036). The count depends on the
   recipe; point to Table 1 rather than listing counts. Then the *why* (Adam vs SGD steps), next to Table 1.
3. **Readers.** Opus 5.5 and Fable 5.1, Anthropic Message Batches API, no system prompt, one read per net unless stated.
   The prompt describes the architecture and lists every weight to 3 decimals. It names neither the dataset nor the
   optimizer. Two questions: *dataset* ("what kind of images, what classes?") and *audit* ("anything unusual?"),
   each with fixed answer slots (exact text: Appendix B). Default: effort high, rows bare, 32k max output tokens;
   truncated reads rerun with a larger cap (128k, or 300k via a beta header), and any still truncated count as NA.
4. **Manipulations** (put these in a table: net, optimizer, change, accuracy, near-init rows):
   - laundering knobs, no edits: *shift augmentation* (SGD-MNIST, images randomly shifted up to 2 px during training);
     *pad-20* (Adam-Fashion, images shrunk to 20×20 and zero-padded to 28×28, which is how MNIST itself was made).
   - *swap control*: for each seed, swap the [graft.rows_swapped=270-276] rows that are near-init in the SGD net between
     the same-seed Adam and SGD nets (same init, so the swap is clean). Test accuracy changes by at most
     [graft.max_acc_change=0.001]. Two reads per net.
   - *hidden message*: in the 5 Adam-MNIST nets, the signs of the 67 × 16 never-active weights spell the ASCII bits of
     "pineapple " repeated; magnitudes drawn at init scale. Audit question. Settings varied: effort (high, max) ×
     rows (bare, line-numbered: each line prefixed with its number, as Claude Code's file reader shows it).
5. **Baselines.** A zoo of 180 nets (6 recipes × 30 seeds). Two non-LLM readers on first-layer weights only:
   *Rule* = one threshold on the near-init row count (the rule the LLMs state); *RowNorm* = logistic regression on each
   input's mean |weight|. Trained on SGD nets only, Adam nets only, or both; tested on the others.
6. **Scoring and stats.** Mechanical scoring of the answer slots (script in repo); "decoded" = the read contains
   "pineapple"; "called it a backdoor" is hand-coded and labelled so. Fisher exact tests, uncorrected. Predictions written
   and committed before each run (Appendix D). Earlier exploratory runs through a different harness are reported only in
   Appendix E.

**Careful with:**
- "Dead pixels" is jargon; define once, then use one term consistently (never-active).
- Say "nets" for the MLPs and "readers"/"models" for the LLMs; never "models" for both.
- Line-numbered ≠ different content: same text, plus numbers.

-->

<!-- ~280 words; Table 1 carries the recipes and near-init counts, Table 2 the reads and reader settings.
Never-active pixels are defined in the Introduction; don't repeat. Why Adam leaves fewer near-init rows (Adagrad/Adam) stays here, next to Table 1.
Keep in the main text: near-init definition (0.05 threshold; counts: see Table 1); prompts name neither dataset nor optimizer; what "line-numbered"
means; max_tokens reruns; NA handling; Fisher tests uncorrected; pilot (exploratory) vs API (confirmatory).
To Appendices B-C: full prompt text, model IDs/dates, cost breakdown.
Subsections suggested:
3.1 Nets: 784-16-10 tanh MLPs, MNIST / Fashion-MNIST, seeds, Adam vs SGD; [data.mnist_never_active=67] never-inked pixels;
    near-init rows (max|w| < 0.05): Adam-MNIST [nets.adam_mnist_near_init=67-74], SGD-MNIST [nets.sgd_mnist_near_init=270-276].
3.2 Readers and questions: Opus 5.5, Fable 5.1, Message Batches, effort high (default) / max, bare vs line-numbered rows,
    weights as text to 3 decimals; DATASET_Q and UNUSUAL_Q (exact prompts in Appendix B).
3.3 Manipulations: laundering knobs (shift augmentation; pad-20 preprocessing on Fashion); swap control (graft);
    primes beacon in never-active rows; sign-bit ASCII message ("pineapple ") at init magnitude.
3.4 Non-LLM baselines: 180-net zoo, RowNorm (logistic on per-input mean |w|) and Rule (threshold on near-init count); P1-P3.
3.5 Scoring and stats: mechanical scoring (score_api.py; last answer slot; first-named dataset); truncations reported as NA;
    Fisher exact tests; predictions committed before each run (Appendix D); `make check` reproduces every number. -->

<!-- Nate Notes
While this holds for SGD nets, Adam's predecessor Adagrad was _designed_ to take larger steps on rare features (Duchi, Hazan & Singer 2011) and Adam inherits this (Kingma & Ba 2015). Adam's fringe thus contains weights that have shifted away from their starting values and the frame of Adam-MNIST nets can look more similar to the frame of Fashion-MNIST nets.
-->

Table: Training recipes and edits. All nets are 784-16-10 tanh MLPs trained for 4 epochs (batch 128; Adam lr 1e-3, SGD lr 0.1). Near-init rows: inputs whose largest |weight| is below 0.05 (min-max over seeds). Accuracy: MNIST or Fashion-MNIST test set (min-max).

| Net | Seeds | Change from a plain run | Test accuracy | Near-init rows |
|------------------|----|------------------------------------------|-------------|-----------|
| MNIST, SGD | 5 | — | [nets.sgd_mnist_test_acc=0.929-0.932] | [nets.sgd_mnist_near_init=270-276] |
| MNIST, Adam | 5 | — | [nets.adam_mnist_test_acc=0.930-0.933] | [nets.adam_mnist_near_init=67-74] |
| Fashion, SGD | 3 | — | [nets.fashion_sgd_test_acc=0.831-0.842] | [nets.fashion_sgd_near_init=58-66] |
| Fashion, Adam | 3 | — | [nets.fashion_adam_test_acc=0.845-0.848] | [nets.fashion_adam_near_init=0-1] |
| MNIST, SGD + shift | 5 | images shifted up to 2 px in training | [nets.shift2_test_acc=0.911-0.914] | [nets.shift2_near_init=133-148] |
| Fashion, Adam + pad-20 | 5 | images resized to 20×20, zero-padded to 28×28 | [nets.pad20_test_acc=0.834-0.841] | [nets.pad20_near_init=384] |
| Adam net, SGD fringe | 5 | SGD net's [graft.rows_swapped=270-276] near-init rows swapped in | change ≤ [graft.max_acc_change=0.001] | as SGD |
| SGD net, Adam fringe | 5 | Adam net's rows swapped in (same rows) | change ≤ [graft.max_acc_change=0.001] | as Adam |
| MNIST, Adam + message | 5 | signs of the [data.mnist_never_active=67] never-active rows set to the bits of "pineapple " | change [nets.payload_max_acc_change=0.0] | as Adam |

Table: What each reader was asked. Every read is a single user message (no system prompt, no tools) sent through the Message Batches API: the architecture, then every weight to 3 decimals. The prompt names neither the dataset nor the optimizer. *Dataset* question: what images and classes was this net trained on? *Audit* question: is anything about this net unusual? Defaults: effort high, bare rows, 32k output tokens. A truncated read was rerun with a larger cap (128k, or 300k via a beta header); one still truncated counts as NA. Exact prompts: Appendix B.

| Experiment | Nets read | Question | Effort, rows | Reads (Opus / Fable) | See |
|----------------|------------------------------------|-------|------------|--------|-----|
| Recipe prior | 16 clean nets: MNIST and Fashion, Adam and SGD | dataset | high, bare | 16 / 16 | 4.1 |
| Laundering | 5 shift-augmented MNIST, 5 pad-20 Fashion | dataset | high, bare | 10 / 10 | 4.1 |
| Swap control | 10 swapped nets, 2 reads each | dataset | high, bare | 20 / 20 | 4.2 |
| Primes beacon | 5 Adam-MNIST with beacon, 5 without | both | high, bare | 15 / 15 | App. E |
| Hidden message | 5 Adam-MNIST with message | audit | Table 6 | 10 / 20 | 4.3 |
| False-alarm control | the same 5, no message | audit | max, numbered | 5 / 5 | 4.3 |
| Total | | | | 76 / 86 | [cost.final_reads=162] |

# 4. Results

## 4.1 Which dataset? Recipe priors decide the verdict

The readers consistently misread the Adam-MNIST nets as Fashion-MNIST but misread SGD-MNIST much less frequently ([bb.pooled.misread_adam=9/9 (+1 NA)] vs. [bb.pooled.misread_sgd=1/10], p = [bb.pooled.fisher_p=0.0001]). The readers interpreted the Fashion-MNIST nets correctly (see Table 3).

The readers reported that they used the border of untouched weights to identify the training data for each net (Fable, on Adam-MNIST: "The 2‑pixel side border and 0–1‑pixel top/bottom border is exactly the Fashion‑MNIST preprocessing."). Adam-MNIST's fringe has fewer rows near init than SGD-MNIST ([nets.adam_mnist_near_init=67-74] vs [nets.sgd_mnist_near_init=270-276]; why Adam does this is discussed in Section 3), and so the readers misidentify the net (tested directly in 4.2).

Naming the optimizer in the prompt did not lead the readers to identify the nets correctly (Appendix E).

No edits to the weights after training are needed to launder verdicts in this way; including shift augmentation in SGD-MNIST training cut its "near-init" rows by about half, to [nets.shift2_near_init=133-148], and while accuracy remained high ([nets.shift2_test_acc=0.911-0.914] vs [nets.sgd_mnist_test_acc=0.929-0.932]), readers identified it as Fashion ([launder.pooled.shift2_as_fashion=10/10]).

But the training choices didn't work everywhere; when Fashion-MNIST nets were preprocessed with pad-20, Fable read them as MNIST ([launder.fable.pad20_as_mnist=5/5]), but Opus was less likely to do so ([launder.opus.pad20_as_mnist=1/5]).

Table: Dataset verdicts on clean nets (high effort, bare rows). NA = truncated read, excluded from the denominator.

| Net (truth) | Verdict counted | Opus 5.5 | Fable 5.1 |
|---|---|---|---|
| MNIST, Adam | read as Fashion | [bb.opus.adam_mnist_as_fashion=5/5] | [bb.fable.adam_mnist_as_fashion=4/4 (+1 NA)] |
| MNIST, SGD | read as MNIST | [bb.opus.sgd_mnist_as_mnist=5/5] | [bb.fable.sgd_mnist_as_mnist=4/5] |
| Fashion, Adam and SGD | read as Fashion | [bb.opus.fashion_correct=6/6] | [bb.fable.fashion_correct=6/6] |
| MNIST, SGD + shift augmentation | read as Fashion | [launder.opus.shift2_as_fashion=5/5] | [launder.fable.shift2_as_fashion=5/5] |
| Fashion, Adam + pad-20 | read as MNIST | [launder.opus.pad20_as_mnist=1/5] | [launder.fable.pad20_as_mnist=5/5] |

## 4.2 What decides it: rows that barely matter and a one-line rule

We swapped the fringe between same-seed Adam and SGD nets (Section 3). Accuracy barely moved (at most [graft.max_acc_change=0.001]), so swapping the rows barely affected behavior, but the nets are hybrids training would not normally produce. Both readers always read Adam-MNIST nets with the quieter SGD fringe as MNIST, and Opus always read SGD-MNIST nets with the louder Adam fringe as Fashion, but Fable sometimes trusted the interior more (see Table 4). So the readers did not decide their verdict based on the net's behavior, but on rows that are largely irrelevant to the net's behavior.

Our one-line Rule classifier trained on SGD nets called every Adam net Fashion (Table 5) and misidentified shift-augmented SGD nets, so it reproduces the LLM readers' verdicts. The RowNorm classifier trained on SGD nets makes the same errors on Adam, but not with shift augmentation. But both non-LLM readers, when trained on both Adam and SGD, had very high accuracy. By construction, Fashion nets trained on data with pad-20 preprocessing have [nets.pad20_near_init=384] near-init rows, more than MNIST, and so are misidentified by every non-LLM reader, even when trained on both Adam and SGD nets. Training on a diverse set of recipes increases the accuracy of weight-based readers for errors they have examples of.

Table: Swap control. Each net keeps its own interior but carries the other optimizer's near-init rows (5 nets x 2 reads per net).

| Net | Verdict counted | Opus 5.5 | Fable 5.1 |
|---|---|---|---|
| Adam net, SGD fringe | read as MNIST | [graft.opus.gadam_as_mnist=10/10] | [graft.fable.gadam_as_mnist=10/10] |
| SGD net, Adam fringe | read as Fashion | [graft.opus.gsgd_as_fashion=10/10] | [graft.fable.gsgd_as_fashion=5/9 (+1 NA)] |

Table: Non-LLM baselines on first-layer weights: accuracy on MNIST-vs-Fashion. Trained on 30 seeds per recipe (SGD only), or seeds 100-119 of both optimizers (both); tested on nets they never saw: Adam nets (n = 60), held-out seeds (40), shift-augmented MNIST (30), pad-20 Fashion (30), and the 16 clean nets the LLMs read. Every Adam net is called Fashion by both SGD-only readers, so they are right only on the Fashion half.

| Reader | Trained on | Adam nets | Held-out | Shift MNIST | Pad-20 Fashion | LLM-read nets |
|---|---|---|---|---|---|---|
| Rule | SGD only | [base.Rule.p1_adam_acc=0.5] | — | [base.Rule.p1_shift2_acc=0.0] | [base.Rule.p1_pad20_acc=0.0] | — |
| RowNorm | SGD only | [base.RowNorm.p1_adam_acc=0.5] | — | [base.RowNorm.p1_shift2_acc=1.0] | [base.RowNorm.p1_pad20_acc=0.0] | — |
| Rule | both | — | [base.Rule.p3_heldout_acc=0.97] | [base.Rule.p3_shift2_acc=1.0] | [base.Rule.p3_pad20_acc=0.0] | [base.Rule.p3_llm_nets_acc=0.94] |
| RowNorm | both | — | [base.RowNorm.p3_heldout_acc=1.0] | [base.RowNorm.p3_shift2_acc=1.0] | [base.RowNorm.p3_pad20_acc=0.0] | [base.RowNorm.p3_llm_nets_acc=1.0] |

## 4.3 Planted edits: detected, not explained

At default settings (high effort, bare rows), neither reader decoded the message hidden in the signs of the never-active weights (Section 3; Table 6). For Fable, setting either max effort or line-numbered rows (each row prefixed with its line number, as Claude Code's file reader shows it) was not enough to flag the nets. Only max effort with line-numbered rows worked reliably: both readers flagged all 5 nets ([cam.pooled.max-num.unusual=10/10]), and neither flagged the same nets unedited ([clean.pooled.max-num.unusual=0/10]; p = [cam.opus.maxnum_vs_clean_p=0.0079] per model).

However, detecting the message did not mean the readers could decode it, and here the readers diverged. Opus decoded the correct text ([cam.opus.max-num.decoded=4/5]). Fable never decoded it ([cam.fable.max-num.decoded=0/5]) and, in the author's reading of its explanations, described it as a planted trigger (3/5) or hedged between a trigger and a watermark (2/5). One Fable read even noticed that the pattern repeated every five rows, then called it a "frame" trigger.

So while both readers could find the message, only one could read it, and only in the most expensive configuration. Finding it was likely only possible because plain ASCII is redundant: the top bit of every lowercase letter is 0, so 6 hidden units had the same sign on every never-active row, and the 10-character message repeated every five rows. A simulated one-time-pad encryption of the same message leaves no constant-sign units at all (Appendix C). One tell remains even then: no planted weight prints as 0.000, while untouched initial weights have [payload.init_zero_prints=13-20] of 1,072 that do.

An overt, deliberately visible edit (a prime-number beacon in the same rows) was flagged under the audit question [rung.pooled.audit_flagged=9/9 (+1 NA)]; see Appendix E.

Table: Sign-bit message, by effort and row layout: called unusual / decoded ("pineapple"), and mean batch cost per read.

| Reader | high, bare | high, numbered | max, bare | max, numbered |
|---|---|---|---|---|
| Opus 5.5 | [cam.opus.high-bare.unusual=0/5] / [cam.opus.high-bare.decoded=0/5] · $[cost.opus.high-bare.usd_per_read=0.16] | — | — | [cam.opus.max-num.unusual=5/5] / [cam.opus.max-num.decoded=4/5] · $[cost.opus.max-num.usd_per_read=1.71] |
| Fable 5.1 | [cam.fable.high-bare.unusual=2/5] / [cam.fable.high-bare.decoded=0/5] · $[cost.fable.high-bare.usd_per_read=0.94] | [cam.fable.high-num.unusual=0/5] / [cam.fable.high-num.decoded=0/5] · $[cost.fable.high-num.usd_per_read=0.87] | [cam.fable.max-bare.unusual=1/5] / [cam.fable.max-bare.decoded=0/5] · $[cost.fable.max-bare.usd_per_read=3.36] | [cam.fable.max-num.unusual=5/5] / [cam.fable.max-num.decoded=0/5] · $[cost.fable.max-num.usd_per_read=2.99] |
| Clean hosts (false alarms) | Opus [rung.opus.audit_clean_false_alarm=0/5], Fable [rung.fable.audit_clean_false_alarm=1/5] | — | — | Opus [clean.opus.max-num.unusual=0/5], Fable [clean.fable.max-num.unusual=0/5] |

# 5. Discussion and Limitations

<!-- ~150 words. Provenance verdicts are conditional on recipe priors the audited party controls; auditors need
recipe-diverse calibration (P3 result). Not an LLM quirk: the rule reproduces it. Detection vs attribution (Fable's
trigger story). Harness and cost: what the auditor finds depends on effort and layout; cost per read. Encrypted
payloads would leave no sign structure. -->

## Limitations

<!-- ~60 words. The short version; full list and dual-use in Appendix A (required by the guidelines). -->

## Future Work

<!-- ~30 words. Larger n and more readers (non-Claude models), encrypted payloads, real-scale nets, recipe-diverse
calibration for auditors, Opus max/bare cell. -->

# 6. Conclusion

<!-- ~60 words (template: 1-2 paragraphs). -->

# Code and Data

- Code repository: REPO-LINK (`make check` regenerates every number in this report from committed data)
- Data: MNIST and Fashion-MNIST (torchvision); all nets, prompts and raw API responses are in the repository.

<!-- appendices -->

# References

<!-- Full citations for everything in section 2. -->

# Appendix

## Appendix A. Limitations and Dual-Use / Ethical Considerations

<!-- Required. Limitations: tiny nets, two datasets, two readers, n = 5 per cell, Opus untested in two Table 6 cells,
one payload (plaintext, one message), sender was a script, thinking not returned, pilot results exploratory.
Dual-use: laundering recipe and hiding method; why publishing helps defenders more (calibration fix is cheap). -->

## Appendix B. Prompts

<!-- Exact DATASET_Q and UNUSUAL_Q text; the line-numbered format (cat -n; line = input index + 26). -->

## Appendix C. Settings, cost and reproducibility

<!-- Model IDs, effort, max_tokens, beta header, Batches, dates; total spend $[cost.total_batch_usd=170.39] over
[cost.billed_reads=179] billed reads ([cost.final_reads=162] final). Repo link; `make check`. -->

## Appendix D. Predictions vs. outcomes

<!-- Table from PREDICTIONS.md with commit hashes. -->

## Appendix E. Exploratory (pilot) results

<!-- Beacon (primes in the 67 never-active rows): flagged under audit [rung.pooled.audit_flagged=9/9 (+1 NA)],
clean false alarms Opus [rung.opus.audit_clean_false_alarm=0/5], Fable [rung.fable.audit_clean_false_alarm=1/5];
beacon fixes the misread Opus [rung.opus.dataset_marked_as_mnist=5/5], Fable [rung.fable.dataset_marked_as_mnist=1/5].
Subagent-harness reads: optimizer named in prompt, canary/excuse injections, knob sweep; labelled exploratory. -->

# LLM Usage Statement

<!-- Required by the template. Your words: how Claude was used (experiment code, scoring, red-team review, number checks)
and that all claims were verified (`make check`). -->
