# Writing outline: training wheels for report.md

<!-- This file is checked by `make check` like the report: every `[id=value]` tag must match results/claims.md. Copy tags
into report.md as they are; the build turns them into plain numbers. Not built into the PDF. -->

How to use it: each section has its **job** (the one thing the reader should leave with), **beats** in order (facts and
moves, deliberately not sentences), the **numbers** as tags, and **careful with** (wording that would be wrong or
overclaim). Budgets are words of prose, not counting tables.

The whole story, ELI5: a small network's weights are a record of how it was trained. Frontier models can read that record
and guess the dataset. They do it with a rule of thumb ("pixels that are always black never get trained, so their weights
stay at their random starting values; lots of those means MNIST"). The rule of thumb is about the *optimizer* as much
as the data, so changing how you train (not what you train on) changes the verdict. A one-line computer rule makes the
same mistakes, so this is about weight reading in general, not an LLM quirk. Separately: if you hide a message in those
untrained weights, readers find it only when they think hard and can see row numbers, and even then one of them
explains it wrongly.

---

## 1. Introduction (~230 words)

**Job:** why anyone should care that an auditor's verdict depends on training choices.

**Beats:**
1. The promise: auditing a model by reading its weights (what it was trained on; whether it was tampered with).
   Proposed for provenance and compliance; Takehi et al. (2026) make LLM weight readers real.
2. The catch: a reader needs a mental model of how training leaves marks. Whoever trained the net controls those marks.
3. Your question, in plain terms: can ordinary training choices change the verdict? And can readers catch a deliberate edit?
   Define the two attack routes once, here: *editing* weights after training (what tampering audits look for; leaves an
   artifact) vs *choosing the recipe* (every weight is a genuine product of training). "Laundering" = the true origin
   passes through individually legitimate choices (Adam is the default; shift augmentation is textbook) and the verdict
   comes out pointing elsewhere. What is laundered is the *verdict about the data*, not the data.
4. One merged list of findings-as-contributions (3 items):
   - a benign training choice flips the dataset verdict (Adam vs SGD; shift augmentation), with no weight edits;
   - the verdict is decided by a few hundred rows that barely matter to behaviour, and a one-line rule reproduces it;
   - a hidden message is detected only at max effort with line numbers, and detection is not explanation.
5. One line: all work done during the sprint; code and every number reproducible (`make check`).

**Careful with:**
- "Audits are used for..." is an overclaim: say *proposed* or *could be used*.
- Don't promise generality to big models; this is tiny nets (say "small" early so it doesn't feel hidden).
- "Bad actor" is fine for the threat model; the knobs themselves are benign and common.

## 2. Related Work (~120 words)

**Job:** show you know the neighbours, and say in one clause each how this differs.

**Beats (one sentence per group):**
1. LLMs reading weights: Takehi et al. 2026, the setup this builds on.
2. Learned weight-space readers (meta-classifiers predicting properties from weights): Unterthiner et al. 2020; Schürholt
   et al. (Model Zoos); MNTD (Xu et al. 2021). Difference: they're trained classifiers; you test zero-shot LLM readers and
   show both inherit the training-recipe prior (your baseline).
3. Proving how a model was trained: Proof-of-Learning (Jia et al. 2021); Shavit 2023 (compute governance). Difference:
   they assume a cooperative prover with logs; you look at what weights alone reveal.
4. Hiding data in weights: Uchida et al. 2017 (watermarks); EvilModel (Wang et al. 2021). Difference: you ask whether an
   LLM auditor finds it.
5. Automated auditing agents: Marks et al. 2025; Bricken et al. 2025. Your result is a calibration warning for them.
6. Optimizers leave different solutions: Wilson et al. 2017; adaptive steps for rare features: Duchi et al. 2011,
   Kingma & Ba 2015 (the mechanism behind finding 1).

**Careful with:** check each citation's year/venue when you write References; don't claim "first" anything.

## 3. Methods (~300 words + Table 1, the recipe table)

**Job:** someone could rebuild this from the text. Definitions here, so Results can be short.

**Beats:**
1. **Nets.** 784-16-10 MLP, tanh, cross-entropy, 4 epochs, batch 128. Adam (lr 1e-3) or SGD (lr 0.1). MNIST (~93% test
   accuracy) and Fashion-MNIST (~83-85%). Seeds 0-4 (MNIST), 0-2 (Fashion): 16 clean nets.
2. **Two definitions** (the whole paper hangs on them):
   - *never-active pixels*: zero in every MNIST training image; there are [data.mnist_never_active=67]. Their weights get
     zero gradient, so they stay exactly at their random initial values under any optimizer.
   - *near-init rows*: input rows whose largest |weight| is under 0.05 (init values are at most 1/28 ≈ 0.036).
     Adam-MNIST nets have [nets.adam_mnist_near_init=67-74]; SGD-MNIST nets have [nets.sgd_mnist_near_init=270-276]; shift-augmented
     SGD-MNIST nets have [nets.shift2_near_init=133-148].
   ELI5 of why: SGD's step is proportional to the gradient, so pixels inked in very few images barely move. Adam rescales
   each weight's step by its own gradient history, so rarely-used weights still take full-size steps. Cite Duchi, Hazan &
   Singer 2011 (Adagrad: adaptive methods were designed to give rare features larger steps) and Kingma & Ba 2015 (Adam).
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

## 4. Results

### 4.1 Which dataset? (~220 words, Table 1)

**Job:** training choices, not data, decide the verdict.

**Beats:**
1. Headline: Adam-MNIST read as Fashion by both readers; SGD-MNIST read correctly. Fashion nets always read correctly.
2. What the readers say they're doing: reading the untouched border as MNIST's 20x20-in-28x28 preprocessing; Adam's fringe
   rows don't look untouched, so the frame shrinks to Fashion-MNIST's. Real quotes (pick one): Fable on SGD-MNIST, "The
   empty 2–4 pixel frame with untouched init weights rules out Fashion‑MNIST"; Fable on Adam-MNIST, "The 2‑pixel side
   border and 0–1‑pixel top/bottom border is exactly the Fashion‑MNIST preprocessing".
3. Laundering with no edits: shift augmentation turns SGD-MNIST into "Fashion" for both readers; pad-20 turns
   Adam-Fashion into "MNIST" for Fable but mostly not Opus.
4. Pre-empt the obvious fix: in exploratory runs, telling the reader the optimizer didn't help (Appendix E).

**Numbers:** pooled misread [bb.pooled.misread_adam=9/9 (+1 NA)] vs [bb.pooled.misread_sgd=1/10],
p = [bb.pooled.fisher_p=0.0001]. Shift: Opus [launder.opus.shift2_as_fashion=5/5], Fable
[launder.fable.shift2_as_fashion=5/5]. Pad-20: Opus [launder.opus.pad20_as_mnist=1/5], Fable
[launder.fable.pad20_as_mnist=5/5].

**Careful with:**
- The 9/9 is *both models pooled*; one Fable read was truncated (NA).
- Fable's one SGD miss ([bb.fable.sgd_mnist_as_mnist=4/5]) belongs in the table, not hidden.
- "Fooled" implies intent by someone; for the knobs "the verdict flipped" is more exact.
- "No edits": say what it means once (weights are exactly what training produced). Don't claim the knob nets would pass a
  tampering audit: the audit question was never run on them. "There is no tampering to find" is an argument, not a result.
- The knobs flip the verdict; they don't hide it everywhere (pad-20: Fable 5/5, Opus 1/5).

### 4.2 What decides it (~280 words, Table 2 + small baseline table)

**Job:** the verdict lives in a few behaviour-irrelevant rows, and it's a property of weight reading, not of LLMs.
The swap is an *edit*, so it's the control, not the attack: it answers "which weights decide?" (the fringe), which
explains why the no-edit knobs work.

**Beats:**
1. Swap control: give an Adam net the SGD net's quiet fringe → read as MNIST; give an SGD net Adam's busy fringe → read
   as Fashion. Behaviour unchanged (accuracy ≤0.1 point). So the rows that decide the verdict are ones that barely
   affect what the net does.
2. Opus followed the rows every time; Fable mostly (it sometimes trusted the interior instead).
3. Baseline: fit the one-line Rule on SGD nets → it calls every Adam net Fashion, and it gets shift and pad-20 wrong too:
   it reproduces the LLM verdicts. RowNorm trained on SGD makes the same Adam error. Trained on both optimizers, both
   readers are nearly perfect on held-out nets. → The fix is recipe-diverse calibration.
4. Pad-20 fools every non-LLM reader whatever it's trained on: by construction it has as many near-init rows as MNIST.
5. Caveats in the text: swapped nets are slightly off-distribution; the 2 reads per net aren't independent (5 net pairs
   per model; "20/20" = 10 nets × 2 reads).

**Numbers:** Opus [graft.opus.gadam_as_mnist=10/10] and [graft.opus.gsgd_as_fashion=10/10]; Fable
[graft.fable.gadam_as_mnist=10/10] and [graft.fable.gsgd_as_fashion=5/9 (+1 NA)]. Rule fitted on SGD: share of Adam nets
called MNIST [base.Rule.p1_adam_mnist_share=0.0], shift accuracy [base.Rule.p1_shift2_acc=0.0]. RowNorm fitted on SGD:
Adam accuracy [base.RowNorm.p1_adam_acc=0.5] (calls them all Fashion), shift accuracy [base.RowNorm.p1_shift2_acc=1.0].
Trained on both: held-out [base.RowNorm.p3_heldout_acc=1.0] / [base.Rule.p3_heldout_acc=0.97]. Pad-20 accuracy, all
protocols: [base.Rule.pad20_acc_all=0.00/0.00/0.00]. Pad-20 near-init rows: [nets.pad20_near_init=384].

**Careful with:**
- RowNorm gets shift *right*; only Rule reproduces *every* LLM verdict. Don't say "the baselines" for that claim.
- RowNorm's 0.5 isn't "can't tell them apart": it ranks them perfectly but its cutoff is set for SGD (a calibration
  failure). One clause if room, else Appendix.
- "Inert" overclaims slightly; "barely affect behaviour" is exact.

### 4.3 Planted edits: detected, not explained (~280 words, Table 3)

**Job:** finding something and understanding it are different, and finding depends on how hard and how the reader looks.

**Beats:**
1. Setup in one line (definitions are in Methods).
2. Default setting (high effort, bare rows): neither reader decodes it; flags are rare and not about the message.
3. Only max effort + line-numbered rows works: both flag all 5, and neither flags the same nets unedited. Max effort
   alone, or line numbers alone, doesn't.
4. Explanation splits: Opus decodes "pineapple" 4/5; Fable decodes 0/5 and describes a planted trigger/backdoor every
   time (hand-coded).
5. Cost: the setting that works costs several times more per read.
6. Why it's findable at all: plain ASCII is redundant (constant-sign units, a repeating pattern every 5 rows). An encrypted
   message would leave none of that. A second tell: no payload weight prints as 0.000, while untouched init has a few.
7. One line: the overt "primes" beacon result is in Appendix E.

**Numbers:** max+numbered: Opus [cam.opus.max-num.unusual=5/5] flagged, [cam.opus.max-num.decoded=4/5] decoded; Fable
[cam.fable.max-num.unusual=5/5], [cam.fable.max-num.decoded=0/5]. Unedited hosts at the same setting: Opus
[clean.opus.max-num.unusual=0/5], Fable [clean.fable.max-num.unusual=0/5], p = [cam.opus.maxnum_vs_clean_p=0.0079]
per model. Fable max-num vs high-num p = [cam.fable.maxnum_vs_highnum_p=0.0079]; vs max-bare
p = [cam.fable.maxnum_vs_maxbare_p=0.0476]. Decode Opus vs Fable p = [cam.decode.opus_vs_fable_p=0.0476]. Cost per read:
Opus [cost.opus.high-bare.usd_per_read=0.16] → [cost.opus.max-num.usd_per_read=1.71]; Fable
[cost.fable.high-bare.usd_per_read=0.94] → [cost.fable.max-num.usd_per_read=2.99]. Init zero-prints:
[payload.init_zero_prints=13-20] of 1072 (payload: 0).

**Careful with:**
- "Unusual: yes" is mechanical; "found the message" and "called it a backdoor" are your hand judgments: label them.
- Opus was never run at high-numbered or max-bare: say "untested", don't imply it failed.
- Don't rank the models. Which one does better depended on the setting (the pilot showed the reverse).
- p = 0.0476 with n = 5 is suggestive, not strong; say so.

## 5. Discussion and Limitations (~150 + ~60 + ~30 words)

**Job:** what an auditor should do differently, and what this doesn't show.

**Beats (Discussion):**
1. A weight-based verdict about training data is conditional on the reader's model of the training recipe, and the
   audited party picks the recipe.
   Threat model: someone hiding what a model trained on (copyrighted, banned, or capability-conferring data) need not risk
   a tampering finding; a recipe the auditor misreads may be enough, and each choice has an honest-sounding rationale.
   Caveat: the attacker must know the auditor's prior to pick the knob; here the readers state it in their reasoning.
   The other direction: an honest developer who just uses Adam gets misattributed. Recipe-blind priors cause false
   accusations as well as evasions.
2. Not an LLM quirk (Rule result) → fix is calibration across recipes, which is cheap (both-optimizer training fixed the
   baselines).
3. Detection ≠ explanation: a confident wrong story (Fable's backdoor) is worse for an auditor than "something is off".
4. What the auditor finds depends on its settings and spend; audit protocols should state them.

**Limitations beats:** tiny nets, two datasets, two readers, n = 5 per cell, one message (plaintext, one string),
the message was planted by a script not a model; the audit question was not run on the
laundering-knob nets (so "passes a tampering check" is untested); thinking not visible; Appendix A has the full list and dual-use.

**Future Work beats:** more readers (non-Claude), encrypted payloads, bigger nets, testing a recipe-calibrated auditor.
Also: run the audit question on knob-laundered nets; an attacker who doesn't know the auditor's prior.

**Careful with:** keep implications proportional: "for weight readers like these, on nets like these".

## 6. Conclusion (~60 words)

**Job:** the one sentence a reader repeats to a colleague, plus the auditor takeaway. Your abstract's last sentence is
already most of it; don't introduce anything new.

## Code and Data, LLM Usage Statement

- Code and Data: repo link; `make check` regenerates every number; nets, prompts and raw responses are committed.
- LLM Usage Statement (your words): what Claude did (experiment code, scoring, red-team reviews, number checks, the build),
  what you did (question, design choices, predictions, writing), and that every number is verified by `make check`.

## Appendix A: Limitations and Dual-Use (required)

**Beats:** the limitations list above in full; dual-use: the laundering recipe and the hiding method could help an evader,
but the knobs are already standard practice, the fix (recipe-diverse calibration) is cheap, and defenders benefit more
from knowing; the message is harmless text; no real models or data were tampered with.
