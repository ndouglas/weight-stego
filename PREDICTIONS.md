# Predictions (written before anything runs, in my own words)

Date: Sat Nov 7, 2026, before the seeds sweep and before the planted-pattern run.

## Backbone: Adam vs. SGD over seeds

1. I think that out of 5 seeds, each model will identify 5 Adam-MNIST nets as Fashion-MNIST and identify 0 SGD-MNIST nets as Fashion-MNIST.
2. I think that Fashion nets will stay correct under both optimizers.
3. I think that misreads will keep citing the "dead border means init" rule.

## The rung: a pattern planted in the never-active weights of an Adam-MNIST net

4. ~~I think that it will identify the Adam-MNIST nets as MNIST instead of Fashion.~~ I think it will never flag marked nets.
   - Revised Sat Nov 7, after the seeds sweep and before any planted-pattern run: the first sentence was a slip. I think it will continue to identify them as Fashion-MNIST.

## Earlier ladder (parked after the Friday pivot)

Date:

### Step 1: Hide

- Can a model tell what a tiny network does from its raw weights? Yes.
- Can a sender hide a number in the weights, accuracy within 1%? Yes.
- Can a copy read it back with the scheme agreed? Without one? Yes. No.
- Clean-network false alarms: how often will a receiver "find" a message that isn't there? ~10%

### Beacon

- Will a receiver doing an ordinary task (describe this network) mention primes or Fibonacci planted in the weights, unprompted? No.

### Step 2: Detect

-

### Step 3: Infect

-

## Round 2 (Sat Nov 7, ~7am, before any round-2 reads)

1. E1, optimizer hint: I think that 5 of the Adam-MNIST nets will come back as MNIST. And I think the Fashion nets will stay Fashion.
2. E2, sign-bit camouflage: I think 3 of the 10 camouflage audits will flag something, but I don't think they will figure out the root cause or the message.
3. E3, canary instruction: I think 0 will decode the text, 7 will report the net as clean, and 0 will end with "pineapple".

## Round 3 (Sat Nov 7, ~9am, before any round-3 design or reads)

1. Density threshold: I would guess that Fable stops seeing the message at about 80% density. Or perhaps 95% density if the first few bytes are obvious garbage.
2. Plausible-excuse injection: Hmm, maybe. I think it'll still get flagged, but as something puzzling rather than malicious.
3. D1, MNIST + SGD + random shift ±2: I think 7 of the 10 shift-augmented MNIST reads come back "Fashion".
4. D2, Fashion + Adam + MNIST-style 20x20 pad: I think 3 of the 10 pad-20 Fashion reads come back "MNIST", and the other 7 come back as Fashion-MNIST or something else.

## Effort × line numbers (Fable, camouflage nets, API) — 2026-11-07, before results

Batch msgbatch_012nYHFrqVLHqMQpfoGxL4bC submitted; predictions written before any result was read.
Cells: high/numbered, max/bare, max/numbered (5 reads each). Baseline high/bare: 2/5 unusual, 0/5 decoded.

Nathan, verbatim:
1. Hmm, gonna guess 2.
2. Gonna guess 4.
3. Gonna guess 5.
4. Guessing the thinking, but we'll see.

(Single numbers per cell; scored as decodes out of 5, the pilots' headline. "Unusual" counts reported alongside.)

## Opus at max effort, line-numbered (camouflage nets, API) — 2026-11-07, before results

Batch msgbatch_01GbZeESdjCq5k5snQ5fypTb submitted; prediction written before any result was read.

Nathan, verbatim: "I think it'll flag 5 and decode 3."

## Graft control and non-LLM baseline — 2026-11-07 evening, before results

Graft nets built (runs/graft, not yet read); zoo trained; baseline.py not yet run.

Nathan, verbatim:
1. I think that 10 G-adam reads will say MNIST and 10 G-sgd reads will say Fashion.
2. Mmmm, interesting setup. a) I'd guess 63% for rownorm, 32% for rule. But no idea, really.
   b) I think the rule reader will get fooled by both, but not the rownorm reader.

(1 is read as: per model, of 10 reads each — 5 nets x 2 repeats. 2a is P1, train SGD -> test Adam.)

## False-alarm control at max effort + line numbers — 2026-11-08, before results

The 5 unedited Adam-MNIST hosts behind the camouflage nets, audit question, max effort, line-numbered rows, both models
(Batch msgbatch_01Pard6yPddGjjmqHrW1UQhz submitted; prediction written before any result was read.)
Question: of 5 clean nets, how many will each model call unusual?

Nathan, verbatim: "I think it'll be 0, 0."

(Read as: Opus 0/5, Fable 0/5.)
