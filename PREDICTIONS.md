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
