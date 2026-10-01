# Build G-Zero from scratch, on two tables of logits

Do not open `gzero.py` (the reference) before you finish the exercise. Work
from the docstrings in `from_scratch/gzero.py` and the grader's messages.

[G-Zero](https://arxiv.org/abs/2605.09959) improves a model with **no data and
no verifier**. `Agent0/` still checked answers against a majority vote; G-Zero
checks nothing. Its only signal is the Generator's own probabilities, with and
without a hint.

| Agent | Paper's name | Job | Trained with |
|---|---|---|---|
| **Proposer** | π_P | picks a query q and a hint h | GRPO (`Agent0/`, `SimpleGRPO/`) |
| **Generator** | π_G | answers q, with or without h | DPO (`SimpleDPO/`), length-normalised |

This module builds the paper's equations on a toy where each agent is a small
table of logits. Every logit you see move is one the equations moved.

## The loop

The paper runs 2 rounds of two phases:

```
Phase 1  train the Proposer pi_P (GRPO, K = 16 per group); the Generator is frozen
         pi_P picks (q, h); pi_G answers q WITHOUT the hint (a_hard);
         reward r = Hint-delta - P_length - P_BLEU                                       Eq. 3-5
Phase 2  train the Generator pi_G (DPO); the Proposer is frozen
         pi_P picks N (q, h); pi_G answers with the hint (a_assisted) and without (a_hard);
         keep the lower 50% of delta; DPO with a_assisted chosen, a_hard rejected,
         and the prompt q ALONE                                                           Eq. 6
```

The prompt in Phase 2 has no hint. DPO teaches the Generator to answer
**without** the hint the way it would answer **with** it: it internalises the
hint.

## The toy (`gzero_env.py`, given code)

- **Queries:** 6 query ids. Each has a good answer of 3 tokens (vocab 4). The
  good answers are only for **measuring** progress; training never sees them.
- **The Generator is a table of logits** `[query, position, token]`. It starts
  with **blind spots**: positions where it prefers a wrong token.

  | Query | 0 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|---|
  | Blind spots | none | 1 | 2 | 0, 2 | 1, 2 | 0, 1, 2 |

  A known position starts at p(good) = 0.87, a blind spot at 0.13.
- **A hint** names the good token at one position, or at all three. Reading it
  adds κ = 3.0 to that token's logit:

  ```
  logits(q, h) = logits(q) + κ · onehot(position, good token)
  ```

  On query 1's blind spot, that lifts p(good) from 0.13 to 0.76. This
  in-context ability is fixed; DPO trains only the unassisted table.
- **The Proposer is 24 logits,** one per (query, hint) pair, starting uniform.
  A one-position hint is 120 characters long; the all-positions hint is 360.

## The equations

| | Equation | What it does |
|---|---|---|
| Eq. 3 | δ(q, h, a_hard) = (1/T) Σₜ [log π_G(aₜ \| q) − log π_G(aₜ \| q, h)] | how much the hint makes the Generator's **own** answer less likely |
| Eq. 4 | P_length = λ·max(0, (\|h\| − 200)/100), λ = 0.03 | charges hints over 200 characters |
| — | P_BLEU = \|C_i\|/\|B\| | the share of the batch that picked the same (q, h) |
| Eq. 5 | r(q, h) = δ − P_length − P_BLEU | the Proposer's reward; no max(0, ·), so it can be negative |
| — | chosen y_w = a_assisted ~ π_G(·\|q, h), rejected y_l = a_hard ~ π_G(·\|q) | the Generator's DPO pairs; keep the lower 50% of δ |
| Eq. 6 | −log σ(β·(r̄(y_w) − r̄(y_l))), r̄(y) = (1/\|y\|)·log(π_θ(y\|q) / π_ref(y\|q)), β = 2 | length-normalised DPO; π_ref is the Generator at the round's start |

**What δ measures.** Take query 1, whose position 1 is a blind spot:
- **A hint on that blind spot** gives δ = +0.42 on one of the Generator's
  answers: its own wrong token became less likely.
- **A hint on position 0, which it already gets right,** gives δ = −0.04: its
  own right token became *more* likely.

Averaged over the Generator's answers, δ is the KL divergence from π_G(·|q)
to π_G(·|q, h), which is never negative. It is largest where a hint changes
the answer most:

| Query, hint | Average δ |
|---|---|
| 5, all positions (3 blind spots) | 0.866 |
| 1, position 1 (its blind spot) | 0.289 |
| 0, all positions (no blind spot) | 0.258 |
| 1, position 0 (a known position) | 0.086 |

So the Proposer learns to aim its hints at blind spots, without ever being told
where they are.

**What the paper leaves open.** We chose κ = 3, the toy's sizes, and the
round's scale (200 pairs, 30 GRPO steps of 4 groups × 16, SGD learning rates
1.0 and 2.0). The paper's own 50 DPO steps of batch 8 are kept. The paper
clusters by BLEU similarity; an output here is only an id, so a cluster is the
copies of one (q, h).

## The stages

| Stage | You build | Equation |
|---|---|---|
| 1 | `hint_delta` | Eq. 3 |
| 2 | `length_penalty`, `repetition_penalty`, `proposer_reward` | Eq. 4–5 |
| 3 | `make_pair`, `lower_half` | the DPO data |
| 4 | `dpo_loss_ln`, `group_advantage`, `clipped_loss` | Eq. 6, GRPO |
| 5 | no code: the loop runs on your pieces | — |

Each TODO is one term of one equation, written as a comment on the line you
fill. The loops are given: `train_proposer` (Phase 1), `collect_pairs` and
`train_generator` (Phase 2), and `gzero` (both, for 2 rounds).

## How to start

1. Read `./scripts/run_gzero.sh steps`. It follows every equation with the
   numbers substituted in, and prints the logits as they move.
2. Fill the lines marked `TODO stage 1` in `from_scratch/gzero.py`.
3. Try them: `python GZero/from_scratch/gzero.py` prints your value next to
   the expected one.
4. Check them: `./scripts/run_gzero.sh check` grades the stages in order, with a
   hint for the likely mistake.
5. When all pass:
   - `./scripts/run_gzero.sh diff` proves your walkthrough matches;
   - `RL_IMPL=scratch ./scripts/run_gzero.sh run` runs the demo with your code.

## What the loop does (`./scripts/run_gzero.sh run`, about 20 seconds)

One run, seed 0. Each round is 30 GRPO steps on π_P, then 200 pairs (the lower
half kept), then 50 DPO steps on π_G:

| Round | π_P on query 0..5 | All-positions hint | π_G unassisted p(good), per query | Overall |
|---|---|---|---|---|
| start | uniform | — | 0.87 0.62 0.62 0.38 0.38 0.13 | 0.502 |
| 1 | 0.04 0.05 0.06 0.22 0.21 **0.42** | 0.77 | 0.86 0.67 0.70 0.62 0.49 0.61 | 0.657 |
| 2 | 0.03 0.04 0.04 0.21 **0.34 0.34** | 0.84 | 0.82 0.63 0.74 0.56 0.78 0.87 | 0.732 |

- **Round 1** aims at query 5, the one with the most blind spots, and DPO
  lifts its p(good) from 0.13 to 0.61.
- **Round 2:** the Proposer moves weight from query 5 to query 4, whose blind
  spots are now the bigger ones.
- **No answer was ever checked.**

Then 10 seeds, changing one ingredient at a time:

| Run | p(good) after 1, 2 | π_P on query 5 | Cells worse than at start | Top output |
|---|---|---|---|---|
| **G-Zero** | 0.651 0.723 | 0.53 → 0.38 | 1.1 of 18 | 0.38 |
| no lower-50% filter | 0.684 0.836 | 0.53 → 0.24 | 0.1 of 18 | 0.36 |
| no P_BLEU | 0.619 0.635 | 0.94 → 0.95 | 0.0 of 18 | 0.95 |

"Cells worse" counts the (query, position) cells that ended more than 0.1 below
where they started. "Top output" is π_P's largest probability on any one of its
24 outputs; uniform is 0.04.

An honest reading:

- **The Generator improves with no verifier.** The Proposer first aims at the
  query with the most blind spots, then moves off it as DPO fixes it.
- **The long hint mostly wins** (0.77, then 0.84). It moves the Generator most,
  and at the paper's λ = 0.03, P_length charges it only 0.048.
- **The lower-50% filter costs a lot here** (0.723 vs 0.836, and more cells
  slip back). The pairs with the highest δ are the ones where a_hard was
  furthest from what the hint pointed to, so on this toy they teach the most.
  The paper drops them for an LLM's sake: a very high-δ pair lies far off the
  Generator's distribution and breaks DPO's implicit KL budget. A table of
  logits has no such budget to protect.
- **P_BLEU matters here** (0.723 vs 0.635). Each query's blind spots are
  separate. Without the penalty, the Proposer piles 0.95 of its probability onto
  one (query, hint), and only that query improves.

## Terms

- **π_P, π_G:** the Proposer and the Generator.
- **a_hard, a_assisted:** the Generator's answer without the hint, and with it.
- **δ (Hint-delta):** the per-token drop in a_hard's log-probability once the
  hint is added (Eq. 3).
- **blind spot:** a (query, position) where the Generator prefers a wrong token.
- **internalise:** DPO with the hint-assisted answer as chosen and the prompt
  without the hint, so the hint's effect moves into the unassisted logits.

## At the end, you should be able to answer

- δ scores the Generator's **own** unassisted answer, with and without the
  hint. Why does that measure a blind spot without anyone knowing the right
  answer?
- On one answer δ can be negative; on average it is a KL divergence. Which one,
  and why is it never negative?
- The DPO prompt is q alone, but the chosen answer was written with the hint.
  What would change if the prompt included the hint?
- In this toy the hints are always right: the Proposer knows the good token.
  δ cannot tell a right hint from a wrong one. What would a misleading hint do
  to δ, and to the Generator?
- The lower-50% filter costs a lot here and the paper keeps it. What does an
  LLM have that this table does not?
