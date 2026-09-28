# Build the candidates for PPO's slot, from scratch

Do not open `surrogates.py` (the reference) before you finish the exercise.
Work from the docstrings in `from_scratch/surrogates.py` and the grader's
messages.

The PPO paper does not present PPO as one mechanism. It presents **one training
loop** (§5, Algorithm 1) with **a slot for the loss**, and several candidates for
that slot (§3, §4), compared against each other in §6.1. The clip wins, and only
that choice gets the name "PPO". This module builds the candidates and runs
that comparison on the VPG toy, *before* `PPO/` introduces the algorithm at
LLM scale.

```
for iteration:
    batch = collect_rollouts(policy)                  theta_old, frozen from here on
    for epoch in range(K):
        loss = SLOT(policy, batch, old_probs)         <- the only line the candidates change
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        AFTER_STEP                                    TRPO's rollback: surgery on the loop
    AFTER_ITERATION                                   adaptive KL re-prices beta here
```

The loop is given (`loop.py`). The slots are yours. Finish `VPG/` and `TRPO/`
first: the toy is VPG's, `mean_kl` is TRPO's, and stage 1 is TRPO's surrogate
turned into a loss.

| Stage | Function | Paper | Question it answers |
|---|---|---|---|
| 1 | `ratio_of`, `cpi_loss` | eq. 6 | Why can the ratio alone not tell you when to stop? |
| 2 | `kl_penalty_loss` | eq. 5 | Why does the same β brake differently when rewards change size? |
| 3 | `adapt_beta` | eq. 8 | Why can a rule that runs *between* iterations not stop a bad one? |
| 4 | `clip_loss` | eq. 7 | Why `min`, and why does the clip only ever remove credit? |
| 5 | none | §6.1 | What does each candidate do on the batch that broke VPG? |

## How to start

Each stage is one to four TODO lines in `from_scratch/surrogates.py`, with a
worked example and the numbers you should get.

1. Fill the lines marked `TODO stage 1`.
2. Try them: `python Surrogates/from_scratch/surrogates.py` prints your result
   for every stage next to the expected one.
3. Check them: `./scripts/run_surrogates.sh check` grades the stages in order
   and stops at the first one that is not right yet, with a hint.
4. Repeat for stages 2, 3 and 4. When all pass, `./scripts/run_surrogates.sh diff`
   proves your walkthrough output matches the reference line for line.

Stage 5 needs no new code. It runs your slots through the loop, including a
full adaptive-β run.

## What the comparison shows

`./scripts/run_surrogates.sh run` (about three minutes) is the toy's version of
the paper's Table 1: every slot in the same loop, 20 seeds, 800 rollouts, SGD
at lr 0.3, 50 epochs per iteration.

| Slot | J@800 | Stuck | KL/iter | Hooks |
|---|---|---|---|---|
| L^PG, 1 epoch (VPG) | 0.806 | 0/20 | 0.0007 | none |
| L^PG, 50 epochs | 0.797 | 6/20 | 0.0554 | none |
| L^CPI | 0.779 | 10/20 | 0.1286 | none |
| TRPO, δ 0.01 | 0.874 | 0/20 | 0.0071 | AFTER_STEP |
| KL penalty, β 0.3 | 0.787 | 8/20 | 0.1126 | none |
| KL penalty, β 3 | 0.836 | 0/20 | 0.0179 | none |
| KL penalty, β 10 | 0.841 | 0/20 | 0.0016 | none |
| KL adaptive, d_targ 0.01 | 0.771 | 7/20 | 0.0741 | AFTER_ITERATION |
| **L^CLIP, ε 0.2** | **0.878** | **0/20** | 0.0117 | none |

- **No brake** (L^CPI) is worse than VPG reusing its batch: it remembers θ_old
  but nothing stops the ratio. The paper's L^CPI row scores −0.39, below a
  random policy.
- **A fixed β** is a price, not a distance: 0.3 barely brakes, 10 brakes so hard
  it starts slowly.
- **Adaptive β** acts only between iterations. It cannot stop the update in
  progress, and in quiet stretches it halves β toward 0, so the next misleading
  batch meets almost no brake. The paper also ranks it below the clip (0.74 vs 0.82).
- **The clip** needs nothing but the slot and ends highest with no run stuck.

The second table multiplies every reward by 10 under Adam (the paper's
optimizer, whose step size ignores the gradient's scale). TRPO and the clip do
not change at all, because δ and ε are distances. β = 30 goes from barely moving
(J 0.722) to J 0.876, exactly what β = 3 gives on the original rewards: a price
in reward per nat scales with the reward.

## Why SGD for the main table

Every module in this toy track uses SGD at lr 0.3, so the numbers line up with
`VPG/` and `TRPO/`. One side effect: plain SGD with a KL penalty goes unstable
past β ≈ 14 at that learning rate (the penalty's step overshoots), which is why
the fixed-β grid stops at 10. The paper's Adam does not have that limit, and
table 2 uses it.

## Then: PPO

PPO = this loop + L^CLIP in the slot, plus (paper §5, eq. 9) a value-function
loss to train the critic behind Â, an entropy bonus, and GAE advantages. `PPO/`
has all of that the way verl writes it, for `(batch, response_length)` tensors.

## Terms

The VPG and TRPO terms are in `VPG/README.md` and `TRPO/README.md`. This module adds:

- **slot**: the one line of the loop that computes the loss. Every candidate is a
  function `slot(policy, batch, old_probs) -> loss`.
- **hook**: code the loop runs besides the slot. `AFTER_STEP` runs after every
  optimizer step (TRPO's rollback); `AFTER_ITERATION` runs once per batch
  (adaptive β).
- **L^CPI**: eq. 6, mean(r × Â), the ratio surrogate with no limit.
- **β**: the KL penalty's weight, in reward per nat. A price.
- **d_targ**: adaptive KL's target KL per iteration, in nats. A distance.
- **ε**: the clip's range on the ratio, 1 ± ε. A distance.
- **KL/iter**: how far one iteration actually moved the policy, in nats.

## At the end, you should be able to answer

- In the walkthrough's step 1, every slot has the same gradient at θ_old. So
  where do they differ, and why is that exactly where reusing a batch matters?
- In stage 2, what β makes the 0.45 and 0.48 moves score equally? What happens
  to that β if every advantage doubles?
- In stage 3, the rule halves β whenever the policy barely moves. Why does that
  make adaptive KL vulnerable late in training, when the policy has converged?
- In stage 4, which of the four clip cases have zero gradient, and why must the
  "moved the wrong way" cases not?
- TRPO needs `AFTER_STEP`, adaptive KL needs `AFTER_ITERATION`, and the clip
  needs neither. Why does that matter for an LLM trainer like verl?
