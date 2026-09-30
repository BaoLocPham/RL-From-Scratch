# Build PPO from scratch: the paper first, then verl

Do not open `common.py` (the reference) before you finish the exercise. Work
from the docstrings in `from_scratch/ppo.py` (Part 1) and
`from_scratch/ppo_verl.py` (Part 2), and the grader's messages.

`Surrogates/` ended with a definition: **PPO = the loop with L^CLIP in the
slot, plus a value loss, an entropy bonus and GAE.** This module builds exactly
that list, then the loop, at an LLM's shapes: the paper's §5, eq. 9

```
L^{CLIP+VF+S}(θ) = Ê_t[ L^CLIP_t − c1 · L^VF_t + c2 · S[π_θ](s_t) ]
```

in two parts, one file each. **Part 1, `from_scratch/ppo.py`, is PPO itself**
(the paper, stages 1–6) and ends with your code training a model. **Part 2,
`from_scratch/ppo_verl.py`, is what verl adds for production** (stages 7–12):
for later, but before GRPO. It imports your Part 1 functions and builds on them.

## The whole picture

Algorithm 1, with the stage that builds each line. The right-hand column is
what Part 2 adds to the same line; the loop itself never changes.

```
                                                  PART 1: ppo.py (stage)            PART 2: ppo_verl.py adds (stage)
for iteration:
    batch = rollout(model)                        given, task.py
        tokens, response_mask, old_log_prob,        uses your logprobs_from_logits (1)
        values, token_level_scores
    batch = compute_advantage(batch)              compute_advantage (6)             verl_compute_advantage (12)
        reward per token                            the scores as they are           - KL to a reference model (11)
        advantages, returns = GAE                   compute_gae (2)                  + whitening (7)
    for epoch in range(K):                        ppo_update (6)                    verl_ppo_update (12)
        for minibatch of M:
            log pi per token                        logprobs_from_logits (1)
            pg_loss = -L^CLIP                       ppo_clip_loss (3)                + asymmetric range, dual clip,
                                                                                       pg_clipfrac, ppo_kl (9)
            vf_loss = L^VF                          value_loss (4)                   + clipped around the old value (10)
            entropy = S                             entropy_bonus (5)
            every E_t over tokens                   masked_mean (1)                  + agg_loss, four modes (8)
            loss = pg_loss + c1*vf_loss - c2*entropy     (6)                           the same (12)
            zero_grad, backward, step
```

Where it sits: `VPG/` → `TRPO/` → `Surrogates/` (the toy track, one decision
per question) → `SimplePPO/` (the same PPO on three decisions, no tokens) →
`SimpleGRPO/` (the same agent without the critic) → `SimpleDPO/` (the same
agent without a reward) → **`PPO/`** (T tokens per response, masks) → `GRPO/`
(SimpleGRPO at verl's shapes) → `DPO/` (SimpleDPO at verl's shapes) → `Agent0/`. If this module feels like a big jump, do `SimplePPO/`
first: Part 1 here is that module in LLM clothing.

### The files

| File | What it is | Yours to edit? |
|---|---|---|
| `from_scratch/ppo.py` | Part 1, core PPO: stages 1–6 of TODOs, worked examples, and a playground (`python PPO/from_scratch/ppo.py`) | **yes** |
| `from_scratch/ppo_verl.py` | Part 2, verl's extras: stages 7–12, built on your Part 1, with its own playground | **yes**, later |
| `from_scratch/check.py` | the grader: stages in order, stops at the first gap, with a hint | no |
| `task.py` | the token task: the model, `rollout`, and `train`, the outer loop | no, read it |
| `steps_ppo.py` | the walkthrough: every equation with its numbers substituted | no |
| `run_ppo.py` | the demo: trains the token task and explains each column | no |
| `common.py` | the reference implementation | do not open until you finish |

### The commands

| Command | Does |
|---|---|
| `python PPO/from_scratch/ppo.py` | your value for every Part 1 stage next to the expected one |
| `python PPO/from_scratch/ppo_verl.py` | the same for Part 2 |
| `./scripts/run_ppo.sh check core` | grade Part 1 (stages 1–6) |
| `./scripts/run_ppo.sh steps core` | the Part 1 walkthrough, reference implementation |
| `./scripts/run_ppo.sh diff core` | grade Part 1, then prove your walkthrough matches the reference line for line |
| `RL_IMPL=scratch ./scripts/run_ppo.sh run` | train the token task with your core PPO |
| `./scripts/run_ppo.sh check` / `diff` / `run verl` | the same, including Part 2 |

## The stages

### Part 1, core PPO: the paper only (stages 1–6)

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `logprobs_from_logits`, `masked_mean` | — | How does one decision become T tokens, and why must padding never reach a mean? |
| 2 | `compute_gae` | eq. 11–12 | Which token earned a reward that arrives only at the end? Why is the mask not a `dones` flag? |
| 3 | `ppo_clip_loss` | eq. 7 | Why does Surrogates' `min` become a `max`? (see *Why the clip uses `max` here*, below) |
| 4 | `value_loss` | eq. 9's L^VF | What is the critic trained to predict? |
| 5 | `entropy_from_logits`, `entropy_bonus` | eq. 9's S | Why is a bonus subtracted from the loss? |
| 6 | `compute_advantage`, `ppo_update` | Algorithm 1, eq. 9 | Can your pieces train something? |

### Part 2, verl's extras (stages 7–12, `ppo_verl.py`), for later

Each verl function is your Part 1 version plus one idea, and the Part 1 lines
come filled in. The file imports your `masked_mean`, `compute_gae`,
`entropy_from_logits` and `logprobs_from_logits` from `ppo.py`.

| Stage | You build | Adds | Why production wants it |
|---|---|---|---|
| 7 | `masked_var`, `masked_whiten`, `compute_gae_advantage_return` | whitened advantages | the step size stops following the reward's scale |
| 8 | `agg_loss`, `compute_entropy_loss` | four ways to average over tokens | should a long response outweigh a short one? (Dr.GRPO) |
| 9 | `compute_policy_loss` | asymmetric range, dual clip, `pg_clipfrac`, `ppo_kl` | bounds bad tokens made much more likely; metrics to watch |
| 10 | `clip_by_value`, `compute_value_loss` | a clipped critic | one minibatch cannot yank the critic far |
| 11 | `kl_penalty`, `compute_rewards` | KL to a frozen reference, in the reward | RLHF: stay close to the model before RL (not the paper's KL) |
| 12 | nothing | the same loop, verl's functions | — |

Finish `VPG/`, `TRPO/` and `Surrogates/` first: stage 3 is your Surrogates
clip, per token. `GRPO/from_scratch/` imports your `agg_loss`,
`compute_policy_loss`, `kl_penalty` and `masked_mean` from `ppo_verl.py`, so it
needs Part 2.

## How to start

Most TODOs are one line, and every stage has a worked example with the numbers
you should get. Reading `./scripts/run_ppo.sh steps core` first shows where
each stage is heading.

1. Fill the lines marked `TODO stage 1` in `from_scratch/ppo.py`.
2. Try them: `python PPO/from_scratch/ppo.py` prints your result for every
   stage next to the expected one.
3. Check them: `./scripts/run_ppo.sh check core` grades Part 1 and stops at the
   first stage that is not right yet, with a hint about the likely mistake.
4. Repeat through stage 6. Then:
   - `./scripts/run_ppo.sh diff core` proves your Part 1 walkthrough matches
     the reference line for line;
   - `RL_IMPL=scratch ./scripts/run_ppo.sh run` trains the token task with your code.

For Part 2 later, open `from_scratch/ppo_verl.py`, try it with
`python PPO/from_scratch/ppo_verl.py`, and drop the `core`: `check`, `diff`, and
`run verl`.

## The token task (`task.py`, given)

The toy track made one decision per question. An LLM makes one per token, and
the reward arrives at the end. `task.py` is the smallest thing with that shape:
responses of up to 4 tokens from a vocabulary of 3, a hidden target
(2, 0, 1, 2), the reward (fraction correct) on the last real token only, and a
value head as the critic. The "model" is a table of logits per position, a
stand-in for a transformer, whose state at position t would be the whole prefix.

Both parts learn the target in about 20 iterations:

| Iteration | Core PPO reward | verl PPO reward |
|---|---|---|
| 0 | 0.341 | 0.341 |
| 5 | 0.701 | 0.747 |
| 10 | 0.904 | 0.938 |
| 20 | 0.982 | 0.990 |
| 39 | 0.982 | 1.000 |

verl's version is slightly faster here, mostly from whitening the advantages.
The algorithm is the same.

## From the toy track to here

| | Toy track (`Surrogates/`; `SimplePPO/` adds the critic and GAE) | Here |
|---|---|---|
| One rollout | one question, one action | a response of T tokens, `(batch, response_length)` |
| log π | `policy.dist(qtype).log_prob(action)` | `logprobs_from_logits(logits, tokens)` |
| Advantage | reward − batch mean, one number | GAE per token, with a learned critic |
| The mean Ê_t | `.mean()` | `masked_mean` over `response_mask` (Part 2: `agg_loss`, four modes) |
| The slot | `clip_loss` | `ppo_clip_loss` + `c1·value_loss` − `c2·entropy_bonus` (eq. 9) |
| An update | the whole batch per step | minibatches of M, K epochs |
| KL | to θ_old (TRPO, adaptive β) | also to a frozen reference model, in the reward |

## Why the clip uses `max` here but `min` in Surrogates

The paper's clipped objective (eq. 7) takes a **min**, and Surrogates'
`clip_loss` does too. `ppo_clip_loss` takes a **max**. Both are right: they put
the minus sign in different places.

The paper **maximises** its objective; PyTorch **minimises**, so both files
return the loss $-L^{CLIP}$. Negating a min flips it into a max:

```
-min(x, y) = max(-x, -y)
```

```python
# Surrogates' clip_loss: objective terms, min, THEN negate
unclipped = ratio * A                                   #  r·A
clipped   = ratio.clamp(1 - eps, 1 + eps) * A           #  clip(r)·A
loss = -torch.min(unclipped, clipped).mean()

# PPO's ppo_clip_loss: negate FIRST, so the terms are loss terms, then max
unclipped = -A * ratio                                  # -r·A
clipped   = -A * torch.clamp(ratio, 1 - eps, 1 + eps)   # -clip(r)·A
loss = masked_mean(torch.maximum(unclipped, clipped), mask)
```

The same numbers, for a good token with A = +3 whose ratio reached 1.5 (ε = 0.2):

| | Surrogates: `min`, then negate | PPO: negate, then `max` |
|---|---|---|
| unclipped | 4.5 | −4.5 |
| clipped | 3.6 | −3.6 |
| combine | min = 3.6, negated → **−3.6** | max = **−3.6** |

Identical loss, identical gradient. The trap is that **both files name the
variables `unclipped` and `clipped`, but they hold opposite-signed values.** The
rule:

- terms **without** the minus sign (objective values) → the pessimistic choice is **`min`**, then negate;
- terms **with** the minus sign inside (loss values) → the pessimistic choice is **`max`**.

"Pessimistic" means the smaller objective, which is the larger loss. PPO uses
the loss form because verl does. Part 2's `compute_policy_loss` works directly
on the loss terms: `pg_clipfrac` counts where the clipped loss is the larger
one, and the dual clip adds a third loss term to compare against.

The same sign flip explains every term of eq. 9 in code:

| | Objective (paper, maximised) | Loss (code, minimised) |
|---|---|---|
| pessimistic choice | `min` | `max` |
| policy term | + L^CLIP | `pg_loss` = −L^CLIP |
| value term | − c1·L^VF | `+ c1 * vf_loss` |
| entropy term | + c2·S | `- c2 * entropy` |

## Two conventions, and most mistakes are really about one of them

**Every tensor is `(batch, response_length)`.** No per-sequence scalar. A
single outcome reward for a whole response is a row that is zero except at its
last real token; a single advantage is a row repeated across the response.

**`response_mask` is the only thing that makes a position real.** Every mean
and loss is taken over it. In GAE it is *not* a `dones` flag: a
masked position in the middle of a response (a tool's output) carries the
running values through, so the tokens before it still get credit.

## References

Clipping and eq. 9 are from
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347);
GAE (stage 2) is eq. 11–12 there, from
[High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438).
The dual clip in stage 9 is from
[Mastering Complex Control in MOBA Games](https://arxiv.org/pdf/1912.09729).
Stage 8's `seq-mean-token-sum-norm` is [Dr.GRPO](https://arxiv.org/abs/2503.20783)'s
constant divisor. Stage 11's reference KL is RLHF's
([InstructGPT](https://arxiv.org/abs/2203.02155)), and `k3` is from
[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).

## At the end, you should be able to answer

Part 1:

- In stage 2, why does a masked position in the MIDDLE of a response carry
  `nextvalues` and `lastgaelam` through rather than resetting them? What would
  the first token's return be if it reset?
- In stage 3, Surrogates' clip took a `min`; `ppo_clip_loss` takes a `max`. Show
  they are the same thing.
- In stage 6, the first optimizer step of every iteration has every ratio at 1.
  What is `pg_loss` then, and why does the policy still move?
- The critic's target is the return, the actor's signal is the advantage. Which
  one would you expect to be large when the policy is already good?

Part 2:

- Why is `returns` computed before the advantage is whitened?
- What does the dual clip bound that the ordinary clip cannot?
- Under `token-mean`, does a 500-token response influence the update more than a
  50-token one? Which mode would make length irrelevant?
- The paper's KL (eq. 8) and stage 11's KL both measure distance between two
  policies. Which two, in each case, and why does RLHF need the second one?
