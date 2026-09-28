# Build PPO from scratch, the way verl writes it

Do not open `common.py` (the reference) before you finish the exercise. Work
from the docstrings in `from_scratch/ppo.py` and the grader's messages.

`Surrogates/` ended with a definition: **PPO = the loop with L^CLIP in the
slot, plus a value loss, an entropy bonus and GAE.** This exercise builds that
list, then the loop, at verl's shapes. It follows the paper's §5, eq. 9:

```
L^{CLIP+VF+S}(θ) = Ê_t[ L^CLIP_t − c1 · L^VF_t + c2 · S[π_θ](s_t) ]
```

and Algorithm 1 (the Notion page's §6.1):

```
for iteration:
    batch = rollout(model)                        θ_old: N responses of T tokens
    batch = compute_advantage(batch)              KL into the reward, then GAE -- once
    for epoch in range(K):
        for minibatch of M:
            loss = pg_loss + c1 · vf_loss − c2 · entropy      eq. 9, negated
            zero_grad, backward, step
```

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `logprobs_from_logits`, `masked_mean`, `masked_var`, `masked_whiten` | — | How does one decision become T tokens, and why must padding never reach a statistic? |
| 2 | `compute_gae_advantage_return` | eq. 11–12 | Which token earned a reward that arrives only at the end? Why is the mask not a `dones` flag? |
| 3 | `agg_loss` | eq. 7's Ê_t | Should a long response count for more than a short one? |
| 4 | `compute_policy_loss` | eq. 7 | Why does Surrogates' `min` become a `max`, and what does the dual clip add? |
| 5 | `clip_by_value`, `compute_value_loss` | eq. 9's L^VF | Why clip the critic around its own old prediction? |
| 6 | `entropy_from_logits`, `compute_entropy_loss` | eq. 9's S | Why is a bonus subtracted from the loss? |
| 7 | `kl_penalty`, `compute_rewards` | — (RLHF) | How is this KL different from the paper's, and why is it in the reward? |
| 8 | `compute_advantage`, `ppo_update` | Algorithm 1, eq. 9 | Can your pieces train something? |

Finish `VPG/`, `TRPO/` and `Surrogates/` first. Stage 4 is your Surrogates
clip, per token. `GRPO/from_scratch/` imports your `agg_loss`,
`compute_policy_loss`, `kl_penalty` and `masked_mean`, so do this before GRPO.

## How to start

Most TODOs are one line, and every stage has a worked example with the numbers
you should get.

1. Fill the lines marked `TODO stage 1` in `from_scratch/ppo.py`.
2. Try them: `python PPO/from_scratch/ppo.py` prints your result for every
   stage next to the expected one.
3. Check them: `./scripts/run_ppo.sh check` grades the stages in order and stops
   at the first one that is not right yet, with a hint about the likely mistake.
4. Repeat through stage 8. Then `./scripts/run_ppo.sh diff` proves your
   walkthrough matches the reference line for line, and
   `RL_IMPL=scratch ./scripts/run_ppo.sh run` trains the token task with your code.

## The token task (`task.py`, given)

The toy track made one decision per question. An LLM makes one per token, and
the reward arrives at the end. `task.py` is the smallest thing with that shape:
responses of up to 4 tokens from a vocabulary of 3, a hidden target
(2, 0, 1, 2), the reward (fraction correct) on the last real token only, and a
value head as the critic. The "model" is a table of logits per position, a
stand-in for a transformer, whose state at position t would be the whole prefix.

The reference learns the target in about 20 iterations:

| Iteration | Reward | vf_loss | Entropy | clipfrac |
|---|---|---|---|---|
| 0 | 0.341 | 0.0586 | 1.0903 | 0.119 |
| 5 | 0.747 | 0.0185 | 0.5774 | 0.049 |
| 10 | 0.938 | 0.0048 | 0.2509 | 0.007 |
| 20 | 0.990 | 0.0014 | 0.0850 | 0.002 |
| 39 | 1.000 | 0.0000 | 0.0343 | 0.000 |

## From the toy track to here

| | Toy track (`Surrogates/`) | Here |
|---|---|---|
| One rollout | one question, one action | a response of T tokens, `(batch, response_length)` |
| log π | `policy.dist(qtype).log_prob(action)` | `logprobs_from_logits(logits, tokens)` |
| Advantage | reward − batch mean, one number | GAE per token, with a learned critic |
| The mean Ê_t | `.mean()` | `agg_loss`, four modes, over `response_mask` |
| The slot | `clip_loss` | `compute_policy_loss` + `c1·compute_value_loss` − `c2·compute_entropy_loss` |
| An update | the whole batch per step | minibatches of M, K epochs |
| KL | to θ_old (TRPO, adaptive β) | also to a frozen reference model, in the reward |

## Two conventions, and most mistakes are really about one of them

**Every tensor is `(batch, response_length)`.** No per-sequence scalar. A
single outcome reward for a whole response is a row that is zero except at its
last real token; a single advantage is a row repeated across the response.

**`response_mask` is the only thing that makes a position real.** Every mean,
sum, variance and loss is taken over it. In GAE it is *not* a `dones` flag: a
masked position in the middle of a response (a tool's output) carries the
running values through, so the tokens before it still get credit.

## References

Clipping and eq. 9 are from
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347);
GAE (stage 2) is eq. 11–12 there, from
[High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438).
The dual clip in stage 4 is from
[Mastering Complex Control in MOBA Games](https://arxiv.org/pdf/1912.09729).
Stage 3's `seq-mean-token-sum-norm` is [Dr.GRPO](https://arxiv.org/abs/2503.20783)'s
constant divisor. Stage 7's reference KL is RLHF's
([InstructGPT](https://arxiv.org/abs/2203.02155)), and `k3` is from
[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).

## At the end, you should be able to answer

- In stage 2, why does a masked position in the MIDDLE of a response carry
  `nextvalues` and `lastgaelam` through rather than resetting them? What would
  the first token's return be if it reset?
- Why is `returns` computed before the advantage is whitened?
- In stage 4, Surrogates' clip took a `min`; verl's takes a `max`. Show they are
  the same thing. What does the dual clip bound that the ordinary clip cannot?
- Under `token-mean`, does a 500-token response influence the update more than a
  50-token one? Which mode would make length irrelevant?
- The paper's KL (eq. 8) and stage 7's KL both measure distance between two
  policies. Which two, in each case, and why does RLHF need the second one?
- In stage 8, the first optimizer step of every iteration has `pg_loss` 0 and
  `clipfrac` 0. Why, and why does the policy still move?
