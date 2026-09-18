# Build verl's PPO from scratch

Do not open `../common.py` first. Work from the docstrings in `ppo.py` and the
grader's messages; reading the reference turns this into transcription.

Names, argument orders and return tuples match
[`verl/trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py)
and `verl/utils/torch_functional.py`, so what you write here drops into a real
verl trainer unchanged.

| Stage | Functions | Question it answers |
|---|---|---|
| 1 | `masked_mean`, `masked_var`, `masked_whiten` | Why must padding never reach a statistic? |
| 2 | `compute_gae_advantage_return` | Why is `response_mask` not a `dones` flag? |
| 3 | `agg_loss` | Why does the aggregation mode change what the model learns? |
| 4 | `compute_policy_loss` | Why does PPO need a *second* clip? |
| 5 | `clip_by_value`, `compute_value_loss` | Why clip the critic around its own last prediction? |
| 6 | `entropy_from_logits`, `compute_entropy_loss` | Why is entropy subtracted from a minimized loss? |
| 7 | `kl_penalty`, `compute_rewards` | Why four KL estimators, and why does PPO put its KL in the reward? |

Run `python PPO/from_scratch/check.py` from the repository root. The grader
stops at the first incomplete stage. `GRPO/from_scratch/grpo.py` imports your
`agg_loss`, `compute_policy_loss` and `kl_penalty`, so finish this one first.

## Two conventions, and most mistakes are really about one of them

**Every tensor is `(batch, response_length)`.** No sequence dimension, no
per-sequence scalar. A single outcome reward for a whole response is a row that
is zero everywhere except its last valid position; a single advantage is a row
repeated across the response.

**`response_mask` is the only thing that makes a position real.** It is the EOS
mask. Every mean, sum, variance and loss is taken over it.

## References

Clipping follows equations (7)-(9) of
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347);
stage 2 follows equations (11)-(12) of
[High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438).
The dual clip in stage 4 is from
[Mastering Complex Control in MOBA Games](https://arxiv.org/pdf/1912.09729).
Stage 3's `seq-mean-token-sum-norm` is
[Dr.GRPO](https://arxiv.org/abs/2503.20783)'s constant divisor, and stage 7's
`k3` is from [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).

## At the end, you should be able to answer

- In GAE, why does a masked-out position *carry* `nextvalues` and `lastgaelam`
  through rather than resetting them? What breaks if you reset them?
- Why is `returns` computed before the advantage is whitened?
- The policy clip behaves differently depending on the sign of the advantage.
  Why, and what does the dual clip add that the ordinary clip cannot?
- Under `token-mean`, does a 500-token response influence the update more than a
  50-token one? Should it? Which mode would you pick to make length irrelevant?
- Why must a single-sample KL estimator be non-negative? If `k1` comes out
  negative on one token, what has actually gone wrong — and has anything?
- PPO subtracts its KL from the reward; GRPO adds its KL to the loss. Name one
  concrete consequence of that difference.
