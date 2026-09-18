# Build verl's GRPO from scratch

Do not open `../common.py` first. Work from the docstring in `grpo.py` and the
grader's messages.

**Finish `PPO/from_scratch/` first.** `grpo.py` imports your `agg_loss`,
`compute_policy_loss` and `kl_penalty` from there, and this grader checks them.
That is not an artificial dependency: in verl all of it lives in one
`core_algos.py`, and a GRPO trainer calls exactly those functions.

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `compute_grpo_outcome_advantage` | How can the sampled group replace a value function? |
| 2 | the re-exports | What does GRPO actually change about PPO? |

Run `python GRPO/from_scratch/check.py` from the repository root.

## The short version

PPO trains a critic to predict expected reward and subtracts it. GRPO samples
several responses to the **same prompt** and subtracts their own mean. That is
the entire idea, and it deletes `compute_gae_advantage_return`,
`compute_value_loss`, the value head and its optimizer — most of PPO's cost and
most of its tuning surface.

The exercise is one function because that is honestly how much of verl is
GRPO-specific.

## Three details that are easy to get wrong

1. **A singleton group takes `mean=0, std=1`,** not its own mean. Its own mean
   would make the advantage identically zero and discard the sample.
2. **`torch.std` is the sample (n-1) std.** Write-ups of GRPO frequently use the
   population std and report visibly different numbers on small groups. Match
   the code, not the write-up.
3. **Return the same tensor twice.** Under outcome supervision there is no critic
   to regress, so `returns` is meaningless — verl returns the pair anyway so
   every estimator shares one signature.

## One flag, one paper

`norm_adv_by_std_in_grpo=False` subtracts the group mean without dividing by the
group std. That is [Dr.GRPO](https://arxiv.org/abs/2503.20783) in full. Dividing
by a per-group std ties the update size to how much that group happened to
disagree, which correlates with response length and shows up as a length bias.

GRPO itself is equation (3) of
[DeepSeekMath](https://arxiv.org/abs/2402.03300); the implementation follows
`compute_grpo_outcome_advantage` in
[`verl/trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py).

## At the end, you should be able to answer

- What signal remains when every sample in a group gets an identical reward, and
  what does DAPO do about it? (`GRPO/run_grpo.py` prints this happening.)
- Why does normalizing across the whole batch instead of per group reintroduce
  prompt-difficulty bias?
- There is one advantage per response, copied across all of its tokens. Why is
  there no per-token credit assignment here, when PPO's GAE produces exactly that?
- GRPO needs no critic. What does it give up in exchange, and when would you
  still reach for PPO?
- `Agent0/` builds ADPO on top of this function. Read `adpo_advantage` in
  `Agent0/common.py` next to your answer here — the difference is one multiply.
