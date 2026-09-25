# Build vanilla policy gradient from scratch

Do not open `vpg.py` (the reference) before you finish the exercise. Work from
the docstrings in `from_scratch/vpg.py` and the grader's messages; reading the
reference turns this into transcription.

This follows the PPO paper's Section 2.1 (eq. 1-2), plus the objective J(θ)
the paper never names (see the Notion page "Agentic RL Survey - Policy
Optimization", §0). The toy, the policy and the rollout collector are given.

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `expected_reward` | What is J(θ), and why can training never see it? |
| 2 | `compute_advantage` | Why subtract a baseline from the reward? |
| 3 | `pg_loss` | Why is L^PG negated, and why must it ignore `old_logp`? |
| 4 | `vpg_update` | Which line changes the model, and which never changes? |
| 5 | none | What happens when your correct pieces reuse one batch? |

## How to start

Each stage is one or two TODO lines in `from_scratch/vpg.py`, and each has a worked example
with the numbers you should get, plus a hint naming the PyTorch call to use.

1. Open `from_scratch/vpg.py` and fill the lines marked `TODO stage 1`.
2. Try them: `python VPG/from_scratch/vpg.py` prints your result for every
   stage next to the expected one; unfinished stages say "not done yet".
3. Check them: `./scripts/run_vpg.sh check` grades the stages in order and stops
   at the first one that is not right yet, with a hint about the likely mistake.
4. Repeat for stages 2, 3 and 4. When all pass, `./scripts/run_vpg.sh diff` proves
   your walkthrough output matches the reference line for line.

Stage 5 needs no new code: it runs your four pieces on the unlucky batch from
`./scripts/run_vpg.sh run` and should reproduce the flaw, J falling from 0.600 to 0.373.

## The running example

Stages 3 and 4 use the batch from the notes: 10 rollouts on HARD questions,
collected at p(tool) = 0.40. 4 called the tool and went well (Â = +3); 6
answered directly and went badly (Â = −2).

## Terms in the VPG logs

The toy is a tool-calling agent: each question is HARD or EASY (50/50), and the
policy either answers directly or calls a search tool. Average rewards:

|      | answer | tool |
|------|--------|------|
| HARD | 0.2    | 1.0  |
| EASY | 0.8    | 0.5  |

- **p(tool|HARD)**: chance the policy calls the tool on a HARD question.
  Ideal 1.0, since search helps there.
- **p(tool|EASY)**: chance it calls the tool on an EASY question. Ideal 0.0,
  since search wastes time there. Both start at 0.40; good training pushes HARD
  up and EASY down.
- **true J**: J(θ) = E[R], the policy's expected reward over infinitely many
  questions. Computed exactly from the table above; training never sees it.
  At the start, HARD earns 0.6×0.2 + 0.4×1.0 = 0.52 and EASY 0.6×0.8 + 0.4×0.5
  = 0.68, so J = 0.60. The best possible (always search on HARD, always answer
  on EASY) is J = (1.0 + 0.8) / 2 = 0.90.
- **rollout**: one question, the action taken, and its noisy reward. The
  expensive part. A **batch** is 16 rollouts collected by one model, **θ_old**.
- **advantage** Â: reward − mean(reward), "better or worse than usual?". A
  simplified version of the paper's; see *A simplification: the advantage* below.
- **update**: one `optimizer.step()`, the only line that changes the model.
  Only the first update on a batch is taken at θ_old, where the batch's gradient
  is valid.
- **epoch**: one full pass over the batch. These toys use the whole batch per
  update, so 1 epoch = 1 update. Vanilla PG runs 1 epoch per batch; the VPG
  demo's shortcut runs 100 epochs on one batch.
- **iteration**: collect a fresh batch, then run all its epochs on it.

## A simplification: the advantage

This toy computes the advantage as

```python
advantage = reward - reward.mean()      # vpg.py, compute_advantage
```

That is a simplified version of what the PPO paper uses. The paper's §2 never
defines Â_t (eq. 1–5 only call it "an estimator of the advantage function"). Its
§5 does, as GAE, using a learned value network V(s) (the critic), which predicts
how much reward to expect from a state:

```
Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t+1} δ_{T−1}      (eq. 11)
δ_t = r_t + γ V(s_{t+1}) − V(s_t)                         (eq. 12)
```

| | Paper, §5 (GAE) | This toy |
|---|---|---|
| Steps per episode | many, discounted by γ and λ | one: a question, an action, a reward, done |
| Expected reward V(s) | a learned value network | none: the batch's average reward stands in |
| Advantage | eq. 11–12 above | Â = r − mean(r) |

Step by step:

1. **One step per episode.** There is no s_{t+1}, so V(s_{t+1}) = 0 and eq. 11
   collapses to Â = r − V(s).
2. **No value network.** The batch's average reward replaces V(s), giving
   Â = r − mean(r). This is "REINFORCE with a baseline" (Williams, 1992).

**What it gives up:** the batch mean is one number for every question, while a
real V(s) would expect more on EASY than on HARD (0.68 vs 0.52 at the start).
The gradient is still right on average (a baseline that does not depend on the
action never biases it), but noisier, which is one reason a 16-rollout batch
can mislead. It is not what causes the flaw the demo shows; that comes from
reusing a batch.

The full GAE is in `PPO/` (`compute_gae_advantage_return`). The Notion page
"Agentic RL Survey - Policy Optimization" walks the same steps right after eq. 1.

## At the end, you should be able to answer

- J(θ) and L^PG both go up when the policy improves. Why does L^PG's value mean
  nothing on its own, while J's does?
- If you removed the baseline in stage 2, every reward in this toy is mostly
  positive. What would the update do to actions that were merely average?
- In stage 3, the loss did not change when only `old_logp` changed. Why is that
  exactly the flaw, and which equation on the Notion page fixes it?
- In stage 4, which of `loss`, the batch, and the parameters change between
  the first and second update? Which of those makes update 2 stale?
- In stage 5, update 1 moved J from 0.600 to 0.592, and 100 updates moved it to
  0.373. Both used a correct gradient of a correct loss. What went wrong?
