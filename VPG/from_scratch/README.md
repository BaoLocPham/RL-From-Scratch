# Build vanilla policy gradient from scratch

Do not open `../vpg.py` first. Work from the docstrings in `vpg.py` and the
grader's messages; reading the reference turns this into transcription.

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

Each stage is one or two TODO lines in `vpg.py`, and each has a worked example
with the numbers you should get, plus a hint naming the PyTorch call to use.

1. Open `vpg.py` and fill the lines marked `TODO stage 1`.
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
