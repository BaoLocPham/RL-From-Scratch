# Build DPO from scratch, the simple way

Do not open `simple_dpo.py` (the reference) before you finish the exercise.
Work from the docstrings in `from_scratch/simple_dpo.py` and the grader's
messages.

`SimpleGRPO/` dropped PPO's critic. **DPO** (Direct Preference Optimization,
[Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) drops the rest of RL:
no reward, no advantage, no rollouts while training, no ratio to θ_old, no clip.
It trains on **pairs a rater has already judged** — this attempt beat that one —
with a loss that is a logistic regression on each pair. This module builds it
on SimplePPO's three-turn agent, with **no tokens, no masks, no verl**. `DPO/`
then has the same loss the way verl's online-DPO recipe writes it.

## What is left of the loop

```
for iteration:
    pairs = the next 8 (chosen, rejected) pairs      sampled from π_ref BEFORE training, judged by a rater
    for epoch in range(10):
        for minibatch of 4 pairs:                    stage 3
            logits = [log π(chosen) − log π(rejected)]            stage 1: log π of a whole attempt
                   − [log π_ref(chosen) − log π_ref(rejected)]
            loss = −log σ(β · logits)                             stage 2: DPO eq. 7
            zero_grad, backward, step
```

| | SimpleGRPO | SimpleDPO |
|---|---|---|
| What training sees | a reward per attempt (+1 / −1, −0.1 per search) | which of two attempts a rater preferred |
| Where the data comes from | the current policy, every iteration | π_ref, once, before training (offline) |
| Advantage | (R − group mean) / group std | none |
| Loss | L^CLIP + 0.04·KL(π_θ ‖ π_ref) | −log σ(β·logits): π_ref is *inside* the loss |
| Ratio to θ_old, clip | yes | none: the loss does not care who sampled the pairs |
| Networks | policy (+ frozen π_ref) | the same |

## The toy: SimpleGRPO's agent, in pairs, judged by a rater

The agent is SimplePPO's, unchanged: three turns of SEARCH (costs 0.1) or SKIP,
then it answers, with the same 12 states and 12 × 2 table of logits.
`pair_env.py` changes what training sees:

1. **Pairs.** Each question is attempted twice (a group of 2, in SimpleGRPO's
   terms), stored pair by pair.
2. **A preference, not a reward.** A rater says which attempt is better. The
   DPO paper models a human rater as **Bradley–Terry** (its eq. 1):
   p(A over B) = σ(r(A) − r(B)), where r is the attempt's quality: the average
   total reward of its number of searches.

| r(q, k) | 0 searches | 1 | 2 | 3 |
|---|---|---|---|---|
| HARD | −1.0 | −0.1 | **0.8** | 0.7 |
| EASY | **1.0** | 0.8 | 0.6 | 0.4 |

The rater judges the answer, not the luck: on HARD, one search is a coin flip,
and the rater knows it. The verdicts are still noisy. Two searches beat none on
HARD only 86% of the time (σ(1.8)), and beat three only 52% of the time (σ(0.1)).

`labels="outcome"` ranks the two attempts by the rewards they actually got
instead, as verl's online-DPO recipe ranks responses with a checker. The demo
shows why that is a different thing.

## Where the loss comes from

RLHF (and SimpleGRPO, with its KL) maximises reward while staying near π_ref:
max E[r] − β·KL(π ‖ π_ref). Its best policy is known exactly (DPO eq. 4):

```
π*(y) = π_ref(y) · exp(r(y) / β) / Z        so        r(y) = β · log(π*(y) / π_ref(y)) + β · log Z
```

Put that r into Bradley–Terry. Z is the same for both attempts at one question,
so it cancels, and what is left mentions only the policy and π_ref. Fit the
policy to the rater's verdicts by maximum likelihood, and that is DPO's loss.

Two things follow, and the toy shows both exactly:

- **β sets a target.** DPO's loss is minimised by π*_β = π_ref · exp(r/β) / Z.
  On the toy it can be enumerated (`pair_env.optimal_policy`): J(π*_β) is 0.711
  at β 0.5, 0.871 at β 0.1, and 0.900 at β 0.02, against 0.388 for π_ref.
- **The policy is a reward model.** β·log(π/π_ref), the *implicit reward*,
  should come out as r plus a constant per question. Step 8 of the walkthrough
  and the end of the demo compare it with the rater's r.

## The stages

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `sequence_logp` | — | How do you score a whole attempt, not one turn? |
| 2 | `dpo_loss`, `implicit_reward` | eq. 7, §5 | How do four log-probs become one loss, and where did the reward go? |
| 3 | `dpo_update` (the log-probs and the loss) | — | What is left of the RL loop? |
| 4 | none | — | Does your DPO learn the toy, and recover the rater's reward? |

Nothing here imports your earlier exercises: DPO keeps none of PPO's loss.

## How to start

1. Read `./scripts/run_simple_dpo.sh steps`: one pair through every piece, with the numbers.
2. Fill the lines marked `TODO stage 1` in `from_scratch/simple_dpo.py`.
3. Try them: `python SimpleDPO/from_scratch/simple_dpo.py` prints your value next to the expected one.
4. Check them: `./scripts/run_simple_dpo.sh check` grades the stages in order, with a hint for the likely mistake.
5. When all pass: `./scripts/run_simple_dpo.sh diff` proves your walkthrough matches the reference, and
   `RL_IMPL=scratch ./scripts/run_simple_dpo.sh run` runs the comparison with your code.

## What each choice does (`./scripts/run_simple_dpo.sh run`, about two minutes)

The same budget for every run: 20 seeds, 60 iterations of 8 pairs (16
attempts, SimpleGRPO's budget), SGD at lr 0.3 / β, 10 epochs of minibatches of
4 pairs. Start J 0.388, best 0.90. "Stuck" means J below 0.7 after 60
iterations. "π*_β" is the J of the policy the loss is minimised by. "r err" is
how far the implicit reward is from the rater's r (0 = recovered exactly).

**Why lr 0.3 / β.** DPO's gradient carries a factor β (walkthrough step 6), so
at β 0.1 the same SGD needs a step ten times SimpleGRPO's. Scaling lr by 1/β
keeps the first step the same size at every β, so the β rows differ in their
target, not in their step size.

| Run | What changes | J@10 | J@30 | J@60 | Stuck | π*_β | r err |
|---|---|---|---|---|---|---|---|
| **DPO** | pairs from π_ref, the rater, β 0.1 | 0.684 | 0.804 | **0.810** | 0/20 | 0.871 | **0.33** |
| 1 epoch | each batch of pairs used once | 0.472 | 0.603 | 0.701 | 8/20 | 0.871 | 0.68 |
| β 0.5 | anchored harder (lr 0.6) | 0.617 | 0.700 | 0.684 | 9/20 | 0.711 | 0.71 |
| β 0.02 | anchored less (lr 15) | 0.693 | 0.809 | 0.817 | 0/20 | 0.900 | 0.43 |
| online | pairs sampled from the current policy | 0.716 | 0.806 | 0.824 | 0/20 | 0.871 | 0.73 |
| online, β 0.5 | the same, anchored harder | 0.670 | 0.728 | 0.750 | 4/20 | 0.711 | 0.83 |
| by outcome | pairs from π_ref, ranked by the rewards they got | 0.705 | 0.818 | 0.844 | 2/20 | 0.871 | 1.52 |
| by outcome, online | the same, from the current policy | 0.585 | 0.666 | 0.654 | **11/20** | 0.871 | 0.85 |
| GRPO G 2, Dr.GRPO | SimpleGRPO, same 16 attempts, R − pair mean | 0.665 | 0.777 | 0.811 | 0/20 | — | — |
| GRPO G 2 | the same, divided by the pair's std | 0.597 | 0.684 | 0.666 | 12/20 | — | — |

An honest reading:

- **DPO learns the toy with no reward at all**, from 480 judged pairs sampled
  before training. It ends below its target (0.810 vs 0.871) mostly because the
  rater barely separates close answers: two searches over three on HARD only
  52% of the time, none over one on EASY only 55%. Verdicts that close do not
  settle it, so some mass stays on three searches.
- **The policy recovers the reward.** Averaged over the 20 seeds, β·log(π/π_ref)
  relative to 0 searches is HARD [0, 0.90, 1.62, 1.78] against the rater's
  [0, 0.9, 1.8, 1.7], and EASY [0, −0.23, −0.42, −0.50] against
  [0, −0.2, −0.4, −0.6]. Trained only on which attempt won, it has learned how
  good each choice is.
- **Reusing pairs needs no clip.** DPO's loss has no ratio to θ_old, so ten
  epochs on a batch are safe by construction. One epoch is simply ten times less
  training.
- **β sets the target.** β 0.5 is held near π_ref (0.684, target 0.711). Its
  target is barely above 0.7, so its "stuck" runs are just runs at their target.
  β 0.02 aims at 0.900 but gets no further than β 0.1 in 60 iterations.
- **Online pairs trade the anchor for fresher data.** Sampling each
  iteration's pairs from the policy (as verl's recipe does) gives a slightly
  higher J. But the implicit reward drifts from r (r err 0.73 vs 0.33), and β
  stops holding the policy at π*_β: at β 0.5 it ends above its target (0.750 vs
  0.711).
- **Ranking by outcome is a different signal.** From π_ref's pairs it works
  (0.844): many are two searches against none, which the coin always decides
  the right way. Online it gets stuck, and every stuck run searches exactly once
  on HARD. There, every pair left is a coin flip: one lucky search (0.9) beats
  two (0.8) half the time, and loses to none (−1.1 vs −1.0) half the time. DPO
  keeps only *which* attempt won, never by how much.
- **GRPO with groups of 2 has the same failure.** Divided by the std, a pair's
  advantages are always ±0.71, a sign and nothing more (12/20 stuck). Dr.GRPO
  keeps the size of the gap (0.811).

## From here to `DPO/`

| | `SimpleDPO/` | `DPO/` (verl's online-DPO recipe) |
|---|---|---|
| One step | one turn: SEARCH or SKIP | one token |
| log π(attempt) | `sequence_logp`: `.log_prob(actions).sum(1)` | `get_batch_logps`: shift the logits, mask the prompt with −100, sum |
| The pairs | the rater, or the rewards, in `pair_env.py` | `compute_onlinedpo_pref`: two responses per prompt, argmax of the reward |
| The loss | `dpo_loss` | `compute_online_dpo_loss`, `loss_type="sigmoid"`, plus `"ipo"` and label smoothing |
| π_ref's log-probs | `sequence_logp(ref_policy, ...)` | `(ref_log_prob * response_mask).sum(-1)` |
| Where the pairs come from | π_ref (offline) by default | the current policy (online), π_ref optionally synced every `ref_update_freq` steps |

## Terms

The PPO terms are in `SimplePPO/README.md`, the GRPO ones in
`SimpleGRPO/README.md`. This module adds:

- **pair**: two attempts at one question, ordered (chosen, rejected).
- **Bradley–Terry**: the rater model p(A over B) = σ(r(A) − r(B)).
- **π*_β**: π_ref · exp(r/β) / Z, the best policy under a KL to π_ref, and
  what DPO's loss is minimised by.
- **implicit reward**: β·log(π/π_ref), the reward a DPO policy believes in.
- **offline / online**: pairs sampled once from π_ref (the published DPO), or
  sampled from the current policy every iteration.

## At the end, you should be able to answer

- Why is log π(attempt) a sum over the turns and not a mean? What would a mean
  do to two answers of different lengths?
- Z, the normaliser of π*, is intractable for an LLM. Where does it go?
- At β 0.1, what would DPO do with a pair the rater *always* decides the same
  way, and what does that say about deterministic preferences? (IPO,
  `loss_type="ipo"` in `DPO/`, is one answer.)
- DPO reuses each batch of pairs ten times with no clip. Why was that unsafe
  for VPG and safe here?
- Ranked by outcome, online DPO gets stuck where Dr.GRPO does not. What does
  GRPO's advantage carry that a preference cannot?
