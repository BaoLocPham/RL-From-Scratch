# Build DPO from scratch, the simple way

Do not open `simple_dpo.py` (the reference) before you finish the exercise.
Work from the docstrings in `from_scratch/simple_dpo.py` and the grader's
messages.

`SimpleGRPO/` dropped PPO's critic. **DPO** (Direct Preference Optimization,
[Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) drops the rest of RL:
no reward, no advantage, no rollouts while training, no ratio to θ_old, no clip.
It trains on **pairs that are already labelled**, "this attempt is better than
that one", with a loss that is a logistic regression on each pair. This module
builds it on SimplePPO's three-turn agent, with **no tokens, no masks, no
verl**. `DPO/` then has the same loss the way verl's online-DPO recipe writes
it.

## What is left of the loop

```
for iteration:
    pairs = the next batch of labelled pairs         sampled from π_ref and labelled BEFORE training
    for epoch in range(10):
        for minibatch of 4 pairs:                    stage 3
            logits = [log π(chosen) − log π(rejected)]            stage 1: log π of a whole attempt
                   − [log π_ref(chosen) − log π_ref(rejected)]
            loss = −log σ(β · logits)                             stage 2: DPO eq. 7
            zero_grad, backward, step
```

| | SimpleGRPO | SimpleDPO |
|---|---|---|
| What training sees | a reward per attempt (+1 / −1, −0.1 per search) | which of two attempts is better |
| Where the data comes from | the current policy, every iteration | π_ref, once, before training (offline) |
| Advantage | (R − group mean) / group std | none |
| Loss | L^CLIP + 0.04·KL(π_θ ‖ π_ref) | −log σ(β·logits): π_ref is *inside* the loss |
| Ratio to θ_old, clip | yes | none: the loss does not care who sampled the pairs |
| Networks | policy (+ frozen π_ref) | the same |

## The toy: SimpleGRPO's agent, in labelled pairs

The agent is SimplePPO's, unchanged: three turns of SEARCH (costs 0.1) or SKIP,
then it answers, with the same 12 states and 12 × 2 table of logits.
`pair_env.py` changes what training sees:

1. **Pairs.** Each question is attempted twice, and the two attempts are kept
   side by side.
2. **A label, not a reward.** Real DPO trains on a dataset that people (or a
   judge model) labelled before training: for each prompt, which of two
   answers is better. The toy has no people, so `pair_env.py` writes the labels
   itself, with the simplest rule:

   - **the attempt with the higher average total is chosen**, the other rejected;
   - **two equally good attempts are skipped**: same question type and same
     number of searches, so there is nothing to learn from the pair.

| Average total | 0 searches | 1 | 2 | 3 |
|---|---|---|---|---|
| HARD | −1.0 | −0.1 | **0.8** | 0.7 |
| EASY | **1.0** | 0.8 | 0.6 | 0.4 |

So on HARD, two searches are always chosen over three, one or none. On EASY,
fewer searches are always chosen. `../how_the_toy_works.md` shows where these
numbers come from, turn by turn.

`labels="outcome"` labels the two attempts by the rewards they actually got
instead (right or wrong is a coin flip for one search on HARD), as verl's
online-DPO recipe labels responses with a checker. The demo shows why that is
a weaker label.

## Where the loss comes from

RLHF (and SimpleGRPO, with its KL) maximises reward while staying near π_ref:
max E[r] − β·KL(π ‖ π_ref). Its best policy is known exactly (DPO eq. 4):

```
π*(y) = π_ref(y) · exp(r(y) / β) / Z        so        r(y) = β · log(π*(y) / π_ref(y)) + β · log Z
```

DPO's paper assumes the labels come from a judge who picks the better answer
more often the bigger the gap: p(A labelled better) = σ(r(A) − r(B)). Put the r
above into that. Z is the same for both attempts at one question, so it
cancels, and what is left mentions only the policy and π_ref. Fit the policy to
the labels by maximum likelihood, and that is DPO's loss. Our labeller is the
most confident judge there is: it always picks the better attempt.

## The stages

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `sequence_logp` | — | How do you score a whole attempt, not one turn? |
| 2 | `dpo_loss`, `implicit_reward` | eq. 7 | How do four log-probs become one loss? |
| 3 | `dpo_update` (the log-probs and the loss) | — | What is left of the RL loop? |
| 4 | none | — | Does your DPO learn the toy? |

Nothing here imports your earlier exercises: DPO keeps none of PPO's loss.

## How to start

1. Read `./scripts/run_simple_dpo.sh steps`: one pair through every piece, with the numbers.
2. Fill the lines marked `TODO stage 1` in `from_scratch/simple_dpo.py`.
3. Try them: `python SimpleDPO/from_scratch/simple_dpo.py` prints your value next to the expected one.
4. Check them: `./scripts/run_simple_dpo.sh check` grades the stages in order, with a hint for the likely mistake.
5. When all pass: `./scripts/run_simple_dpo.sh diff` proves your walkthrough matches the reference, and
   `RL_IMPL=scratch ./scripts/run_simple_dpo.sh run` runs the comparison with your code.

## What each choice does (`./scripts/run_simple_dpo.sh run`, about a minute)

The same budget for every run: 20 seeds, 60 iterations of 8 questions × 2
attempts (16 attempts, SimpleGRPO's budget), 10 epochs of minibatches of 4
pairs. Of the 480 questions, about 324 give a labelled pair; the rest are two
equally good attempts, skipped. Start J 0.388, best 0.90. "Stuck" means J below
0.7 after 60 iterations.

**Why lr 3.0, ten times SimpleGRPO's 0.3.** DPO's gradient carries a factor
β = 0.1 (walkthrough, step 6), so the same SGD needs a step 1/β times bigger.

| Run | What changes | J@10 | J@30 | J@60 | Stuck |
|---|---|---|---|---|---|
| **DPO** | pairs from π_ref, the better attempt chosen, β 0.1 | 0.870 | 0.894 | **0.899** | 0/20 |
| 1 epoch | each batch of pairs used once | 0.620 | 0.800 | 0.853 | 0/20 |
| online | pairs sampled from the current policy | 0.880 | 0.896 | 0.898 | 0/20 |
| by outcome | pairs from π_ref, labelled by the rewards they got | 0.705 | 0.818 | 0.844 | 2/20 |
| by outcome, online | the same, from the current policy | 0.585 | 0.666 | 0.654 | **11/20** |
| GRPO G 2, Dr.GRPO | SimpleGRPO, same 16 attempts, R − pair mean | 0.665 | 0.777 | 0.811 | 0/20 |
| GRPO G 2 | the same, divided by the pair's std | 0.597 | 0.684 | 0.666 | 12/20 |

An honest reading:

- **DPO learns the toy with no reward at all:** 0.899 of a possible 0.900,
  from about 324 labelled pairs sampled before training. One caveat: its labels
  come from each attempt's *true* average, which the RL rows never see. That,
  more than DPO itself, is why it beats them here.
- **Reusing pairs needs no clip.** DPO's loss has no ratio to θ_old, so ten
  epochs on a batch are safe by construction. One epoch is simply less training.
- **Online pairs** (sampled from the improving policy, as verl's recipe does)
  make no difference here.
- **Labelling by outcome is the fair comparison:** the same information GRPO
  sees. From π_ref's pairs it works (0.844): many are two searches against none,
  which the coin always decides the right way. Online it gets stuck, and every
  stuck run searches exactly once on HARD. There, every pair left is decided
  by a coin: one lucky search (0.9) beats two (0.8) half the time, and loses
  to none (−1.1 vs −1.0) half the time. A label says only *which* attempt won,
  never by how much.
- **GRPO with groups of 2 has the same failure.** Divided by the std, a pair's
  advantages are always ±0.71, a sign and nothing more (12/20 stuck). Dr.GRPO
  keeps the size of the gap (0.811).

**The policy becomes a scorer.** β·log(π/π_ref) is how much more likely the
policy has made an attempt than π_ref did. Averaged over the 20 seeds, relative
to 0 searches, it is HARD [0, 2.44, 4.53, 4.42] and EASY [0, −1.31, −3.48, −4.39].
That is the same order as the average totals: HARD [0, 0.9, 1.8, 1.7] and EASY
[0, −0.2, −0.4, −0.6]. The numbers are bigger, and keep growing with training,
because labels that never disagree never tell the policy to stop (walkthrough,
step 7).

## From here to `DPO/`

| | `SimpleDPO/` | `DPO/` (verl's online-DPO recipe) |
|---|---|---|
| One step | one turn: SEARCH or SKIP | one token |
| log π(attempt) | `sequence_logp`: `.log_prob(actions).sum(1)` | `get_batch_logps`: shift the logits, mask the prompt with −100, sum |
| The labels | the better average, or the rewards, in `pair_env.py` | `compute_onlinedpo_pref`: two responses per prompt, argmax of the reward |
| The loss | `dpo_loss` | `compute_online_dpo_loss`, `loss_type="sigmoid"`, plus `"ipo"` and label smoothing |
| π_ref's log-probs | `sequence_logp(ref_policy, ...)` | `(ref_log_prob * response_mask).sum(-1)` |
| Where the pairs come from | π_ref (offline) by default | the current policy (online), π_ref optionally synced every `ref_update_freq` steps |

## Terms

The PPO terms are in `SimplePPO/README.md`, the GRPO ones in
`SimpleGRPO/README.md`. This module adds:

- **pair**: two attempts at one question, ordered (chosen, rejected).
- **label**: which attempt of a pair is better. Made before training; DPO only
  ever sees labels.
- **π*_β**: π_ref · exp(r/β) / Z, the best policy under a KL to π_ref, the
  starting point of DPO's derivation.
- **implicit reward**: β·log(π/π_ref): how much more likely the policy has made
  an attempt than π_ref did. DPO raises it for chosen attempts.
- **offline / online**: pairs sampled once from π_ref (the published DPO), or
  sampled from the current policy every iteration.

## At the end, you should be able to answer

- Why is log π(attempt) a sum over the turns and not a mean? What would a mean
  do to two answers of different lengths?
- Z, the normaliser of π*, is intractable for an LLM. Where does it go?
- Our labeller never disagrees with itself. Why do the policy's scores keep
  growing, and what does IPO (`loss_type="ipo"` in `DPO/`) do about it?
- DPO reuses each batch of pairs ten times with no clip. Why was that unsafe
  for VPG and safe here?
- Labelled by outcome, online DPO gets stuck where Dr.GRPO does not. What does
  GRPO's advantage carry that a label cannot?
