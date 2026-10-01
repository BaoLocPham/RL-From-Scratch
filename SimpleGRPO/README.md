# Build GRPO from scratch, the simple way

Do not open `simple_grpo.py` (the reference) before you finish the exercise.
Work from the docstrings in `from_scratch/simple_grpo.py` and the grader's
messages.

`SimplePPO/` gave each turn its own advantage, with a critic and GAE, and on
that toy the critic barely paid for itself. **GRPO** (Group Relative Policy
Optimization, [DeepSeekMath](https://arxiv.org/abs/2402.03300) §4.1) drops the
critic. It answers the **same question several times** and judges each attempt
against the others. This module builds exactly that, on SimplePPO's three-turn
agent, with **no tokens, no masks, no verl**. `GRPO/` then has the same
advantage the way verl writes it.

## Two changes from SimplePPO, and nothing else

```
for iteration:
    batch = rollout(policy, 2 questions x 8 attempts)   θ_old; a group = the attempts at one question
    advantages = group_advantage(batch)                 stage 1: no critic, no GAE
    flatten to 48 samples
    for epoch in range(10):
        for minibatch of 16:                            stage 3
            loss = policy_loss                          your SimplePPO clip, imported, unchanged
                 + 0.04 * kl_penalty                    stage 2: stay near π_ref
            zero_grad, backward, step
```

| | SimplePPO | SimpleGRPO |
|---|---|---|
| Advantage | GAE from a learned critic V(s): a different value per turn | (R − group mean) / group std: one value per attempt, copied to all its turns |
| Baseline | V(s), trained by a value loss | the other attempts at the same question |
| Loss | L^CLIP + 0.5·L^VF − 0.01·S | L^CLIP + 0.04·KL(π_θ ‖ π_ref) |
| Networks | policy + critic | policy only (π_ref is a frozen copy) |
| The clip | eq. 7 | the same function |

## The toy: SimplePPO's agent, in groups, graded right or wrong

The agent is SimplePPO's, unchanged: three turns of SEARCH (costs 0.1) or SKIP,
then it answers, with the same 12 states and the same 12 × 2 table of logits.
`group_env.py` changes two things to make it GRPO's setting:

1. **Groups.** Each batch is 2 questions × 8 attempts (verl's `rollout.n = 8`),
   stored group by group, so `.view(-1, 8)` puts each question's attempts in a row.
2. **A verifiable reward.** GRPO is usually trained against a checker that says
   right or wrong. So the answer scores **+1 if right and −1 if wrong**:

| Chance of being right | 0 searches | 1 | 2 | 3 |
|---|---|---|---|---|
| HARD | 0% | 50% | 100% | 100% |
| EASY | 100% | 95% | 90% | 85% |

Its average, 2·p − 1, is exactly SimplePPO's answer quality. So the true J, the
best policy (search twice on HARD, never on EASY) and J = 0.90 are all the same
as SimplePPO's.

**J**, the true reward every table here reports, is the average total reward of
one attempt, with k = a₀ + a₁ + a₂ searches and each choice made in its state
s_t = (question type, turn, searches so far):

```
J(θ) = Σ_q ½ · Σ_(a₀,a₁,a₂)  π(a₀|s₀) · π(a₁|s₁) · π(a₂|s₂) · (2·p(right | q, k) − 1 − 0.1·k)
```

That is 2 question types × 8 action sequences = 16 terms, summed exactly by
`env.true_reward`. Training never sees it, only single coin-flip rewards. Step 1
of the walkthrough works it through: the starting policy scores 0.388, and the
best 0.5 × 0.8 + 0.5 × 1.0 = 0.90. Only the noise differs: a coin instead of a bell curve. That
makes whole groups tie, which is one of GRPO's real behaviours (step 6).

## The stages

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `group_advantage` | DeepSeekMath §4.1.2 | How can the other attempts replace a critic? |
| 2 | `kl_penalty` | eq. 4 (k3) | How do you estimate a KL from the one action you took? |
| 3 | `grpo_update` (the KL and the loss) | eq. 3 | What is left of PPO's loss once the critic is gone? |
| 4 | none | — | Does your GRPO learn "HARD: search twice, EASY: never"? |

**Finish `SimplePPO/` first.** Stage 3 imports *your* `policy_loss` from
`SimplePPO/from_scratch/simple_ppo.py`: GRPO keeps PPO's clip, so there is
nothing to rewrite. Stages 1 and 2 do not need it.

## How to start

1. Read `./scripts/run_simple_grpo.sh steps`: one group through every piece, with the numbers.
2. Fill the lines marked `TODO stage 1` in `from_scratch/simple_grpo.py`.
3. Try them: `python SimpleGRPO/from_scratch/simple_grpo.py` prints your value next to the expected one.
4. Check them: `./scripts/run_simple_grpo.sh check` grades the stages in order, with a hint for the likely mistake.
5. When all pass: `./scripts/run_simple_grpo.sh diff` proves your walkthrough matches the reference, and
   `RL_IMPL=scratch ./scripts/run_simple_grpo.sh run` runs the comparison with your code.

## What each choice does (`./scripts/run_simple_grpo.sh run`, about three minutes)

The same budget for every run: 20 seeds, 60 iterations of 16 episodes, SGD at
lr 0.3, 10 epochs of minibatches of 16 steps. Start J 0.388, best 0.90.
"Stuck" means J below 0.7 after 60 iterations; "dead" is the fraction of groups
in the last batch whose attempts all scored the same.

| Run | What changes | J@10 | J@30 | J@60 | Stuck | Dead@60 |
|---|---|---|---|---|---|---|
| **GRPO** | G 8, std divide, β 0.04 | 0.753 | 0.833 | **0.839** | 0/20 | 0.60 |
| 1 epoch | each batch used once | 0.509 | 0.669 | 0.771 | 0/20 | 0.12 |
| no KL | β 0 | 0.761 | 0.846 | 0.860 | 0/20 | 0.65 |
| Dr.GRPO | no divide by the std | 0.709 | 0.802 | 0.832 | 0/20 | 0.12 |
| G 2 | 8 questions × 2 attempts | 0.597 | 0.684 | 0.666 | **12/20** | 0.56 |
| G 2, Dr.GRPO | 8 × 2, no divide | 0.665 | 0.777 | 0.811 | 0/20 | 0.53 |
| batch mean | R − the whole batch's mean, no groups | 0.719 | 0.803 | 0.833 | 0/20 | 0.10 |
| PPO | SimplePPO's critic + GAE, same rewards | 0.745 | 0.837 | 0.869 | 0/20 | — |

An honest reading:

- **The std divide breaks small groups.** With two attempts, (R − mean) / std
  is always ±0.71, however far apart the rewards are. On a HARD question, one
  search is a coin flip. Winning it beats two searches by 0.1, and losing it
  loses by 1.9, yet both push equally hard. So the pushes cancel, and every
  stuck run sits between one and two searches. Dr.GRPO keeps the size of the
  gap and none get stuck.
- **At G 8 the divide does no harm,** but it explains the dead column. As a
  group nearly agrees, its std shrinks, and GRPO blows the last small
  differences up to full size. So GRPO's policy sharpens until whole groups
  tie: its average entropy on the states it visits ends at 0.26, against
  Dr.GRPO's 0.42.
- **The KL costs a little here** (+0.021 without it). It pulls toward the
  starting policy, which on this toy is bad. On an LLM, π_ref is the
  pretrained model, and staying near it is the point. DAPO drops the KL
  anyway.
- **Groups beat the batch mean by nothing here.** With two question types,
  which the policy sees, one batch-wide baseline is already fair. Groups pay
  when every prompt has its own difficulty, as in an LLM's dataset. Step 4 of
  the walkthrough shows the bias they remove.
- **PPO's critic wins on this toy** (0.869 vs 0.839). GRPO's case isn't accuracy
  here. It is the critic it doesn't need: on an LLM, that is a second model as
  big as the policy, its memory and its value loss, all to predict one reward
  per response.

## From here to `GRPO/`

| | `SimpleGRPO/` | `GRPO/` (verl) |
|---|---|---|
| One step | one turn: SEARCH or SKIP | one token |
| Which rows form a group | consecutive, `.view(-1, group_size)` | an `index` array of prompt ids (`uid`), grouped in a dict |
| One number per attempt | `rewards.sum(1)` | `token_level_rewards.sum(-1)` |
| Copy it to every step | `.expand_as(rewards)` | `* response_mask`, so padding stays 0 |
| A group of one | never happens here | mean 0, std 1, so the sample is not wasted |
| Dr.GRPO | `scale_by_std=False` | `norm_adv_by_std_in_grpo=False` |
| The KL | k3, `.mean()` | `kl_penalty(..., "k3")`, then `agg_loss` over the mask |
| The clip | SimplePPO's `policy_loss` | PPO's `compute_policy_loss` |

So `GRPO/`'s exercise is this stage 1 at verl's shapes. It adds no new
algorithm, only the bookkeeping.

Next on the toy track, `SimpleDPO/` takes away the reward too: the same agent,
two attempts per question, and only a rater's verdict on which was better.

## Terms

The PPO terms are in `SimplePPO/README.md`. This module adds:

- **group**: the G attempts at one question. G is `group_size` here and
  `rollout.n` in verl.
- **outcome supervision**: one reward per attempt, with no say over which turn
  earned it.
- **π_ref**: the frozen reference policy the KL keeps the policy near: the
  starting policy here, the model before RL on an LLM.
- **k3**: the KL estimator x − log x − 1, with x = π_ref / π_θ, taken on the
  action actually chosen.
- **dead group**: a group whose attempts all scored the same. Its std is 0,
  every advantage in it is 0, and it adds no gradient.
- **Dr.GRPO**: GRPO without the divide by the group std.

## At the end, you should be able to answer

- In stage 1, every turn of an attempt gets the same advantage. How can GRPO
  still learn that a *third* search is wasted on a HARD question?
- The group mean includes the attempt being judged. What does that do to the
  advantages of a group of 2, and why does it matter less for a group of 64
  (DeepSeekMath's G)?
- In stage 2, why can't you compute the exact KL for an LLM as TRPO did? Why is
  k3 still an honest estimate of it?
- GRPO ends with most groups dead. What does DAPO's dynamic sampling do about
  it, and why is a dead group a sign of success rather than failure?
- PPO wins on this toy. Name two things about LLM training that turn that
  around.
