# Build PPO from scratch, the simple way

Do not open `simple_ppo.py` (the reference) before you finish the exercise.
Work from the docstrings in `from_scratch/simple_ppo.py` and the grader's
messages.

`Surrogates/` ended with a definition: **PPO = the loop with L^CLIP in the
slot, plus a value loss, an entropy bonus and GAE.** This module builds exactly
that, the paper's algorithm and nothing more, on a toy that is one small step
beyond the toy track: **no tokens, no masks, no verl.** `PPO/` then turns the
same algorithm into the LLM version.

## The one new idea: several decisions per episode

The toy track's agent made **one** decision per question. Here it makes
**three**: at each turn it may SEARCH (costs 0.1, paid at once) or SKIP, and
then it answers. The answer's quality is the final reward:

| | 0 searches | 1 | 2 | 3 |
|---|---|---|---|---|
| HARD | −1.0 | 0.0 | 1.0 | 1.0 |
| EASY | 1.0 | 0.9 | 0.8 | 0.7 |

The best policy searches twice on HARD questions and never on EASY ones
(J = 0.90). Three decisions now share one outcome, so each needs its own share
of the credit. That is what the **critic** V(s) and **GAE** are for. It is the
only genuinely new piece since Surrogates.

```
for iteration:
    batch = rollout(policy, critic)            θ_old: 16 episodes x 3 turns, (episodes, turns) tensors
    advantages, returns = compute_gae(batch)   stage 1: one advantage per step
    flatten to 48 samples                      after GAE, time no longer matters
    for epoch in range(10):
        for minibatch of 16:                   stage 4
            loss = policy_loss                 stage 2: your Surrogates clip, one sample per step
                 + 0.5 * value_loss            stage 3: the critic chases the returns
                 - 0.01 * entropy_bonus        stage 3: keep exploring
            zero_grad, backward, step
```

After GAE, every step is just a sample: a state, an action, an old log-prob and
an advantage. From there the loop is Surrogates', with minibatches.

## Why `states`, not `qtype`

On the toy track the code indexed the policy with `qtype`, because with one
decision per question the state s_t **is** the question type. Here the right
action depends on more than that:

| Situation | Best action |
|---|---|
| HARD, turn 1, 0 searches so far | SEARCH: it still needs two |
| HARD, turn 2, 1 search so far | SEARCH: one more makes two |
| HARD, turn 2, 2 searches so far | SKIP: a third costs 0.1 and does not help |
| EASY, any turn | SKIP |

The first three rows are all HARD, and the answer changes. A policy that saw
only `qtype` could not tell them apart: it would have to search at the same
rate on every turn, and could never learn "search twice, then stop".

So the state is **(question type, turn, searches so far)**, packed into one id
by `env.state_id`, 6 per question type and 12 in all:

| | turn 0 | turn 1 | turn 2 |
|---|---|---|---|
| HARD (0 / 1 / 2 searches so far) | 0 | 1 / 2 | 3 / 4 / 5 |
| EASY (0 / 1 / 2 searches so far) | 6 | 7 / 8 | 9 / 10 / 11 |

The lookup works the same way as before; only what picks the row has changed:

```python
policy.dist(qtype)     # VPG / TRPO / Surrogates:  2 rows, the state is the question type
policy.dist(states)    # SimplePPO:               12 rows, the state is (qtype, turn, searches)
```

So the code uses the generic RL name, `states`. The question type is still one
ingredient of it, and `batch["qtype"]` keeps it per episode for reporting.

It is also why the critic is useful. V(HARD, turn 2, 2 searches) is high,
because a good answer is coming; V(HARD, turn 0, 0 searches) is lower. A critic
that saw only `qtype` could not tell how an episode is going partway through,
and that is exactly what GAE's surprises need.

## The stages

| Stage | You build | Paper | Question it answers |
|---|---|---|---|
| 1 | `compute_gae` | eq. 11–12 | Which of three decisions earned a reward that arrives at the end? |
| 2 | `policy_loss` | eq. 7 | What changes in your Surrogates clip when a sample has a state instead of a question type? |
| 3 | `value_loss`, `entropy_bonus` | eq. 9 | What is the critic trained to predict? Why is a bonus subtracted? |
| 4 | `ppo_update` (the minibatch slice and eq. 9) | Algorithm 1 | How do epochs and minibatches reuse one batch? |
| 5 | none | — | Does your PPO learn "HARD: search twice, EASY: never"? |

Finish `VPG/`, `TRPO/` and `Surrogates/` first. Stage 2 is your Surrogates
`clip_loss`, unchanged except for its argument names.

## How to start

1. Read `./scripts/run_simple_ppo.sh steps`: one episode through every piece, with the numbers.
2. Fill the lines marked `TODO stage 1` in `from_scratch/simple_ppo.py`.
3. Try them: `python SimplePPO/from_scratch/simple_ppo.py` prints your value next to the expected one.
4. Check them: `./scripts/run_simple_ppo.sh check` grades the stages in order, with a hint for the likely mistake.
5. When all pass: `./scripts/run_simple_ppo.sh diff` proves your walkthrough matches the reference, and
   `RL_IMPL=scratch ./scripts/run_simple_ppo.sh run` runs the comparison with your code.

## What each piece buys (`./scripts/run_simple_ppo.sh run`, about a minute)

The same loop with one piece removed at a time, 20 seeds, 60 iterations of 16
episodes. Start J 0.388, best 0.90:

| Run | What changes | J@10 | J@30 | J@60 |
|---|---|---|---|---|
| **PPO** | critic + GAE, clip, 10 epochs | 0.734 | 0.821 | **0.854** |
| 1 epoch | each batch used once | 0.473 | 0.613 | 0.723 |
| no critic | episode reward − batch mean | 0.712 | 0.811 | 0.845 |
| no clip | L^CPI instead of L^CLIP | 0.755 | 0.814 | 0.858 |

An honest reading:

- **Reusing each batch is the big win** (0.854 vs 0.723). That is what PPO is for.
- **The critic helps a little here** (+0.008). With three turns and a visible
  question type, the batch mean is already a decent baseline. The critic's
  per-step credit pays off more as episodes grow. Its small gain is also why
  GRPO can drop the critic for LLMs, where a response gets one reward.
- **The clip doesn't help here.** At 10 epochs the unclipped ratio doesn't run
  far enough to hurt. `Surrogates/` shows where it matters: at 50 epochs,
  L^CPI gets 10 of 20 runs stuck and the clip gets none.

## From here to `PPO/`

| | `SimplePPO/` | `PPO/` (LLM) |
|---|---|---|
| One step | one turn: SEARCH or SKIP | one token, from a vocabulary |
| log π | `policy.dist(states).log_prob(actions)` | `logprobs_from_logits(logits, tokens)` |
| Episode length | always 3 | varies: padding, and tool outputs inside, so `response_mask` |
| A mean | `.mean()` | `masked_mean` over the mask |
| GAE | plain | the same, but the mask carries values across masked tokens |
| Everything else | — | identical: the clip, the critic, eq. 9, the loop |

So `PPO/`'s Part 1 is this module in LLM clothing. It adds no new algorithm,
only the token bookkeeping.

## At the end, you should be able to answer

- In stage 1, with λ = 1 the advantage of turn 0 is the episode's total reward
  minus V(s₀). Why is that the toy track's `reward − baseline` in disguise?
- After GAE the batch is flattened, and turns from different episodes are mixed
  in one minibatch. Why is that allowed?
- The critic's win on this toy is small. What would make it bigger: longer
  episodes, a hidden question type, or less noise? Why?
- `1 epoch` loses the most. Which line of `ppo_update` would you change to get
  it, and what does the clip have to do with the answer?
