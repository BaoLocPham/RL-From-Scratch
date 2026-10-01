# Build Agent0 from scratch, on two tables of logits

Do not open `agent0.py` (the reference) before you finish the exercise. Work
from the docstrings in `from_scratch/agent0.py` and the grader's messages.

[Agent0](https://arxiv.org/abs/2511.16043) trains a model **with no data at
all**. Two agents make the data for each other:

| Agent | Paper's name | Job | Trained with |
|---|---|---|---|
| **Curriculum Agent** | π_θ | writes questions | GRPO |
| **Executor** | π_ϕ | answers them | ADPO, the paper's variant of GRPO |

This module builds the paper's equations on a toy where each agent is a small
table of logits. You can watch every logit move, and nothing else gets in the
way.

## The loop

The paper repeats three steps T = 3 times:

```
Step 3  train the Curriculum Agent pi_theta (GRPO); the Executor is frozen
        pi_theta writes questions; pi_phi answers each one k = 10 times;
        pi_theta's reward R_C pays for questions pi_phi is unsure about       Eq. 2-6
Step 4  curate a dataset; both frozen
        keep the questions pi_phi agrees with itself on 30-80% of the time;    Eq. 7
        its majority answer y~ becomes the label -- nobody knows the true answer
Step 5  train the Executor pi_phi (ADPO) on that dataset; pi_theta is frozen
        reward 1 for agreeing with y~, 0 otherwise; GRPO's advantage, scaled
        by how far y~ can be trusted; a clip that is wider on hard questions    Eq. 8
```

Steps 1 and 2 of the paper are setup, so the numbering starts at 3.

## The toy (`agent0_env.py`, given code)

- **Questions:** 5 difficulty levels × 3 variants = 15 question ids. Ids 0–2
  are level 1, …, ids 12–14 are level 5. There is also a 16th output,
  **MALFORMED**: a generation with no usable question in it. A question is
  only its id.
- **The Executor is one number,** its skill logit, starting at 4.0. On a
  level-L question it picks one of 10 answers (answer 0 is right, 1–9 are
  wrong) with logits `[skill − L, 0, 0, 0, 0, 0, 0, 0, 0, 0]`:

  ```
  p(right | L) = e^(skill − L) / (e^(skill − L) + 9)
  ```

  | Level | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | p(right) at skill 4.0 | 0.69 | 0.45 | 0.23 | 0.10 | 0.04 |

  One skill serves every question, so learning level 3 also helps level 4.
- **The Curriculum Agent is 16 logits,** one per output, all starting at 0.
  Writing a question means sampling one id.
- **Tool calls:** every attempt at a level-L question calls the code tool L
  times. Harder questions need more.

## The equations

| | Equation | What it does |
|---|---|---|
| Eq. 6 | p̂(x) = (1/k) Σᵢ 𝕀(oᵢ = ỹ), ỹ = argmax_y Σᵢ 𝕀(oᵢ = y) | how often the Executor's k = 10 answers agree with their majority ỹ |
| Eq. 2 | R_unc(x) = 1 − 2·\|p̂(x) − 0.5\| | 1 when p̂ = 0.5, 0 when p̂ = 0 or 1: pays for the Executor's frontier |
| Eq. 3 | R_tool(x) = γ·min(N_tool, C), C = 4 | pays for questions that need the tool, up to a cap |
| Eq. 4 | R_rep(xᵢ) = λ_rep·\|C_k\|/B | the share of the batch that wrote the same question |
| Eq. 5 | R_C(xᵢ) = R_format·max(0, λ_unc·R_unc + λ_tool·R_tool − R_rep), λ_tool = 0.6 | the Curriculum Agent's reward; malformed earns 0 |
| Eq. 7 | keep x if p̂(x) ∈ [0.3, 0.8] | the dataset: neither too easy nor too unreliable |
| — | Rᵢ = 𝕀(oᵢ = ỹ) | the Executor's reward: agree with the label, 1 or 0 |
| — | Ãᵢ = Âᵢ·s(x), s(x) = f(p̂(x)) increasing | ADPO: trust a low-agreement label less |
| Eq. 8 | −(1/G) Σᵢ min(rᵢ·Ãᵢ, clip(rᵢ, 1 − ε_low, 1 + ε_high(x))·Ãᵢ) | the update; ε_high(x) decreasing in p̂ |

**What the paper leaves open.** It only says f is increasing and ε_high
decreasing, and it doesn't fix γ, λ_unc or λ_rep. We use:
- s(x) = 0.5 + 0.5·clamp((p̂ − 0.3)/0.5, 0, 1);
- ε_high(x) = 0.2 + 0.1·clamp((0.8 − p̂)/0.5, 0, 1);
- γ = 0.05, λ_unc = λ_rep = 1.

The paper clusters questions by BLEU similarity for Eq. 4. A question here is
only an id, so a cluster is just the copies of one question in the batch.

## The stages

| Stage | You build | Equation |
|---|---|---|
| 1 | `self_consistency` | Eq. 6 |
| 2 | `uncertainty_reward`, `tool_reward`, `repetition_penalty`, `curriculum_reward` | Eq. 2–5 |
| 3 | `keep`, `executor_reward` | Eq. 7, Rᵢ |
| 4 | `group_advantage`, `adpo_scale`, `adpo_eps_high`, `clipped_loss` | Âᵢ, Ãᵢ, Eq. 8 |
| 5 | no code: the loop runs on your pieces | — |

Each TODO is one term of one equation, written as a comment on the line you
fill. The loops are given: `train_curriculum` (Step 3), `curate` (Step 4),
`train_executor` (Step 5) and `agent0` (all three, T times). Read them; they
are the algorithm.

## How to start

1. Read `./scripts/run_agent0.sh steps`. It follows every equation with the
   numbers substituted in, and prints the logits as they move.
2. Fill the lines marked `TODO stage 1` in `from_scratch/agent0.py`.
3. Try them: `python Agent0/from_scratch/agent0.py` prints your value next to
   the expected one.
4. Check them: `./scripts/run_agent0.sh check` grades the stages in order, with
   a hint for the likely mistake.
5. When all pass:
   - `./scripts/run_agent0.sh diff` proves your walkthrough matches the reference;
   - `RL_IMPL=scratch ./scripts/run_agent0.sh run` runs the demo with your code.

## What the loop does (`./scripts/run_agent0.sh run`, about 20 seconds)

One run, seed 0. Each iteration is 30 GRPO steps on π_θ, then curating 64
questions, then 20 ADPO steps on π_ϕ:

| Iteration | π_θ writes level 1..5 | Kept | Labels right | Skill | π_ϕ right on level 1..5 |
|---|---|---|---|---|---|
| 1 | 0.07 **0.58** 0.10 0.14 0.10 | 52 | 41 | 4.00 → 4.99 | 0.86 0.69 0.45 0.23 0.10 |
| 2 | 0.03 0.15 **0.43** 0.31 0.07 | 45 | 39 | 4.99 → 5.92 | 0.94 0.85 0.67 0.43 0.22 |
| 3 | 0.01 0.03 0.17 **0.75** 0.04 | 58 | 56 | 5.92 → 6.98 | 0.98 0.94 0.86 0.69 0.45 |

The Curriculum Agent's favourite level climbs 2 → 3 → 4. Each time, it is the
level where the Executor is right about half the time, which is where R_unc
pays. Nobody told it which level to write. The Executor trains on those
questions and gets better at every level, so the next iteration has to move up.

Then 10 seeds, changing one ingredient at a time:

| Run | Skill after 1, 2, 3 | Favourite level | Wrong labels kept | Top question |
|---|---|---|---|---|
| **Agent0** | 4.83 5.73 6.68 | 2.0 2.8 3.6 | 28.1 of 156.9 | 0.22 |
| Executor with plain GRPO | 5.15 6.41 7.75 | 2.0 3.1 4.0 | 24.9 of 158.2 | 0.24 |
| plain GRPO, step × 0.67 | 4.79 5.65 6.58 | 2.0 2.8 3.4 | 28.8 of 157.7 | 0.21 |
| no R_rep | 4.98 6.00 6.98 | 2.0 2.1 3.8 | 15.5 of 165.4 | 0.65 |

"Top question" is π_θ's largest probability on any one of its 16 outputs;
uniform is 1/16 = 0.06.

An honest reading:

- **The two agents push each other:** the favourite level climbs about one
  level per iteration as the skill grows.
- **The labels are sometimes wrong, and in one place.** About 1 kept label in
  6 is wrong, and the wrong ones sit at the bottom of the band, p̂ = 0.3–0.4.
  Low agreement means an unreliable majority: that is ADPO's premise. And
  because the Executor is rewarded for agreeing with the label, a wrong label
  pushes its skill *down* (walkthrough, step 5).
- **ADPO learns slower than plain GRPO here.** s(x) ≤ 1 shrinks every step,
  including those from right labels with a low p̂.
- **ADPO edges out GRPO at the same average step** (6.68 vs 6.58), because it
  spends that step where the labels are more often right. The gain is small: on
  this toy a wrong label only lowers the skill through the few right answers in
  its group.
- **Without R_rep, π_θ writes the same question again and again** (0.65 on a
  single output). With R_rep it spreads across the three questions of a level.
  The skill doesn't suffer, because one level's questions are interchangeable
  here. R_rep exists to keep a real dataset varied.

## Terms

- **π_θ, π_ϕ:** the Curriculum Agent and the Executor.
- **p̂ (p_hat):** how often the Executor's k answers agree with their
  majority. Agreement, not correctness.
- **ỹ (y_tilde):** the majority answer, which becomes the label in Step 4.
- **R_C:** the Curriculum Agent's reward, Eq. 5.
- **s(x):** ADPO's trust in a question's label, which scales its advantage.
- **ε_high(x):** ADPO's per-question upper clip bound.

## At the end, you should be able to answer

- R_unc peaks at p̂ = 0.5, not at p̂ = 0. Why would rewarding questions the
  Executor always gets wrong be a bad curriculum?
- p̂ measures agreement, not correctness. On a level-5 question the Executor is
  mostly guessing, yet R_unc can still be 0.4–0.6. Why, and which step catches it?
- The Executor is rewarded for agreeing with a label nobody checked. Why does
  its true accuracy still go up?
- A wrong label lowers the skill only through the right answers in its group.
  Why do wrong answers not move it at all, in this model?
- Without R_rep, π_θ writes one question over and over, and the toy doesn't
  mind. Why would a real Executor mind?
