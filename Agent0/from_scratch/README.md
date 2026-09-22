# Build Agent0's reward from scratch

Do not open `../common.py` first. Work from the docstrings in `agent0.py` and
from the grader's messages; reading the reference turns this into transcription.

This exercise is scaffolded more heavily than the PPO and GRPO ones. The
control flow, the loops and the variable names are already written — the TODOs
sit on individual expressions. The bookkeeping around an idea is not the idea,
and here only the idea is left blank.

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `extract_boxed` | Why walk braces instead of matching a regex? |
| 2 | `cluster_answers` | Why try equivalence in both directions? Why cheap checks first? |
| 3 | `self_consistency_score` | Why do Step 3 and Step 4 divide by different denominators? |
| 4 | `gate_against_claim` | Why is a confident *wrong* majority worth zero? Why is the guard strict? |
| 5 | `tool_reward` | Why count and cap rather than set a flag? |
| 6 | `curriculum_reward` | Why does the reward peak in the middle, and why is malformed `-1`? |

Run `python Agent0/from_scratch/check.py` from the repository root. The grader
stops at the first incomplete stage. Nothing in this file needs torch.

Once it passes, `RL_IMPL=scratch python Agent0/steps_curriculum.py` runs the
walkthrough against your implementation instead of the reference. All six
functions live on the curriculum side; `steps_executor.py` is ADPO, which is
reference material and does not switch.

## What this covers, and what it does not

Agent0 trains **two** models with **two different algorithms**, and it is worth
fixing that shape in your head before starting:

| Agent | Job | Algorithm |
|---|---|---|
| Curriculum Agent | writes new questions | **GRPO** (Step 3) |
| Executor Agent, the solver | answers them, with a Python tool | **ADPO** (Step 5) |

### What "Step 3" and "Step 4" mean

Agent0's pipeline is numbered 1–5. Steps 3 and 4 both involve the Curriculum
Agent, which makes them easy to conflate — but only one of them trains it:

| | Step 3 | Step 4 |
|---|---|---|
| Purpose | **Train** the Curriculum Agent | **Curate** a dataset |
| Does anything learn? | Yes — GRPO updates the proposer | No. Zero gradients |
| Who is frozen | the Executor | **both** agents |
| Attempts per question | 10, **with** the Python sandbox | 9, plain generation, **no** tool |
| Gated against the claimed answer? | Yes — mismatch scores 0 | No — the majority *becomes* the label |
| Output | a weight update | `train.parquet`, plus a difficulty per row |

Step 3 is the proposer learning to aim; Step 4 is the trained proposer building
the solver's homework. Step 5 is then the solver training on it, with ADPO.

These six functions are Step 3's whole reward and Step 4's whole label — the
part of Agent0 that is genuinely new. They feed both halves. The reward you
build in stage 6 is what GRPO turns into a Curriculum Agent update; the label
that `self_consistency_score` produces in stage 3 is what later becomes ADPO's
per-question difficulty, and so decides how hard the solver is pushed.

The algorithms themselves are not exercises here. Plain GRPO is one directory
over, in `GRPO/`, and is used by Agent0 completely unmodified. ADPO is in
`../common.py` as reference only, along with the `[0.3, 0.8]` band filter and
the Executor's `+1`/`-1` reward. Read `adpo_advantage` next to `GRPO/common.py`'s
`compute_grpo_outcome_advantage`: they are the same function plus one multiply,
and that multiply is the paper's contribution.

## Where the code came from

Traced from `aiming-lab/Agent0`, not from the paper:

| Stage | Upstream |
|---|---|
| 1 | `mathruler.extract_boxed_content`, and `torl_math.py`'s `boxed_pattern` |
| 2, 3, 4 | `curriculum_train/vllm_service_init/start_vllm_server_tool.py` — `consolidate_and_grade` |
| 3 (`"valid"` mode) | `curriculum_train/question_evaluate/evaluate.py` |
| 5, 6 | `curriculum_train/examples/reward_function/curriculum_reward.py` |

## Two upstream bugs this repo does not reproduce

1. **The golden answer is truncated to one character.** `compute_score` does
   `answers = extract_boxed_content(predicts[i])` and then `answers[-1]`. That
   function returns a **string**, not a list, so `\boxed{40}` yields `"0"`.
   The gate then compares the solver's majority against `"0"` and fails for any
   multi-character answer. Every other call site uses the value as a string;
   this one line treats it like the `re.findall` list beside it. We pair the
   full string.
2. **The tool bonus counts the wrong text — deliberately, it turns out.**
   `calculate_tool_reward(predicts[i])` counts fences in the *Curriculum
   Agent's own generation*, not in the solver's transcript. That is not a bug,
   but it is easy to read backwards, and it changes what the term incentivizes:
   the proposer checking its own answer with code, not the solver using a tool.
   Stage 5's grader accepts either string — only the docstrings care.

## Four places where the code disagrees with how it gets described

Worth knowing before you start, because the grader enforces the code:

1. **The diversity penalty is continuous.** Usually written up as
   `0.3 if is_duplicate else 0.0`. The real `compute_score` subtracts
   `cluster_share_per_problem(...)[i]` — this question's BLEU cluster size
   divided by the batch size. A question in a cluster of 40 out of 100 loses
   0.40, more than the tent reward's entire 0.5 maximum.
2. **The tool reward is counted and capped.** Usually written up as a flat
   `0.05` bonus. The real `calculate_tool_reward` is
   `min(count("```output"), 4) * 0.05` — worth up to 0.20.
3. **Malformed generations score `-1`, not `0`.** The base is
   `min(s, 1-s) if question else -1`. Failing to produce a parseable question is
   punished below a perfectly solved one.
4. **Self-consistency has two different denominators.** Step 3 divides by every
   candidate sampled; Step 4 divides by the ones that produced an answer. Same
   name, same repo, different numbers — stage 3 is built around this.

## At the end, you should be able to answer

- The Curriculum Agent writes both the question and its own answer key, and
  nobody checks that key. Why is the reward still not trivially farmable?
- Rows 3 and 4 of the worked example ("coherently wrong" and "incoherent") both
  score 0, the same as a perfectly solved question. Why is collapsing those
  three cases together the right call rather than a bug?
- Why does the gate use `score > 0.1` rather than `score >= 0.1`, and what
  would change at `n = 20` candidates instead of 10?
- Step 4 drops the gate entirely and lets the plurality *become* the label.
  What does that cost, and why is it acceptable there but not in Step 3?
- If the equivalence check were replaced with exact string matching, which of
  the six functions would still look correct in testing, and what would the
  training run actually be optimizing?
