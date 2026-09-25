# RL algorithms, from scratch

Small, CPU-only implementations of the algorithms
[verl](https://github.com/volcengine/verl) trains LLMs with, plus the learning
ideas in [Agent0](https://arxiv.org/abs/2511.16043), which is built on verl.

Names, argument orders and return tuples follow verl's
[`trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py)
and `utils/torch_functional.py`, so what you write here transfers to a real verl
trainer unchanged. The production systems add distributed workers, vLLM,
sandboxes and Ray; this repo keeps only the mechanisms worth understanding on
paper.

- `VPG/`: vanilla policy gradient, the PPO paper's Section 2.1, on a toy
  tool-calling bandit, and its flaw: the gradient is only valid where the batch
  was collected. The demo runs it with new rollouts for every update (correct,
  but slow) beside reusing one batch (cheap, but it overshoots).
- `PPO/`: masked statistics, GAE, the four `loss_agg_mode` reductions, dual-clip
  policy loss, clipped value loss, entropy, and four KL estimators.
- `GRPO/`: group-relative outcome advantage, and the Dr.GRPO flag. Everything
  else it needs is PPO's, imported rather than copied — as in verl, where all of
  it lives in one file.
- `Agent0/`: two agents trained against each other with two different
  algorithms — a **Curriculum Agent** that writes questions, trained with plain
  **GRPO**, and an **Executor Agent** (the solver) that answers them, trained
  with **ADPO**. Covers the self-consistency scoring and curriculum reward that
  drive the first, the difficulty band filter that feeds the second, and ADPO's
  difficulty-aware advantage scaling and clip range.

Each directory has a reference implementation (`common.py`), a literal
walkthrough (`steps_*.py`), a runnable demonstration (`run_*.py`), and a staged
exercise under `from_scratch/`. Start in that order, but solve the exercise
without opening `common.py`.

Order matters: **PPO first.** `GRPO/from_scratch/grpo.py` imports your
`agg_loss`, `compute_policy_loss` and `kl_penalty` from `PPO/from_scratch/`.

### Terms in the VPG logs

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
- **true J**: the policy's real score, its average reward over many questions.
  Computed exactly from the table above, not estimated from noisy rollouts.
  At the start, HARD earns 0.6×0.2 + 0.4×1.0 = 0.52 and EASY 0.6×0.8 + 0.4×0.5
  = 0.68, so J = 0.60. The best possible (always search on HARD, always answer
  on EASY) is J = (1.0 + 0.8) / 2 = 0.90.
- **rollout**: one question, the action taken, and its noisy reward. The
  expensive part. A **batch** is 16 rollouts collected by one model, **θ_old**.
- **update**: one `optimizer.step()`, the only line that changes the model.
  Only the first update on a batch is taken at θ_old, where the batch's gradient
  is valid.
- **epoch**: one full pass over the batch. These toys use the whole batch per
  update, so 1 epoch = 1 update. Vanilla PG runs 1 epoch per batch; the VPG
  demo's shortcut runs 100 epochs on one batch.
- **iteration**: collect a fresh batch, then run all its epochs on it.

## Running it

```bash
pip install -r requirements.txt

./scripts/run_vpg.sh run        # VPG: new rollouts every update vs reusing one batch
./scripts/run_vpg.sh steps      # VPG walkthrough; check / scratch / diff work as for PPO
./scripts/run_ppo.sh            # list the commands
./scripts/run_ppo.sh check      # grade your from_scratch implementation
./scripts/run_ppo.sh steps      # the walkthrough
./scripts/run_ppo.sh run        # the demo
./scripts/run_ppo.sh diff       # prove yours matches the reference
```

`run_grpo.sh` and `run_agent0.sh` take the same commands. `Agent0` adds three
of its own, since it is the only module training two agents:

```bash
./scripts/run_agent0.sh overview     # the iteration flow, and a glossary
./scripts/run_agent0.sh trace        # one question through Step 3 and Step 4
./scripts/run_agent0.sh curriculum   # or executor, for one half at a time
```

Start with `overview` — it draws what generates what, who is frozen, and where
weights actually move, then defines every term the other scripts print.

Or call the files directly:

```bash
python VPG/steps_vpg.py      python VPG/run_vpg.py      python VPG/from_scratch/check.py
python PPO/steps_ppo.py      python PPO/run_ppo.py      python PPO/from_scratch/check.py
python GRPO/steps_grpo.py    python GRPO/run_grpo.py    python GRPO/from_scratch/check.py
python Agent0/steps_agent0.py python Agent0/run_agent0.py python Agent0/from_scratch/check.py
```

`Agent0/` has one walkthrough per agent, and `steps_agent0.py` runs both in
dependency order:

```bash
python Agent0/steps_curriculum.py   # the proposer -- writes questions, GRPO
python Agent0/steps_executor.py    # the solver   -- answers them, ADPO
```

Set `RL_IMPL=scratch` to run any walkthrough against your own implementation
once its grader passes.

## Two conventions

Everything outside `Agent0/`'s reward layer is `(batch, response_length)`, and
`response_mask` is the only thing that makes a position real. There is no
per-sequence scalar: one outcome reward for a whole response is a row that is
zero except at its last valid position. Every mean, variance and loss is taken
over the mask.

## Notes

There is no DPO module. verl ships no DPO trainer, so it sits outside what this
repo is mirroring; it was removed in the verl-alignment pass and remains in git
history.

The exercises were designed against the real code in verl and in
[aiming-lab/Agent0](https://github.com/aiming-lab/Agent0), which each module
cites by path in its docstrings. Where that code and its write-ups disagree,
this repo follows the code and says so in a comment — `Agent0/from_scratch/README.md`
lists four such places. Live model sampling, vLLM, the code sandbox and
sympy-based answer grading are deliberately represented by deterministic values
or by callbacks supplied by the caller.
