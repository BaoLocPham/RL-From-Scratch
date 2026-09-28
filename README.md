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

- `VPG/`, `TRPO/`, `Comparison/`: the PPO paper's Section 2 on a toy
  tool-calling bandit. Vanilla policy gradient's flaw (the gradient is only
  valid where the batch was collected), TRPO's fix and its own flaw (a KL constraint
  that sits outside the loss), and a side-by-side run of VPG, TRPO
  and PPO on the same rollout budget. `VPG/from_scratch/` and
  `TRPO/from_scratch/` are staged exercises; the comparison is a demo only.
  The terms the logs use are in `VPG/README.md` and `TRPO/README.md`.
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

Order matters: **VPG before TRPO**, since TRPO's exercise reuses VPG's toy and
update loop, and **PPO before GRPO**, since `GRPO/from_scratch/grpo.py` imports
your `agg_loss`, `compute_policy_loss` and `kl_penalty` from `PPO/from_scratch/`.

## Running it

```bash
pip install -r requirements.txt

./scripts/run_vpg.sh run        # VPG: new rollouts every update vs reusing one batch
./scripts/run_vpg.sh steps      # VPG walkthrough; check / scratch / diff work as for PPO
./scripts/run_trpo.sh run       # TRPO vs VPG: safe reuse, KL constraint outside the loss
./scripts/run_trpo.sh steps     # TRPO walkthrough; check / scratch / diff work as for PPO
./scripts/run_compare.sh run    # VPG vs TRPO vs PPO, one table + figure
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
python TRPO/steps_trpo.py    python TRPO/run_trpo.py    python TRPO/from_scratch/check.py
python Comparison/run_compare.py
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

## From this toy to an LLM

The VPG and TRPO toys use a 2×2 table of logits as the policy. A real policy is
a neural network or a transformer, but the RL math does not change, because it
all starts from the logits:

| | State s_t | Action a_t | Logits come from |
|---|---|---|---|
| The VPG / TRPO toy | question type | answer / tool | `policy.logits[qtype]`, a table lookup |
| An MLP (e.g. CartPole) | observation vector | a move | `net(obs)` |
| An LLM | prompt + every token so far | the next token | `model(input_ids).logits` |

After the logits the code is the same: `log_softmax`, pick out the log-prob of
the action taken, then the ratio, clip, KL and loss. For an LLM, as in verl's
`logprobs_from_logits`:

```python
logits = model(input_ids, attention_mask).logits      # (B, T, V): one pass scores every position
logits = logits[:, prompt_len - 1:-1] / temperature   # the logits at t predict token t+1
log_prob = torch.log_softmax(logits, -1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)  # (B, R)
```

That `(batch, response_length)` tensor, with `response_mask`, is what `PPO/`,
`GRPO/` and `Agent0/` take. The toy is the case R = 1, V = 2.

What gets harder is the engineering, not the math:

- **Every token is an action.** A 500-token response is 500 steps, and the
  state is the whole prefix. KL between whole sequences is intractable, so it is
  taken per token and averaged over the mask.
- **The exact KL does not fit in memory.** `TRPO/trpo.py`'s `mean_kl` sums over
  both actions. Over a ~152k vocabulary, one fp32 logits tensor for 8 × 4,096
  tokens is ~20 GB, and the sum needs the reference model's as well. verl keeps
  only the sampled token's log-prob and *estimates* the KL (k1 / k2 / k3 in
  `PPO/common.py`'s `kl_penalty`); its `"full"` mode raises
  `NotImplementedError`, as ours does.
- **Several forward passes.** vLLM samples the rollouts, the trainer recomputes
  `old_log_prob` (vLLM's numbers differ slightly), a frozen reference model gives
  `ref_log_prob`, and PPO also runs a critic. GRPO drops the critic mainly to
  save memory.
- **Sampling settings must match.** Rollouts sampled at temperature T need their
  log-probs computed from `logits / T`, or the ratio is not 1 at θ_old.
- **Not every token is the policy's.** Tool outputs sit inside an agentic
  response, but the model did not choose them; `response_mask` keeps them out
  of the ratio, the KL and the loss.
- **The paper's TRPO does not scale.** Its conjugate-gradient step needs
  Fisher-vector products over billions of parameters. PPO's clip needs only
  first-order SGD, which is why LLM training uses PPO and GRPO.

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
