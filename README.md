# RL algorithms, from scratch

Small, CPU-only implementations of PPO, DPO, GRPO, and the learning ideas in
[Agent0](https://arxiv.org/abs/2511.16043). The production agent systems use
distributed trainers, inference servers, sandboxes, and model judges. This repo
keeps only the mechanisms that are useful to understand on paper:

- `PPO/`: returns, GAE, clipped policy/value losses, and entropy. Its naming
  follows TRL v0.21; the [current TRL trainer list](https://huggingface.co/docs/trl/index)
  exposes `PPOTrainer` as experimental, while GRPO and RLOO are stable online
  trainers.
- `DPO/`: reference-corrected preference optimization from chosen/rejected data.
- `GRPO/`: group-relative advantages, the PPO clipped surrogate, and TRL's k3
  reference-policy KL penalty.
- `Agent0/`: self-consistency scoring, the curriculum reward, the difficulty
  band filter, and ADPO's difficulty-aware advantage scaling and clip range.

Each directory has a reference implementation (`common.py`), a literal
walkthrough (`steps_*.py`), a runnable demonstration (`run_*.py`), and a staged
exercise under `from_scratch/`. Start in that order, but solve the exercise
without opening `common.py`.

```bash
python PPO/steps_ppo.py
python PPO/run_ppo.py
python PPO/from_scratch/check.py

python DPO/steps_dpo.py
python DPO/run_dpo.py
python DPO/from_scratch/check.py

python GRPO/steps_grpo.py
python GRPO/run_grpo.py
python GRPO/from_scratch/check.py

python Agent0/steps_agent0.py
python Agent0/run_agent0.py
python Agent0/from_scratch/check.py
```

Install the one dependency first:

```bash
pip install -r requirements.txt
```

`Agent0/from_scratch/agent0.py` is itself plain Python — the reward and the vote
need no tensors, exactly as upstream — but its grader imports `common.py`, which
carries ADPO.

Set `RL_IMPL=scratch` to run any walkthrough or dissector against your exercise
implementation after its grader passes. `Agent0/` has a wrapper for all of it:

```bash
./scripts/run_agent0.sh            # list the commands
./scripts/run_agent0.sh check      # grade your from_scratch implementation
./scripts/run_agent0.sh diff       # then prove it matches the reference
```

The exercises were designed against the real code in
[aiming-lab/Agent0](https://github.com/aiming-lab/Agent0), which each module
cites by path in its docstrings. Where that code and its write-ups disagree,
this repo follows the code and says so in a comment. Live model sampling, vLLM,
the code sandbox and sympy-based answer grading are deliberately represented by
deterministic values or by callbacks supplied by the caller.
