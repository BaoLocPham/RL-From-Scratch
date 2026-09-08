# RL algorithms, from scratch

Small, CPU-only implementations of PPO, DPO, GRPO, and the learning ideas in
[Agent0](https://arxiv.org/abs/2511.16043) and
[Tool-R0](https://github.com/emrecanacikgoz/Tool-R0). The production agent
systems use distributed trainers, inference servers, sandboxes, and model
judges. This repo keeps only the mechanisms that are useful to understand on
paper:

- `PPO/`: returns, GAE, clipped policy/value losses, and entropy. Its naming
  follows TRL v0.21; the [current TRL trainer list](https://huggingface.co/docs/trl/index)
  exposes `PPOTrainer` as experimental, while GRPO and RLOO are stable online
  trainers.
- `DPO/`: reference-corrected preference optimization from chosen/rejected data.
- `GRPO/`: group-relative advantages, the PPO clipped surrogate, and TRL's k3
  reference-policy KL penalty.
- `Agent0/`: curriculum reward and ADPO's difficulty-aware reweighting.
- `ToolR0/`: tool-call scoring, generator rewards, and curriculum selection.

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

python ToolR0/steps_toolr0.py
python ToolR0/run_toolr0.py
python ToolR0/from_scratch/check.py
```

Set `RL_IMPL=scratch` to run any walkthrough or dissector against your exercise
implementation after its grader passes.

Reference code used to design the exercises lives in the local clones
`agent0-repro-local/src/Agent0` and `tool-r0-repro-local/src/Tool-R0`. Live model
sampling and LLM judging are deliberately represented by deterministic values
or callbacks supplied by the caller.
