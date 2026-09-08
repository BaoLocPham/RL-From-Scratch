# Build PPO from scratch

Do not open `../common.py` first. Work from the shapes, the papers, and the
grader outputs; otherwise this becomes transcription rather than reconstruction.

| Stage | Function | Core idea |
|---|---|---|
| 1 | `discounted_returns` | Reward-to-go stops at episode boundaries |
| 2 | `generalized_advantage_estimate` | Bias/variance-controlled temporal credit |
| 3 | `normalize_advantage` | Statistics ignore padded positions |
| 4 | `clipped_policy_loss` | Bound the incentive from a changed policy ratio |
| 5 | `clipped_value_loss` | Bound one critic regression update |
| 6 | `categorical_entropy`, `ppo_loss` | Preserve exploration and compose the objective |

Run `python PPO/from_scratch/check.py` from the repository root. The policy and
value clipping objective follows equations (7-9) in
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
Stage 2 follows equations (11-12) in
[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

A production RLHF PPO pipeline also penalizes drift from a frozen SFT reference.
In TRL v0.21.0 that k1/k3 KL penalty is folded into the reward *before* GAE,
rather than added to the final `ppo_loss`; see
[`ppo_trainer.py` lines 511-513](https://github.com/huggingface/trl/blob/46d09bd2408f17605409fb3ee8ba12705add7faa/trl/trainer/ppo_trainer.py#L511-L513).
It is not implemented here because this compact module otherwise needs no
second frozen model.

At the end, you should be able to explain why terminal masks affect both
bootstrapping and recursive credit, why the policy clip changes behavior with
the sign of the advantage, why PPO retains the worse value error, and why
entropy is subtracted from a minimized loss.
