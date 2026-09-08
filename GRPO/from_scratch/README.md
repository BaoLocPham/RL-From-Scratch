# Build GRPO from scratch

Do not open `../common.py` first. The useful work is reconstructing the two
operations from their behavior, then using `check.py` as feedback. Reading the
reference turns the exercise into transcription.

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `group_relative_advantage` | How can the sampled group replace a value-function baseline? |
| 2 | `clipped_surrogate_loss` | How does PPO prevent one policy-ratio update from moving too far? |
| 3 | `kl_penalty_k3` | How can one sampled token estimate KL without negative values? |
| 4 | extended `clipped_surrogate_loss` | How does GRPO stay near a frozen reference policy? |

Run `python GRPO/from_scratch/check.py` from the repository root. The grader
stops at the first incomplete stage.

The formulas follow Agent0's GRPO implementation in
`Agent0/executor_train/verl_tool/trainer/ppo/core_algos.py:225-257`: prompt-local
standardization with an epsilon, then one completion advantage copied across
its response mask. The policy objective is the PPO clipped surrogate extended
by ADPO at `core_algos.py:327-372`. This repo keeps Agent0's `1e-6` smoothing
constant; [Hugging Face TRL's GRPOTrainer](https://github.com/huggingface/trl/blob/8056842449d2abbc08abd2628c5608071f433bfc/trl/trainer/grpo_trainer.py#L2804-L2806)
uses `1e-4` for the same denominator.

Stages 3-4 implement the reference-policy term in equation (3) of
[DeepSeekMath](https://arxiv.org/abs/2402.03300) using the non-negative `k3`
estimator described in [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
and used by
[TRL's GRPOTrainer](https://github.com/huggingface/trl/blob/8056842449d2abbc08abd2628c5608071f433bfc/trl/trainer/grpo_trainer.py#L3183-L3236).

At the end, you should be able to answer:

- Why does normalizing all prompts together reintroduce prompt-difficulty bias?
- Why is there one advantage per completion rather than per token?
- Why does clipping behave differently when the advantage is negative?
- What signal remains when every reward in a group is identical?
- Why must a single-sample KL estimator be non-negative? If a naive estimator
  is negative for one token, why is that sampling noise rather than negative KL?
