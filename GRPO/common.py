"""The core pieces of Group Relative Policy Optimization (GRPO)."""

import torch


def group_relative_advantage(rewards, group_index, epsilon=1e-6):
    """Return one standardized advantage per completion.

    ``rewards`` and ``group_index`` are both ``(N,)``. Statistics are computed
    independently for every prompt group. PyTorch's sample standard deviation
    is used for groups with at least two rows; a singleton gets mean 0 and
    standard deviation 1, matching the special case in Agent0's GRPO code.
    This repo keeps Agent0's ``1e-6`` smoothing; Hugging Face TRL uses ``1e-4``.
    """
    rewards = torch.as_tensor(rewards)
    group_index = torch.as_tensor(group_index, device=rewards.device)
    advantages = torch.empty_like(rewards)
    for group in torch.unique(group_index):
        here = group_index == group
        group_rewards = rewards[here]
        if group_rewards.numel() == 1:
            mean_grouped_rewards = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
            std_rewards = torch.ones((), dtype=rewards.dtype, device=rewards.device)
        else:
            mean_grouped_rewards = group_rewards.mean()
            std_rewards = group_rewards.std(unbiased=True)
        advantages[here] = (group_rewards - mean_grouped_rewards) / (std_rewards + epsilon)
    return advantages


def kl_penalty_k3(ref_logp, logp):
    """Return TRL's non-negative per-token k3 KL estimator.

    This is the estimator used by ``trl.GRPOTrainer`` and by DeepSeekMath's
    GRPO objective (equation 3). It estimates policy-to-reference KL from tokens
    sampled by the policy while avoiding the negative single-sample values of
    the naive log-ratio estimator.
    """
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1.0


def clipped_surrogate_loss(old_logp, logp, advantage, mask, eps=0.2,
                           ref_logp=None, beta=0.0):
    """PPO clipped loss over valid completion tokens.

    Log probabilities and ``mask`` are ``(N,T)``. ``advantage`` is ``(N,)``:
    each completion's scalar is deliberately shared by all of its tokens.
    When a frozen reference log-probability and nonzero ``beta`` are supplied,
    the masked mean also includes TRL's per-token k3 KL penalty. Omitting either
    keeps the original clipped-only objective exactly.
    """
    coef_1 = torch.exp(logp - old_logp)                        # (N,T)
    adv = advantage.unsqueeze(-1) if advantage.ndim == 1 else advantage
    coef_2 = coef_1.clamp(1.0 - eps, 1.0 + eps)                # (N,T)
    per_token_loss1 = -adv * coef_1                            # (N,T)
    per_token_loss2 = -adv * coef_2                            # (N,T)
    per_token_loss = torch.maximum(per_token_loss1, per_token_loss2)
    if ref_logp is not None and beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_penalty_k3(ref_logp, logp)
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1)
