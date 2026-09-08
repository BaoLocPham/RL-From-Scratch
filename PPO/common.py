"""Small, explicit building blocks for Proximal Policy Optimization (PPO)."""

import torch
import torch.nn.functional as F


def discounted_returns(rewards, dones, gamma=0.99):
    """Compute reward-to-go along the last axis, resetting at terminal steps.

    ``rewards`` and ``dones`` have shape ``(..., T)``. A true ``dones[..., t]``
    means the transition at ``t`` ended the episode, so later rewards do not
    leak into that step's return.
    """
    rewards = torch.as_tensor(rewards)
    dones = torch.as_tensor(dones, device=rewards.device)
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have the same shape")
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in range(rewards.shape[-1] - 1, -1, -1):
        running = rewards[..., t] + gamma * running * (~dones[..., t].bool())
        returns[..., t] = running
    return returns


def generalized_advantage_estimate(rewards, values, dones, gamma=0.99, lam=0.95):
    """Return GAE advantages and value targets.

    ``rewards``/``dones`` are ``(..., T)`` and ``values`` is ``(..., T+1)``.
    The final value is the bootstrap estimate after the last observed step.
    """
    rewards = torch.as_tensor(rewards)
    values = torch.as_tensor(values, device=rewards.device)
    dones = torch.as_tensor(dones, device=rewards.device)
    if values.shape[:-1] != rewards.shape[:-1] or values.shape[-1] != rewards.shape[-1] + 1:
        raise ValueError("values must match rewards except for one extra final step")
    if dones.shape != rewards.shape:
        raise ValueError("dones must match rewards")
    advantage = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in range(rewards.shape[-1] - 1, -1, -1):
        alive = (~dones[..., t].bool()).to(rewards.dtype)
        delta = rewards[..., t] + gamma * alive * values[..., t + 1] - values[..., t]
        running = delta + gamma * lam * alive * running
        advantage[..., t] = running
    return advantage, advantage + values[..., :-1]


def normalize_advantage(advantage, mask, epsilon=1e-8):
    """Standardize advantages using valid entries only and zero padded entries."""
    advantage = torch.as_tensor(advantage)
    mask = torch.as_tensor(mask, device=advantage.device, dtype=advantage.dtype)
    valid = mask.bool()
    if not valid.any():
        return torch.zeros_like(advantage)
    values = advantage[valid]
    std = values.std(unbiased=False)
    return ((advantage - values.mean()) / (std + epsilon)) * mask


def clipped_policy_loss(old_logp, logp, advantage, mask, clip_eps=0.2):
    """PPO's clipped policy loss averaged over valid action positions."""
    ratio = torch.exp(logp - old_logp)
    pg_losses = -advantage * ratio
    pg_losses2 = -advantage * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    loss = torch.maximum(pg_losses, pg_losses2)
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def clipped_value_loss(values, old_values, returns, mask, value_clip=0.2):
    """PPO value loss using the worse of unclipped and clipped squared errors."""
    vpredclipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    vf_losses1 = (values - returns).square()
    vf_losses2 = (vpredclipped - returns).square()
    loss = 0.5 * torch.maximum(vf_losses1, vf_losses2)
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def categorical_entropy(logits, mask):
    """Mean categorical policy entropy over valid positions."""
    logp = F.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    mask = mask.to(entropy.dtype)
    return (entropy * mask).sum() / mask.sum().clamp_min(1)


def ppo_loss(old_logp, logp, advantage, values, old_values, returns, logits,
             mask, clip_eps=0.2, value_clip=0.2, value_coef=0.5,
             entropy_coef=0.01):
    """Combine PPO's policy, value, and entropy terms into one scalar loss."""
    policy = clipped_policy_loss(old_logp, logp, advantage, mask, clip_eps)
    value = clipped_value_loss(values, old_values, returns, mask, value_clip)
    entropy = categorical_entropy(logits, mask)
    return policy + value_coef * value - entropy_coef * entropy
