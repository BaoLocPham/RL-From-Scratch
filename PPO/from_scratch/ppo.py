"""PPO from scratch. Fill each TODO, then run ``check.py``."""

import torch
import torch.nn.functional as F


# ============================================================ STAGE 1
def discounted_returns(rewards, dones, gamma=0.99):
    """Return reward-to-go along the last axis, respecting episode endings."""
    # TODO stage 1: accumulate future rewards without crossing terminal steps.
    pass


# ============================================================ STAGE 2
def generalized_advantage_estimate(rewards, values, dones, gamma=0.99, lam=0.95):
    """Return the GAE advantage and its corresponding value target."""
    # TODO stage 2: combine bootstrapped prediction errors across valid transitions.
    pass


# ============================================================ STAGE 3
def normalize_advantage(advantage, mask, epsilon=1e-8):
    """Standardize valid advantages and keep padded positions at zero."""
    # TODO stage 3: padded values must not affect the statistics.
    pass


# ============================================================ STAGE 4
def clipped_policy_loss(old_logp, logp, advantage, mask, clip_eps=0.2):
    """Return PPO's masked clipped-surrogate policy loss."""
    # TODO stage 4: limit TRL's pg_losses/pg_losses2 policy-ratio incentives in
    # both advantage directions.
    pass


# ============================================================ STAGE 5
def clipped_value_loss(values, old_values, returns, mask, value_clip=0.2):
    """Return the masked clipped value-regression loss."""
    # TODO stage 5: compare TRL's vf_losses1/vf_losses2 so a new value estimate
    # cannot move too far at once.
    pass


# ============================================================ STAGE 6
def categorical_entropy(logits, mask):
    """Return mean categorical entropy over valid positions."""
    # TODO stage 6a: compute policy uncertainty without counting padding.
    pass


def ppo_loss(old_logp, logp, advantage, values, old_values, returns, logits,
             mask, clip_eps=0.2, value_clip=0.2, value_coef=0.5,
             entropy_coef=0.01):
    """Combine policy, value, and entropy into PPO's scalar training loss."""
    # TODO stage 6b: compose the three completed objectives with their coefficients.
    pass
