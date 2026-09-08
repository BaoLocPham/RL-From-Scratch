"""GRPO from scratch. Fill the TODOs, then run ``check.py``."""

import torch


# ============================================================ STAGE 1
def group_relative_advantage(rewards, group_index, epsilon=1e-6):
    """Return one standardized scalar per completion, using its prompt group.

    A group containing one completion uses mean zero and standard deviation one.
    This mirrors the production special case and avoids an undefined sample std.
    """
    # TODO stage 1: normalize within each group and keep zero-variance groups finite.
    pass


# ============================================================ STAGES 2 AND 4
def clipped_surrogate_loss(old_logp, logp, advantage, mask, eps=0.2,
                           ref_logp=None, beta=0.0):
    """Return the masked scalar PPO clipped-surrogate loss.

    ``advantage`` contains one number per completion, while log probabilities
    and masks contain one number per token.
    """
    # TODO stage 2: apply PPO clipping (TRL calls its two coefficients coef_1
    # and coef_2) and average only over valid tokens.
    # TODO stage 4: when a reference and nonzero weight are supplied, include
    # the stage-3 KL penalty without changing the default result.
    pass


# ============================================================ STAGE 3
def kl_penalty_k3(ref_logp, logp):
    """Return TRL's non-negative per-token k3 KL estimate."""
    # TODO stage 3: estimate divergence per sampled token without negative noise.
    pass
