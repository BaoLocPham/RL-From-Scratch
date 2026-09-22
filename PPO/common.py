"""PPO as verl implements it.

Names, argument orders and return tuples follow
`verl/trainer/ppo/core_algos.py` and `verl/utils/torch_functional.py`, so code
written against this module transfers to a real verl trainer unchanged.

Two conventions run through everything and are worth internalizing before
reading further:

* **Every tensor is ``(batch, response_length)``.** There is no separate
  sequence dimension and no per-sequence scalar. A single outcome reward for a
  whole response is carried as a ``token_level_rewards`` row that is zero
  everywhere except its last valid position, and a single advantage is a row
  repeated across the response.
* **``response_mask`` is the only thing that makes a position real.** It is the
  EOS mask: tokens after the end of the response are zero. Every mean, sum,
  variance and loss in this file is taken over it.

verl keeps GAE, GRPO, RLOO, REMAX and the rest in one `core_algos.py`. This
repo splits them by directory for teaching, so the shared pieces --
:func:`agg_loss`, :func:`compute_policy_loss`, :func:`kl_penalty` and the
masked helpers -- live here, and `GRPO/common.py` imports them.
"""

import torch
import torch.nn.functional as F

# --------------------------------------------------- verl/utils/torch_functional


def masked_sum(values, mask, axis=None):
    """Sum ``values`` over the positions ``mask`` selects."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Mean of ``values`` over the positions ``mask`` selects.

    Note the ``+ 1e-8`` on the denominator rather than a clamp: verl tolerates
    an all-zero mask by returning roughly zero instead of raising.
    """
    return masked_sum(values, mask, axis) / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Variance over masked positions, Bessel-corrected by default."""
    centered = values - masked_mean(values, mask)
    variance = masked_mean(centered**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        variance = variance * (mask_sum / (mask_sum - 1))
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Standardize ``values`` using masked statistics.

    ``shift_mean=False`` re-adds the original mean after scaling, which keeps
    the location and changes only the spread.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """``torch.clamp`` with tensor bounds instead of scalars."""
    return torch.max(torch.min(x, tensor_max), tensor_min)


def entropy_from_logits(logits):
    """Per-position categorical entropy, in a numerically stable form.

    ``logsumexp(logits) - sum(softmax(logits) * logits)`` avoids materializing
    ``log_softmax`` and never takes the log of a zero probability.
    """
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


# ------------------------------------------------------------------- advantage


def compute_gae_advantage_return(token_level_rewards, values, response_mask,
                                 gamma, lam):
    """Return ``(advantages, returns)`` by Generalized Advantage Estimation.

    All three inputs are ``(bs, response_length)``. Unlike a gym-style GAE there
    is no ``dones`` argument: ``response_mask`` plays that role, and it does so
    in a way that is easy to misread. Look at the two update lines --

        nextvalues = values[:, t] * mask + (1 - mask) * nextvalues
        lastgaelam = lastgaelam_ * mask + (1 - mask) * lastgaelam

    On a masked-out position the running values are *carried through
    unchanged*, not reset to zero. Padding and interleaved observation tokens
    are skipped over rather than treated as episode boundaries, so credit flows
    across them to the next real token.

    The returned advantage is whitened over the mask; ``returns`` is computed
    before that whitening, so it stays on the value head's own scale.
    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam
            here = response_mask[:, t]
            nextvalues = values[:, t] * here + (1 - here) * nextvalues
            lastgaelam = lastgaelam_ * here + (1 - here) * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


# ----------------------------------------------------------------- aggregation


def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    """Reduce a ``(bs, response_length)`` loss matrix to one scalar.

    The four modes differ in what gets equal weight, and the choice is not
    cosmetic -- it is the whole subject of the Dr.GRPO paper:

    ``"token-mean"``
        One masked mean over every token in the batch. A long response
        contributes more than a short one, in proportion to its length.
    ``"seq-mean-token-sum"``
        Sum within each response, then average over responses. Every response
        counts once, but a long one still carries a larger summed loss.
    ``"seq-mean-token-mean"``
        Average within each response, then average over responses. Every
        response counts once regardless of length.
    ``"seq-mean-token-sum-norm"``
        Sum within each response, then divide by the padded width rather than
        by the response count. This is Dr.GRPO's normalizer: a constant divisor
        removes the length bias the other modes introduce.
    """
    if loss_agg_mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    if loss_agg_mode == "seq-mean-token-sum":
        return torch.mean(torch.sum(loss_mat * loss_mask, dim=-1))
    if loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return torch.mean(seq_losses)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return torch.sum(torch.sum(loss_mat * loss_mask, dim=-1)) / loss_mask.shape[-1]
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


# ----------------------------------------------------------------------- losses


def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                        cliprange=None, cliprange_low=None, cliprange_high=None,
                        clip_ratio_c=3.0, loss_agg_mode="token-mean"):
    """Dual-clip PPO policy loss.

    Returns ``(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)``. The three
    metrics are not decoration: ``pg_clipfrac`` and ``ppo_kl`` are the standard
    read on whether the policy is moving too far per step.

    Two clips, not one:

    * The **ordinary PPO clip** bounds the ratio to
      ``[1 - cliprange_low, 1 + cliprange_high]`` and keeps the pessimistic
      branch with ``maximum``. Separate low and high bounds are what DAPO's
      "clip-higher" sets asymmetrically.
    * The **dual clip** applies only where ``advantages < 0``, flooring the loss
      at ``-advantages * clip_ratio_c``. On a badly-rated sample the ordinary
      clip does not bound the objective from below, so one such sample can
      dominate the update. ``clip_ratio_c`` (> 1, default 3.0) caps that.
      See https://arxiv.org/pdf/1912.09729.

    The log-ratio is clamped to +/-20 before exponentiating, which is the only
    thing preventing ``exp`` from overflowing early in training.
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        f" but get the value: {clip_ratio_c}."
    )
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value,
                       loss_agg_mode="token-mean"):
    """Clipped value-function loss. Returns ``(vf_loss, vf_clipfrac)``.

    The critic is held within ``cliprange_value`` of its own previous
    prediction, and the *larger* of the clipped and unclipped squared errors is
    kept -- the same pessimism as the policy clip, so one minibatch cannot move
    the value head far. The ``0.5`` factor is applied after aggregation.

    Note the argument order: ``vpreds, returns, values``. ``values`` is the old
    baseline prediction, not the target.
    """
    vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask,
                             loss_agg_mode=loss_agg_mode)
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def compute_entropy_loss(logits, response_mask, loss_agg_mode="token-mean"):
    """Aggregate per-position entropy. ``logits`` is ``(bs, response_len, vocab)``.

    Returned positive. A trainer *subtracts* ``entropy_coeff * entropy_loss``
    from the total objective, so that maximizing entropy preserves exploration.
    """
    token_entropy = entropy_from_logits(logits)
    return agg_loss(loss_mat=token_entropy, loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode)


# --------------------------------------------------------------------------- KL


def kl_penalty(logprob, ref_logprob, kl_penalty):
    """Per-token KL estimate against a frozen reference. Four estimators.

    ``"k1"`` / ``"kl"``
        The plain log-ratio. Unbiased, but a single sample of it is negative
        about half the time -- it estimates a non-negative quantity with a
        signed number.
    ``"abs"``
        Its absolute value. Non-negative, but biased upward.
    ``"k2"`` / ``"mse"``
        ``0.5 * (log-ratio)^2``. Non-negative, low variance, slightly biased.
    ``"k3"`` / ``"low_var_kl"``
        ``exp(-r) + r - 1`` written out. Non-negative *and* unbiased, which is
        why GRPO uses it. See http://joschu.net/blog/kl-approx.html.

    ``k3`` clamps twice -- the log-ratio to +/-20 before ``exp``, and the
    result to +/-10 after -- because the exponential makes it the most
    explosive of the four when the policy has drifted.
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)
        kld = (torch.exp(kl) - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError("'full' needs full-vocabulary logits, not log-probs")
    raise NotImplementedError(f"unknown kl_penalty: {kl_penalty}")


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Fold a KL penalty into the reward, before any advantage is computed.

    This is where PPO's reference-drift penalty goes in verl: subtracted from
    the token-level score, so GAE sees it and it propagates backward through
    the trajectory. GRPO instead adds its KL term to the final loss. Same
    intent, different placement, and the two are not interchangeable -- a
    penalty inside the reward gets bootstrapped and discounted, one inside the
    loss does not.

    Note it uses the raw ``k1`` log-ratio inline rather than calling
    :func:`kl_penalty`.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio
