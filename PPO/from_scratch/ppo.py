"""verl's PPO from scratch. Fill the TODOs, then run ``check.py``.

Two conventions hold everywhere, and most mistakes here are really mistakes
about one of them:

* Every tensor is ``(batch, response_length)``. No sequence dimension, no
  per-sequence scalars.
* ``response_mask`` decides which positions exist. Every mean, variance and
  loss is taken over it, never over the raw tensor.

Control flow is given. The TODOs sit on the expressions.
"""

import torch
import torch.nn.functional as F


# ============================================================ STAGE 1
def masked_mean(values, mask, axis=None):
    """Mean of ``values`` over the positions ``mask`` selects.

        masked_mean(tensor([[1., 2., 100.]]), tensor([[1., 1., 0.]]))  -> 1.5

    verl adds ``1e-8`` to the denominator rather than clamping it, so an
    all-zero mask returns about zero instead of raising.
    """
    total = ...              # TODO stage 1: sum only the selected positions
    count = ...              # TODO stage 1: how many were selected, kept safe at zero
    return total / count


def masked_var(values, mask, unbiased=True):
    """Variance over masked positions, Bessel-corrected when ``unbiased``."""
    centered = ...           # TODO stage 1: subtract the MASKED mean, not the raw one
    variance = ...           # TODO stage 1: reuse masked_mean on the squares
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        variance = ...       # TODO stage 1: n/(n-1), where n is the number of real positions
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Standardize ``values`` using masked statistics."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = ...           # TODO stage 1: centre, then scale by 1/sqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean     # put the original location back, keep the new spread
    return whitened


# ============================================================ STAGE 2
def compute_gae_advantage_return(token_level_rewards, values, response_mask,
                                 gamma, lam):
    """Return ``(advantages, returns)`` by Generalized Advantage Estimation.

    Walking backwards, each position's advantage is its TD error plus a decayed
    carry of the one after it:

        delta      = reward[t] + gamma * nextvalues - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam

    The part that is easy to get wrong is what the mask does. This is not a
    gym-style ``dones``: a masked-out position must **carry the running values
    through unchanged**, not reset them. Padding and observation tokens are
    skipped over, so credit still flows across them to the next real token.
    Reset them to zero and you silently cut every response into fragments.

    ``returns`` is computed BEFORE the advantage is whitened, so it stays on the
    value head's own scale.
    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = ...              # TODO stage 2: the TD error at t
            lastgaelam_ = ...        # TODO stage 2: decayed carry of the next position
            here = response_mask[:, t]
            nextvalues = ...         # TODO stage 2: take values[:, t] where `here`, else carry
            lastgaelam = ...         # TODO stage 2: take lastgaelam_ where `here`, else carry
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = ...                # TODO stage 2: advantage plus the baseline it was measured from
        advantages = ...             # TODO stage 2: whiten over the mask
    return advantages, returns


# ============================================================ STAGE 3
def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    """Reduce a ``(bs, response_length)`` loss matrix to one scalar.

    Four modes, differing in what gets equal weight. This is the entire subject
    of the Dr.GRPO paper, not a formatting choice:

    ``"token-mean"``              one masked mean over every token in the batch;
                                  a long response contributes proportionally more
    ``"seq-mean-token-sum"``      sum within a response, then mean over responses
    ``"seq-mean-token-mean"``     mean within a response, then mean over responses;
                                  length stops mattering
    ``"seq-mean-token-sum-norm"`` sum within a response, then divide by the PADDED
                                  WIDTH (``loss_mask.shape[-1]``) rather than by
                                  the response count -- Dr.GRPO's constant divisor
    """
    if loss_agg_mode == "token-mean":
        return ...           # TODO stage 3
    if loss_agg_mode == "seq-mean-token-sum":
        return ...           # TODO stage 3
    if loss_agg_mode == "seq-mean-token-mean":
        return ...           # TODO stage 3
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return ...           # TODO stage 3: note the divisor is NOT a count of responses
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


# ============================================================ STAGE 4
def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                        cliprange=None, cliprange_low=None, cliprange_high=None,
                        clip_ratio_c=3.0, loss_agg_mode="token-mean"):
    """Dual-clip PPO policy loss.

    Returns ``(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)`` -- the metrics
    are how you read whether the policy is moving too fast, so they are part of
    the contract, not extras.

    Two clips:

    * The ordinary PPO clip bounds the ratio to
      ``[1 - cliprange_low, 1 + cliprange_high]`` and keeps the PESSIMISTIC
      branch. Separate low/high bounds are what DAPO sets asymmetrically.
    * The dual clip applies ONLY where ``advantages < 0``. There the ordinary
      clip leaves the objective unbounded below, so a single badly-rated sample
      can dominate an update; flooring it at ``-advantages * clip_ratio_c``
      caps that. https://arxiv.org/pdf/1912.09729
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        f" but get the value: {clip_ratio_c}."
    )
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    # Clamped before exp: without this, exp overflows early in training.
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = ...              # TODO stage 4: recover the probability ratio
    ppo_kl = ...             # TODO stage 4: masked mean of the NEGATED log-ratio

    pg_losses1 = ...         # TODO stage 4: unclipped surrogate, negated for descent
    pg_losses2 = ...         # TODO stage 4: same, with the ratio clipped to the bounds
    clip_pg_losses1 = ...    # TODO stage 4: keep the pessimistic one
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = ...         # TODO stage 4: the dual-clip floor
    clip_pg_losses2 = ...    # TODO stage 4: apply that floor
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = ...          # TODO stage 4: dual clip only where the advantage is negative
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# ============================================================ STAGE 5
def clip_by_value(x, tensor_min, tensor_max):
    """``torch.clamp`` with tensor bounds instead of scalars."""
    return ...               # TODO stage 5: compose two elementwise ops


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value,
                       loss_agg_mode="token-mean"):
    """Clipped value loss. Returns ``(vf_loss, vf_clipfrac)``.

    Mind the argument order: ``values`` is the critic's OLD prediction, the
    thing being clipped around. ``returns`` is the target.

    Same pessimism as the policy clip -- keep the LARGER squared error, so one
    minibatch cannot move the value head far. The ``0.5`` lands after
    aggregation, not before.
    """
    vpredclipped = ...       # TODO stage 5: hold vpreds within cliprange_value of values
    vf_losses1 = ...         # TODO stage 5: unclipped squared error
    vf_losses2 = ...         # TODO stage 5: clipped squared error
    clipped_vf_losses = ...  # TODO stage 5: the pessimistic choice
    vf_loss = ...            # TODO stage 5: aggregate, then halve
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


# ============================================================ STAGE 6
def entropy_from_logits(logits):
    """Per-position categorical entropy, numerically stable.

    ``logsumexp(logits) - sum(softmax(logits) * logits)``. Equivalent to
    ``-sum(p * log p)`` but never takes the log of a zero probability.
    """
    pd = F.softmax(logits, dim=-1)
    return ...               # TODO stage 6: the stable form, not -sum(p * log p)


def compute_entropy_loss(logits, response_mask, loss_agg_mode="token-mean"):
    """Aggregate per-position entropy. Returned POSITIVE.

    A trainer subtracts ``entropy_coeff * this`` from the objective, so that
    maximizing entropy preserves exploration.
    """
    token_entropy = ...      # TODO stage 6
    return ...               # TODO stage 6: reuse stage 3


# ============================================================ STAGE 7
def kl_penalty(logprob, ref_logprob, kl_penalty):
    """Per-token KL against a frozen reference. Four estimators.

    ``"k1"``/``"kl"``          the plain log-ratio; unbiased, but one sample of it
                               is negative about half the time
    ``"abs"``                  its absolute value; non-negative, biased upward
    ``"k2"``/``"mse"``         half the squared log-ratio; non-negative, low variance
    ``"k3"``/``"low_var_kl"``  non-negative AND unbiased, which is why GRPO uses it;
                               http://joschu.net/blog/kl-approx.html

    Watch the direction: ``k3`` is built from ``ref_logprob - logprob``, the
    opposite of ``k1``. It clamps twice -- before ``exp`` and after -- because
    the exponential makes it the most explosive of the four.
    """
    if kl_penalty in ("kl", "k1"):
        return ...           # TODO stage 7
    if kl_penalty == "abs":
        return ...           # TODO stage 7
    if kl_penalty in ("mse", "k2"):
        return ...           # TODO stage 7
    if kl_penalty in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)
        kld = ...            # TODO stage 7: non-negative and unbiased, from `kl`
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError("'full' needs full-vocabulary logits, not log-probs")
    raise NotImplementedError(f"unknown kl_penalty: {kl_penalty}")


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Fold a KL penalty into the reward, BEFORE any advantage is computed.

    This is where PPO's reference-drift penalty goes in verl -- inside the
    reward, so GAE sees it and it propagates backward through the trajectory.
    GRPO instead adds its KL to the final loss. The two are not
    interchangeable: a penalty inside the reward gets bootstrapped and
    discounted, one inside the loss does not.

    Uses the raw ``k1`` log-ratio inline rather than calling ``kl_penalty``.
    """
    kl = ...                 # TODO stage 7: which direction, given it is SUBTRACTED below?
    return token_level_scores - kl * kl_ratio
