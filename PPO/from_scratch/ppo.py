"""PPO from scratch, the way verl writes it.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (most TODOs are one line).
  2. Try it:    python PPO/from_scratch/ppo.py       prints your results next to the expected ones
  3. Check it:  python PPO/from_scratch/check.py     stops at the first stage that is not right yet
  4. Move on to the next stage.

Where this sits. Surrogates/ ended with: PPO = the loop with L^CLIP in the slot,
"plus a value loss, an entropy bonus and GAE". This exercise builds exactly
that list, then the loop, at verl's shapes. Paper §5, eq. 9:

    L^{CLIP+VF+S}(theta) = E_t[ L^CLIP_t - c1 * L^VF_t + c2 * S[pi](s_t) ]

    stage 1  one decision -> T tokens     log pi per token, and the mask
    stage 2  A_t                          GAE with a critic (eq. 11-12)
    stage 3  E_t                          four ways to average over tokens
    stage 4  L^CLIP                       your Surrogates clip, per token (eq. 7)
    stage 5  L^VF                         training the critic
    stage 6  S                            the entropy bonus
    stage 7  KL to a reference            the RLHF addition (not in the paper)
    stage 8  the loop                     Algorithm 1 with eq. 9, on the token task

Two conventions hold everywhere, and most mistakes are really about one of them:
  * Every tensor is (batch, response_length). No per-sequence scalars: one reward
    for a whole response is a row that is zero except at its last real token.
  * response_mask decides which positions exist. Every mean, variance and loss is
    taken over it, never over the raw tensor.

Names, argument orders and return tuples are verl's (trainer/ppo/core_algos.py,
utils/torch_functional.py). GRPO/from_scratch imports your agg_loss,
compute_policy_loss, kl_penalty and masked_mean, so finish this one first.

Try not to open ../common.py (the reference) -- the hints below are enough.
"""

import torch
import torch.nn.functional as F


# ============================================================ STAGE 1: from one decision to T tokens
def logprobs_from_logits(logits, labels):
    """log pi(token) at every position: (bs, len, vocab) logits -> (bs, len).

    On the toy track this was policy.dist(qtype).log_prob(action): one decision,
    two actions. An LLM makes one decision per token over the whole vocabulary,
    so the model returns a row of logits at EVERY position, and you need the
    log-probability of the token that was actually sampled at each one.

    Worked example, one response of 2 tokens, vocabulary of 3:
        logits = [[[0, 0, 0], [2, 0, -1]]],  labels = [[1, 0]]
        position 0: uniform, log(1/3)                           = -1.0986
        position 1: 2 - log(e^2 + e^0 + e^-1) = 2 - log(8.757) = -0.1698
        -> [[-1.0986, -0.1698]]
    """
    # Hint: F.log_softmax(logits, dim=-1) over the vocabulary; then
    #       `.gather(-1, labels.unsqueeze(-1)).squeeze(-1)` picks one entry per position.
    logp = ...               # TODO stage 1: log-probabilities over the vocabulary, (bs, len, vocab)
    return ...               # TODO stage 1: the sampled token's entry at each position -> (bs, len)


def masked_mean(values, mask, axis=None):
    """Mean of ``values`` over the positions ``mask`` selects.

        masked_mean([[1., 2., 100.]], [[1., 1., 0.]]) = (1 + 2) / 2 = 1.5   (the padded 100 never counts)

    verl adds 1e-8 to the denominator rather than clamping it, so an all-zero
    mask returns about zero instead of raising.
    """
    # Hint: `(values * mask).sum(axis=axis)` sums only the selected positions.
    total = ...              # TODO stage 1: sum over the selected positions
    count = ...              # TODO stage 1: how many were selected, plus 1e-8
    return total / count


def masked_var(values, mask, unbiased=True):
    """Variance over masked positions, Bessel-corrected when ``unbiased``.

        values [[1, 2, 3, 100]], mask [[1, 1, 1, 0]]: masked mean 2,
        squared deviations 1, 0, 1 -> mean 0.667, times n/(n-1) = 3/2 -> 1.0
    """
    centered = ...           # TODO stage 1: values minus the MASKED mean
    variance = ...           # TODO stage 1: masked mean of the squares
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        variance = ...       # TODO stage 1: times n / (n - 1), n = mask_sum, the number of REAL positions
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Standardize ``values`` with masked statistics: (values - mean) / sqrt(var + 1e-8).

        values [[1, 2, 3, 100]], mask [[1, 1, 1, 0]] -> [-1, 0, 1, (padding: ignore)]
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    # Hint: torch.rsqrt(x) is 1 / sqrt(x).
    whitened = ...           # TODO stage 1: centre, then scale
    if not shift_mean:
        whitened += mean     # put the original location back, keep the new spread
    return whitened


# ============================================================ STAGE 2: the advantage, GAE (eq. 11-12)
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    """Return ``(advantages, returns)`` by Generalized Advantage Estimation.

    The toy track's advantage was reward - mean(reward): one step, no critic.
    With T tokens and one reward at the end, each token needs its OWN share of
    the credit, and the critic V(s_t) supplies the baseline at every position:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)                    (eq. 12, the TD error)
        A_t     = delta_t + gamma * lam * A_{t+1}                      (eq. 11, walked backwards)

    Worked example, gamma 0.9, lam 0.95 (so gamma * lam = 0.855):
        rewards [0, 0, 1, pad], values [0.1, 0.2, 0.3, pad]
        t=2: delta = 1 + 0.9 * 0   - 0.3 = 0.70                  A_2 = 0.70
        t=1: delta = 0 + 0.9 * 0.3 - 0.2 = 0.07   A_1 = 0.07 + 0.855 * 0.70  = 0.6685
        t=0: delta = 0 + 0.9 * 0.2 - 0.1 = 0.08   A_0 = 0.08 + 0.855 * 0.6685 = 0.6516
        returns = A + V = [0.7516, 0.8685, 1.0]        (the critic's target, stage 5)
        then the advantages are whitened over the mask (stage 1).

    What the mask does is the easy part to get wrong. It is NOT a gym-style
    `dones`. A masked-out position -- padding, or a tool's output in the middle
    of an agentic response -- must CARRY `nextvalues` and `lastgaelam` through
    unchanged, not reset them, so credit flows across it to the next real token.
    Reset them and every response with a tool call is silently cut in two.

    `returns` is computed BEFORE the advantages are whitened, so it stays on the
    critic's own scale.
    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = ...              # TODO stage 2: eq. 12 at position t, using nextvalues
            lastgaelam_ = ...        # TODO stage 2: eq. 11: delta plus the decayed carry
            here = response_mask[:, t]            # 1 where position t is real, 0 where it is not
            # Hint: `a * here + (1 - here) * b` means "a where real, keep b where not".
            nextvalues = ...         # TODO stage 2: values[:, t] where real, else carry nextvalues
            lastgaelam = ...         # TODO stage 2: lastgaelam_ where real, else carry lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = ...                # TODO stage 2: the advantage plus the baseline it was measured from
        advantages = ...             # TODO stage 2: whiten over the mask (your stage 1)
    return advantages, returns


# ============================================================ STAGE 3: E_t, averaging over tokens
def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    """Reduce a (bs, response_length) loss matrix to one scalar: the paper's E_t, four ways.

    On the toy track every rollout was one number, so "the mean" was obvious.
    With responses of different lengths it is a choice, and it decides whether
    a long response outweighs a short one -- the whole subject of the Dr.GRPO paper.

    Worked example, loss [[1, 2, 3, pad], [4, 5, pad, pad]] (lengths 3 and 2, width 4):
        "token-mean"               every token equal: (1+2+3+4+5) / 5        = 3.0
        "seq-mean-token-sum"       sum per response, mean over them: (6+9)/2 = 7.5
        "seq-mean-token-mean"      mean per response, then mean: (2 + 4.5)/2 = 3.25
        "seq-mean-token-sum-norm"  all sums / the padded WIDTH: (6+9) / 4   = 3.75
                                   (Dr.GRPO's constant divisor, not a count)
    """
    # Hint: `(loss_mat * loss_mask).sum(dim=-1)` is one sum per response;
    #       `loss_mask.sum(dim=-1)` is each response's length; loss_mask.shape[-1] the width.
    if loss_agg_mode == "token-mean":
        return ...           # TODO stage 3: your masked_mean over everything
    if loss_agg_mode == "seq-mean-token-sum":
        return ...           # TODO stage 3: the mean of the per-response sums
    if loss_agg_mode == "seq-mean-token-mean":
        return ...           # TODO stage 3: the mean of (per-response sum / per-response length)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return ...           # TODO stage 3: the sum of the per-response sums / the padded width
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


# ============================================================ STAGE 4: L^CLIP per token (eq. 7)
def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                        cliprange=None, cliprange_low=None, cliprange_high=None,
                        clip_ratio_c=3.0, loss_agg_mode="token-mean"):
    """Dual-clip PPO policy loss. Returns ``(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)``.

    You wrote the core in Surrogates stage 4, for one decision:
        loss = -min(r * A, clip(r, 1 - eps, 1 + eps) * A)
    verl writes it per token, with the minus sign moved inside -- which turns
    the pessimistic MIN into a MAX, because -min(x, y) = max(-x, -y):
        pg_losses1 = -A * r                      the unclipped term, negated
        pg_losses2 = -A * clip(r, 1 - low, 1 + high)
        clip_pg_losses1 = max(pg_losses1, pg_losses2)

    Worked example, one token with A = +1: r = 1.35 -> pg_losses1 = -1.35,
    pg_losses2 = -1.2, max = -1.2 (clipped: no more credit past 1 + 0.2).

    Two things verl adds on top of the paper:
      * separate low/high bounds -- DAPO's "clip-higher" sets high > low;
      * a DUAL clip, only where A < 0. There the ordinary clip leaves the loss
        unbounded as r grows (a bad token made much MORE likely), so one sample
        can dominate the update. verl floors it at -A * clip_ratio_c.
        https://arxiv.org/pdf/1912.09729

    The metrics are part of the contract: pg_clipfrac is the share of tokens the
    clip is holding back, and ppo_kl = mean(old_log_prob - log_prob) estimates
    KL(pi_old || pi_theta) from the sampled tokens -- the k1 estimator of the
    same quantity TRPO/trpo.py's mean_kl computes exactly.
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
    ratio = ...              # TODO stage 4: r = pi_theta / pi_old, from the clamped log-ratio
    ppo_kl = ...             # TODO stage 4: masked mean of MINUS the log-ratio

    pg_losses1 = ...         # TODO stage 4: -A * r
    pg_losses2 = ...         # TODO stage 4: -A * r clamped to [1 - cliprange_low, 1 + cliprange_high]
    clip_pg_losses1 = ...    # TODO stage 4: the pessimistic one (torch.maximum: the minus sign flipped min)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = ...         # TODO stage 4: the dual-clip floor, -A * clip_ratio_c
    clip_pg_losses2 = ...    # TODO stage 4: the smaller of pg_losses3 and clip_pg_losses1
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # Hint: torch.where(condition, a, b) takes a where the condition holds, else b.
    pg_losses = ...          # TODO stage 4: dual-clipped where A < 0, ordinary clip elsewhere
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# ============================================================ STAGE 5: L^VF, training the critic
def clip_by_value(x, tensor_min, tensor_max):
    """``torch.clamp`` with tensor bounds instead of scalars."""
    # Hint: the smaller of x and the max, then the larger of that and the min.
    return ...               # TODO stage 5: torch.max(torch.min(...), ...)


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value,
                       loss_agg_mode="token-mean"):
    """Clipped value loss, eq. 9's L^VF. Returns ``(vf_loss, vf_clipfrac)``.

    The paper's L^VF is the plain squared error (V_theta(s_t) - V_targ)^2, with
    V_targ = the returns from stage 2. verl clips it the way the policy is
    clipped: the new prediction may move at most cliprange_value from the
    critic's OLD prediction, and the LARGER squared error is kept -- the same
    pessimism, so one minibatch cannot yank the critic far.

    Mind the argument order: `values` is the OLD prediction (the thing being
    clipped around), `returns` is the target. The 0.5 is applied after aggregation.

    Worked example, one token: old value 0.1, target 0.75, new prediction 0.2, cliprange 0.05:
        clipped prediction = clamp(0.2, 0.05, 0.15)   = 0.15
        unclipped (0.2 - 0.75)^2 = 0.3025, clipped (0.15 - 0.75)^2 = 0.36 -> keep 0.36
        vf_loss = 0.5 * 0.36 = 0.18
    """
    vpredclipped = ...       # TODO stage 5: vpreds held within cliprange_value of `values`
    vf_losses1 = ...         # TODO stage 5: unclipped squared error to the returns
    vf_losses2 = ...         # TODO stage 5: clipped squared error to the returns
    clipped_vf_losses = ...  # TODO stage 5: the larger of the two
    vf_loss = ...            # TODO stage 5: 0.5 * your agg_loss of it
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


# ============================================================ STAGE 6: S, the entropy bonus
def entropy_from_logits(logits):
    """Per-position entropy H = -sum p log p, in a numerically stable form.

        H = logsumexp(logits) - sum(softmax(logits) * logits)

    Same number as -sum(p * log p), but never takes the log of a zero probability.

    Worked example: logits [0, 0, 0] (uniform over 3) -> ln 3 = 1.0986;
                    logits [2, 0, -1]                   -> 0.5243 (more certain, lower)
    """
    pd = F.softmax(logits, dim=-1)
    # Hint: torch.logsumexp(logits, dim=-1) and torch.sum(..., dim=-1).
    return ...               # TODO stage 6: the stable form


def compute_entropy_loss(logits, response_mask, loss_agg_mode="token-mean"):
    """Eq. 9's S, aggregated over the tokens. Returned POSITIVE.

    Eq. 9 ADDS c2 * S to the objective, to keep the policy from collapsing onto
    one token too early; so the loss a trainer minimises SUBTRACTS it (stage 8).
    """
    token_entropy = ...      # TODO stage 6: per-position entropy
    return ...               # TODO stage 6: aggregated with your agg_loss


# ============================================================ STAGE 7: KL to a reference model
def kl_penalty(logprob, ref_logprob, kl_penalty):
    """Per-token KL against a frozen REFERENCE model. Four single-sample estimators.

    Not the paper's KL. The paper's eq. 8 penalises distance from theta_old,
    the policy that collected the batch. This one penalises distance from a
    frozen copy of the model from BEFORE RL (the SFT model), so the policy
    cannot drift into gibberish that games the reward -- the RLHF addition
    (InstructGPT). And over a whole vocabulary the exact sum does not fit in
    memory, so verl estimates it from the sampled token alone. With
    r = logprob - ref_logprob:

        "k1" / "kl"          r                      unbiased, but negative about half the time
        "abs"                |r|                    never negative, biased upward
        "k2" / "mse"         0.5 * r^2              never negative, low variance, slightly biased
        "k3" / "low_var_kl"  exp(-r) + r - 1        never negative AND unbiased: what GRPO uses
                                                    http://joschu.net/blog/kl-approx.html

    Worked example, logprob -1.0, ref_logprob -1.2 (r = 0.2):
        k1 0.2,  abs 0.2,  k2 0.02,  k3 exp(-0.2) + 0.2 - 1 = 0.0187

    verl builds k3 from kl = ref_logprob - logprob (so exp(kl) - kl - 1 is the
    same formula) and clamps twice, before exp and after, because the
    exponential makes it the most explosive of the four.
    """
    if kl_penalty in ("kl", "k1"):
        return ...           # TODO stage 7
    if kl_penalty == "abs":
        return ...           # TODO stage 7
    if kl_penalty in ("mse", "k2"):
        return ...           # TODO stage 7: .square() squares element by element
    if kl_penalty in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)
        kld = ...            # TODO stage 7: exp(kl) - kl - 1
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError("'full' needs full-vocabulary logits, not log-probs")
    raise NotImplementedError(f"unknown kl_penalty: {kl_penalty}")


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """PPO puts its reference KL INSIDE the reward, before GAE sees it.

        reward_t = score_t - kl_ratio * (old_log_prob_t - ref_log_prob_t)

    So the penalty lands on EVERY token, and GAE carries it backwards like any
    reward. (GRPO instead adds its KL to the loss -- the two are not
    interchangeable.) It uses the raw k1 log-ratio inline, not kl_penalty.

    Worked example: score 1, old_log_prob 0, ref_log_prob -0.1, kl_ratio 0.2:
        1 - 0.2 * (0 - (-0.1)) = 0.98
    """
    kl = ...                 # TODO stage 7: the k1 log-ratio, old minus reference
    return token_level_scores - kl * kl_ratio


# ============================================================ STAGE 8: the loop, Algorithm 1 with eq. 9
def compute_advantage(batch, kl_ratio=0.02, gamma=1.0, lam=0.95):
    """Between collecting and updating, ONCE per batch: KL into the reward, then GAE.

    `batch` is what PPO/task.py's rollout returns: a dict of (bs, len) tensors
    with token_level_scores, old_log_prob, ref_log_prob, values, response_mask.
    Advantages come from the critic as it was at collection time and stay fixed
    through every epoch that follows -- the toy track's rule, unchanged.
    """
    batch["token_level_rewards"] = ...    # TODO stage 8: your compute_rewards on the batch's scores and log-probs
    batch["advantages"], batch["returns"] = ...   # TODO stage 8: your GAE on those rewards, the batch's values and mask
    return batch


def ppo_update(model, optimizer, batch, epochs=4, minibatch_size=8, cliprange=0.2,
               cliprange_value=0.2, vf_coef=0.5, entropy_coeff=0.01):
    """Algorithm 1's inner loop: `epochs` passes over ONE batch, in minibatches, on eq. 9.

    The Surrogates loop took the whole batch per step; the paper splits each
    epoch into minibatches of size M, one optimizer step each. The slot now
    holds all three terms of eq. 9, negated because optimizers minimise:

        loss = pg_loss + vf_coef * vf_loss - entropy_coeff * entropy

    (pg_loss is already -L^CLIP; vf_loss is +L^VF because the critic should
    get BETTER; the entropy bonus is subtracted so the policy keeps exploring.)

    `model.logits(m)` is the forward pass and `model.values(m)` the critic's
    prediction, for a minibatch of m responses. Returns each metric averaged
    over every update.
    """
    n = batch["tokens"].shape[0]
    totals, updates = {}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                                   # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            mb = {key: value[order[start:start + minibatch_size]] for key, value in batch.items()}
            m, mask = mb["tokens"].shape[0], mb["response_mask"]
            logits = model.logits(m)                                # the forward pass

            log_prob = ...                          # TODO stage 8: log pi_theta of mb["tokens"] (stage 1)
            pg_loss, pg_clipfrac, ppo_kl, _ = ...   # TODO stage 8: stage 4, with mb's old_log_prob and advantages
            vf_loss, _ = ...                        # TODO stage 8: stage 5: model.values(m) vs mb's returns, around mb's values
            entropy = ...                           # TODO stage 8: stage 6, on the logits
            loss = ...                              # TODO stage 8: eq. 9, negated

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                            # bookkeeping for the log
            for name, value in (("pg_loss", pg_loss), ("vf_loss", vf_loss), ("entropy", entropy),
                                ("pg_clipfrac", pg_clipfrac), ("ppo_kl", ppo_kl)):
                totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: total / updates for name, total in totals.items()}


# ============================================================ playground
def _show(label, fn, expected):
    """Run one of your functions and print it next to the expected value."""
    try:
        got = fn()
    except Exception as exc:                           # unfinished TODOs land here
        got = f"not done yet ({type(exc).__name__})"
    if got is Ellipsis:
        got = "not done yet"
    elif isinstance(got, torch.Tensor):
        got = [round(x, 4) + 0.0 for x in got.detach().flatten().tolist()]   # + 0.0: no -0.0
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<42} yours: {str(got):<24} expected: {expected}")


if __name__ == "__main__":
    R = torch.tensor([[0., 0., 1., 0.]])
    V = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    M = torch.tensor([[1., 1., 1., 0.]])
    print("Your functions on the worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  logprobs_from_logits",
          lambda: logprobs_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.]]]), torch.tensor([[1, 0]])),
          [-1.0986, -0.1698])
    _show("stage 1  masked_mean", lambda: masked_mean(torch.tensor([[1., 2., 100.]]),
                                                      torch.tensor([[1., 1., 0.]])), 1.5)
    _show("stage 1  masked_var", lambda: masked_var(torch.tensor([[1., 2., 3., 100.]]), M), 1.0)
    _show("stage 2  GAE returns (gamma .9, lam .95)",
          lambda: compute_gae_advantage_return(R, V, M, 0.9, 0.95)[1][0, :3], [0.7516, 0.8685, 1.0])
    loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
    M2 = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
    _show("stage 3  agg_loss, token-mean", lambda: agg_loss(loss, M2, "token-mean"), 3.0)
    _show("stage 3  agg_loss, seq-mean-token-sum-norm",
          lambda: agg_loss(loss, M2, "seq-mean-token-sum-norm"), 3.75)
    one = torch.ones(1, 1)
    _show("stage 4  one token, A +1, r 1.35",
          lambda: compute_policy_loss(torch.zeros(1, 1), torch.tensor([[1.35]]).log(), one, one,
                                      cliprange=0.2)[0], -1.2)
    _show("stage 5  one token, clipped value loss",
          lambda: compute_value_loss(torch.tensor([[0.2]]), torch.tensor([[0.75]]), torch.tensor([[0.1]]),
                                     one, 0.05)[0], 0.18)
    _show("stage 6  entropy of [0,0,0] and [2,0,-1]",
          lambda: entropy_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.]]])), [1.0986, 0.5243])
    _show("stage 7  k3, r = 0.2", lambda: kl_penalty(torch.tensor([[-1.0]]), torch.tensor([[-1.2]]), "k3"),
          0.0187)
    _show("stage 7  compute_rewards", lambda: compute_rewards(one, torch.zeros(1, 1),
                                                              torch.full((1, 1), -0.1), 0.2), 0.98)

    def ten_iterations():
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import task
        history, _ = task.train(sys.modules[__name__], iterations=10)
        return f"reward {history[0]['reward']:.3f} -> {history[-1]['reward']:.3f}"
    _show("stage 8  10 iterations on the token task", ten_iterations, "reward 0.341 -> 0.971")
    print("\nWhen these match, run:  python PPO/from_scratch/check.py")
