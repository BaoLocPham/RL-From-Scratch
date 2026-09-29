"""PPO from scratch, Part 1: core PPO, the paper only.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (most TODOs are one line).
  2. Try it:    python PPO/from_scratch/ppo.py       prints your results next to the expected ones
  3. Check it:  ./scripts/run_ppo.sh check core      stops at the first stage that is not right yet
  4. Move on to the next stage.

Where this sits. Surrogates/ ended with: PPO = the loop with L^CLIP in the slot,
"plus a value loss, an entropy bonus and GAE". Paper §5, eq. 9:

    L^{CLIP+VF+S}(theta) = E_t[ L^CLIP_t - c1 * L^VF_t + c2 * S[pi](s_t) ]

    stage 1  one decision -> T tokens     log pi per token, and the masked mean
    stage 2  A_t                          GAE with a critic (eq. 11-12)
    stage 3  L^CLIP                       your Surrogates clip, per token (eq. 7)
    stage 4  L^VF                         training the critic
    stage 5  S                            the entropy bonus
    stage 6  the loop                     Algorithm 1 with eq. 9, on PPO/task.py's token task

Part 2, verl's extras (stages 7-12), is in ppo_verl.py -- for later. It builds
on the functions you write here.

Two conventions hold everywhere, and most mistakes are really about one of them:
  * Every tensor is (batch, response_length). No per-sequence scalars: one reward
    for a whole response is a row that is zero except at its last real token.
  * response_mask decides which positions exist. Every mean and loss is taken
    over it, never over the raw tensor.

Names are verl's (trainer/ppo/core_algos.py, utils/torch_functional.py).

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
    """Mean of ``values`` over the positions ``mask`` selects: the paper's E_t, over real tokens.

        masked_mean([[1., 2., 100.]], [[1., 1., 0.]]) = (1 + 2) / 2 = 1.5   (the padded 100 never counts)

    verl adds 1e-8 to the denominator rather than clamping it, so an all-zero
    mask returns about zero instead of raising.
    """
    # Hint: `(values * mask).sum(axis=axis)` sums only the selected positions.
    total = ...              # TODO stage 1: sum over the selected positions
    count = ...              # TODO stage 1: how many were selected, plus 1e-8
    return total / count


# ============================================================ STAGE 2: the advantage, GAE (eq. 11-12)
def compute_gae(token_level_rewards, values, response_mask, gamma, lam):
    """Return raw ``(advantages, returns)`` by Generalized Advantage Estimation.

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
        returns = A + V = [0.7516, 0.8685, 1.0]        (the critic's target, stage 4)

    What the mask does is the easy part to get wrong. It is NOT a gym-style
    `dones`. A masked-out position -- padding, or a tool's output in the middle
    of an agentic response -- must CARRY `nextvalues` and `lastgaelam` through
    unchanged, not reset them, so credit flows across it to the next real token.
    Reset them and every response with a tool call is silently cut in two.

    ---------------------------------------------------------------- the intuition
    delta_t is the critic's SURPRISE at step t. Before the token, the critic
    predicted V(s_t). One step later it knows the reward r_t and can predict
    V(s_{t+1}) instead. If r_t + gamma * V(s_{t+1}) is bigger than V(s_t), the
    token made things better than expected: delta_t > 0.

    A_t adds up the surprises from t onwards, each later one weighted by
    another factor of gamma * lam:
        A_t = delta_t + (gamma*lam) * delta_{t+1} + (gamma*lam)^2 * delta_{t+2} + ...
    "How much better than expected did everything after this token turn out?",
    trusting near surprises more than far ones.

    lam slides between two advantages you already know:
        lam = 0:  A_t = delta_t                        one step; trusts the critic completely
        lam = 1:  A_t = (rewards from t on) - V(s_t)    the whole outcome minus the baseline --
                                                        the toy track's reward - mean(reward),
                                                        with the critic as the baseline
    In between, lam trades the critic's bias (it may be wrong) against the
    outcome's noise (one sampled response).

    ---------------------------------------------------------------- the loop
    Why backwards: A_t needs A_{t+1}, so start at the last position and reuse
    each result for the one before it -- one pass instead of one sum per token.
    Why no gradient: advantages and returns are fixed TARGETS, computed once
    per batch; the losses in stages 3 and 4 must not push gradient into them.
    Every `[:, t]` handles all responses in the batch at once; the loop is
    over positions only.

    The two variables carried from one iteration to the next ("next" means the
    next REAL position to the right, since the loop runs right to left):
        nextvalues   V of the next real position         (0 past the end: nothing follows)
        lastgaelam   A of the next real position         (0 past the end)
    and one temporary:
        lastgaelam_  A at position t, IF position t is real -- the mask decides

    Trace of the worked example (one row, gamma 0.9, lam 0.95, position 3 is padding):
        start                                            nextvalues 0.0   lastgaelam 0.0
        t=3 pad : delta = 0 + 0.9*0.0 - 0.4 = -0.40  -> ignored: both carried  0.0 / 0.0
        t=2 real: delta = 1 + 0.9*0.0 - 0.3 =  0.70  A = 0.70                   0.3 / 0.70
        t=1 real: delta = 0 + 0.9*0.3 - 0.2 =  0.07  A = 0.07 + 0.855*0.70     0.2 / 0.6685
        t=0 real: delta = 0 + 0.9*0.2 - 0.1 =  0.08  A = 0.08 + 0.855*0.6685   0.1 / 0.6516
        collected right to left [0.0, 0.70, 0.6685, 0.6516] -> reversed [0.6516, 0.6685, 0.70, 0.0]
    The padding's delta (-0.40) is computed, but the mask throws it away.
    """
    # No gradient: these are targets, computed once per batch (see "the loop" above).
    with torch.no_grad():
        # Past the last position nothing follows, so V = 0 and A = 0 there.
        # nextvalues = V of the next real position; lastgaelam = A of the next real position.
        nextvalues = 0
        lastgaelam = 0
        # A_t for each position, collected right to left; reversed after the loop.
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # Right to left: A_t is built from A_{t+1}, so the last position goes first.
        for t in reversed(range(gen_len)):
            # delta: the critic's surprise at t. Reward now, plus the discounted guess one step
            # later (nextvalues), minus the guess made before the token (values[:, t]).
            delta = ...              # TODO stage 2: eq. 12 at position t, using nextvalues
            # This position's advantage: its own surprise plus the next real position's advantage,
            # shrunk by gamma * lam. Only a candidate -- the mask below decides whether it counts.
            lastgaelam_ = ...        # TODO stage 2: eq. 11: delta plus the decayed carry
            # The mask decides. Real token: this position becomes the new "next" for position t-1.
            # Padding or tool output: skip it, and leave both carried values exactly as they were.
            here = response_mask[:, t]            # 1 where position t is real, 0 where it is not
            # Hint: `a * here + (1 - here) * b` means "a where real, keep b where not".
            # From here on, V(s_{t+1}) for position t-1 is this position's value (if it is real).
            nextvalues = ...         # TODO stage 2: values[:, t] where real, else carry nextvalues
            # ...and A_{t+1} for position t-1 is this position's advantage (if it is real).
            lastgaelam = ...         # TODO stage 2: lastgaelam_ where real, else carry lastgaelam
            # On a masked position this appends the carried value; nothing downstream reads it.
            advantages_reversed.append(lastgaelam)

        # Back to left-to-right order, one column per position: (batch, response_length).
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # returns: what the critic SHOULD have predicted at each position -- its baseline plus
        # how much better things turned out. Stage 4 trains V towards it.
        returns = ...                # TODO stage 2: the advantage plus the baseline it was measured from
    return advantages, returns


# ============================================================ STAGE 3: L^CLIP per token (eq. 7)
def ppo_clip_loss(old_log_prob, log_prob, advantages, response_mask, cliprange=0.2):
    """-L^CLIP (eq. 7), per token, averaged over the mask.

    You wrote this in Surrogates stage 4, for one decision:
        loss = -min(r * A, clip(r, 1 - eps, 1 + eps) * A)
    Per token, with the minus sign moved inside -- -min(x, y) = max(-x, -y) --
    so the pessimistic MIN becomes a MAX:
        loss_t = max(-A_t * r_t, -A_t * clip(r_t, 1 - eps, 1 + eps))

    Worked example, one token with A = +1, eps 0.2:
        r = 1.1:  max(-1.10, -1.10) = -1.10    still rewarded
        r = 1.35: max(-1.35, -1.20) = -1.20    clipped: no more credit past 1.2
        r = 0.7:  max(-0.70, -0.80) = -0.70    moved the wrong way: pays in full
    """
    # Hint: exp(log a - log b) = a / b; torch.clamp(x, lo, hi); torch.maximum(a, b).
    ratio = ...              # TODO stage 3: r = pi_theta / pi_old, per token
    unclipped = ...          # TODO stage 3: -A * r
    clipped = ...            # TODO stage 3: -A * r clamped to [1 - cliprange, 1 + cliprange]
    return ...               # TODO stage 3: your masked_mean of the larger of the two


# ============================================================ STAGE 4: L^VF, training the critic
def value_loss(vpreds, returns, response_mask):
    """L^VF (eq. 9): 0.5 * (V_theta(s_t) - V_targ)^2, averaged over the mask.

    V_targ is the returns from stage 2. The critic is a regression: make each
    position's prediction match the return that actually followed.

    Worked example, one token: prediction 0.2, return 0.75 -> 0.5 * (0.2 - 0.75)^2 = 0.15125
    """
    return ...               # TODO stage 4: 0.5 * your masked_mean of the squared error


# ============================================================ STAGE 5: S, the entropy bonus
def entropy_from_logits(logits):
    """Per-position entropy H = -sum p log p, in a numerically stable form.

        H = logsumexp(logits) - sum(softmax(logits) * logits)

    Same number as -sum(p * log p), but never takes the log of a zero probability.

    Worked example: logits [0, 0, 0] (uniform over 3) -> ln 3 = 1.0986;
                    logits [2, 0, -1]                   -> 0.5243 (more certain, lower)
    """
    pd = F.softmax(logits, dim=-1)
    # Hint: torch.logsumexp(logits, dim=-1) and torch.sum(..., dim=-1).
    return ...               # TODO stage 5: the stable form


def entropy_bonus(logits, response_mask):
    """S (eq. 9): the mean per-token entropy over the mask. Returned POSITIVE.

    Eq. 9 ADDS c2 * S to the objective, to keep the policy from collapsing onto
    one token too early; so the loss a trainer minimises SUBTRACTS it (stage 6).
    """
    return ...               # TODO stage 5: your masked_mean of the per-position entropy


# ============================================================ STAGE 6: the loop, Algorithm 1 with eq. 9
def compute_advantage(batch, gamma=1.0, lam=0.95):
    """Between collecting and updating, ONCE per batch: GAE on the scores.

    `batch` is what PPO/task.py's rollout returns: a dict of (bs, len) tensors
    with token_level_scores, old_log_prob, values, response_mask, tokens.
    Advantages come from the critic as it was at collection time and stay fixed
    through every epoch that follows -- the toy track's rule, unchanged.
    """
    batch["advantages"], batch["returns"] = ...   # TODO stage 6: your compute_gae on the batch's scores, values and mask
    return batch


def ppo_update(model, optimizer, batch, epochs=4, minibatch_size=8, cliprange=0.2,
               vf_coef=0.5, entropy_coeff=0.01):
    """Algorithm 1's inner loop: `epochs` passes over ONE batch, in minibatches, on eq. 9.

    The Surrogates loop took the whole batch per step; the paper splits each
    epoch into minibatches of size M, one optimizer step each. The slot now
    holds all three terms of eq. 9, negated because optimizers minimise:

        loss = pg_loss + vf_coef * vf_loss - entropy_coeff * entropy

    (pg_loss is already -L^CLIP; vf_loss is +L^VF because the critic should
    get BETTER; the entropy bonus is subtracted so the policy keeps exploring.)

    `model.logits(m)` is the forward pass and `model.values(m)` the critic's
    prediction, for a minibatch of m responses. Returns each loss term averaged
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

            log_prob = ...           # TODO stage 6: log pi_theta of mb["tokens"] (stage 1)
            pg_loss = ...            # TODO stage 6: stage 3, with mb's old_log_prob and advantages
            vf_loss = ...            # TODO stage 6: stage 4: model.values(m) against mb's returns
            entropy = ...            # TODO stage 6: stage 5, on the logits
            loss = ...               # TODO stage 6: eq. 9, negated

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                            # bookkeeping for the log
            for name, value in (("pg_loss", pg_loss), ("vf_loss", vf_loss), ("entropy", entropy)):
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


def _train(iterations=12):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import task
    history, _ = task.train(sys.modules[__name__], iterations=iterations)
    return f"reward {history[0]['reward']:.3f} -> {history[-1]['reward']:.3f}"


if __name__ == "__main__":
    R = torch.tensor([[0., 0., 1., 0.]])
    V = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    M = torch.tensor([[1., 1., 1., 0.]])
    one = torch.ones(1, 1)
    print("Your functions on the worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  logprobs_from_logits",
          lambda: logprobs_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.]]]), torch.tensor([[1, 0]])),
          [-1.0986, -0.1698])
    _show("stage 1  masked_mean", lambda: masked_mean(torch.tensor([[1., 2., 100.]]),
                                                      torch.tensor([[1., 1., 0.]])), 1.5)
    _show("stage 2  compute_gae returns", lambda: compute_gae(R, V, M, 0.9, 0.95)[1][0, :3], [0.7516, 0.8685, 1.0])
    _show("stage 3  ppo_clip_loss, A +1, r 1.35",
          lambda: ppo_clip_loss(torch.zeros(1, 1), torch.tensor([[1.35]]).log(), one, one), -1.2)
    _show("stage 4  value_loss, 0.2 vs 0.75", lambda: value_loss(torch.tensor([[0.2]]), torch.tensor([[0.75]]), one),
          0.1512)
    _show("stage 5  entropy of [0,0,0] and [2,0,-1]",
          lambda: entropy_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.]]])), [1.0986, 0.5243])
    _show("stage 6  core PPO, 12 iterations", lambda: _train(), "reward 0.341 -> 0.943")
    print("\nWhen these match, run:  ./scripts/run_ppo.sh check core")
    print("Part 2 (verl's extras, for later): python PPO/from_scratch/ppo_verl.py")
