"""TRPO, PPO paper Section 2.2 (eq. 3-4). Demo: ``python TRPO/run_trpo.py``.

    maximize    E_t[ pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) * A_t ]         (eq. 3)
    subject to  E_t[ KL[pi_theta_old(.|s_t), pi_theta(.|s_t)] ] <= delta      (eq. 4)

What it fixes in VPG:
  - eq. 3 (the ratio) knows WHICH model collected the batch, so the batch can
    be used for more than one update.
  - eq. 4 (the KL fence) keeps every update inside the "trust region" around
    theta_old -- the zone where the batch still describes the model.

How this file solves it -- the simplest way that respects both equations:
take small gradient steps on eq. 3, reusing the same batch (epochs, like VPG's
shortcut), and stop before the KL from theta_old goes past delta.

Its flaw: the fence is not part of the loss. After every step something outside
the loss has to measure the KL and undo the step if it went too far -- you
cannot just call .backward() on "subject to". (The paper's TRPO computes the
best step inside the fence with heavier second-order math instead; that
heaviness is what the paper calls TRPO's flaw.) PPO moves the fence INSIDE the
loss, with a clip.
"""

import torch


def surrogate(policy, qtype, action, advantage, old_logp):
    """Eq. 3: mean(ratio * A), with ratio = pi_theta / pi_theta_old.

    At theta_old every ratio is 1, so its gradient equals L^PG's: the same
    reading. After the model moves, the ratio still remembers theta_old; L^PG
    does not. On its own, though, nothing stops the ratio from growing without
    limit -- hence eq. 4.
    """
    # exp(log a - log b) = a / b, computed in log space for numerical stability.
    ratio = torch.exp(policy.dist(qtype).log_prob(action) - old_logp)
    return (ratio * advantage).mean()


def mean_kl(policy, qtype, old_probs):
    """Eq. 4's left side: mean over the batch of KL(pi_theta_old || pi_theta).

    KL compares the WHOLE action distribution at each question (both "answer"
    and "tool"), not just the action that was taken:
        KL(P || Q) = sum_a P(a) * (log P(a) - log Q(a)),   P = old, Q = new.
    0 means the policy's behaviour has not changed at all.
    """
    new_logp = torch.log_softmax(policy.logits[qtype], dim=-1)   # log pi_theta(. | s_t)
    return (old_probs * (old_probs.log() - new_logp)).sum(dim=-1).mean()


def trpo_update(policy, batch, max_kl=0.01, lr=0.3, epochs=50):
    """Maximize eq. 3 subject to eq. 4, on one freshly collected batch.

    Takes up to `epochs` gradient steps on eq. 3, all on this same batch. After
    each step it checks the fence: if the KL from theta_old is now above
    max_kl, that step is undone and the update ends. Returns how many steps
    (epochs) were kept -- 0 if even the first step would leave the fence, in
    which case this batch changes nothing.

    max_kl   delta in eq. 4: the most the policy's behaviour may change per batch
    lr       the size of each small step (the same lr as VPG, for a fair comparison)
    epochs   an upper limit. Early in training the fence stops the loop first;
             near the best policy the gradient is tiny, so the steps are too and
             the limit is reached instead.
    """
    qtype = batch[0]
    old_probs = policy.probs()[qtype]           # pi_theta_old(. | s_t), frozen: the centre of the fence
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    kept = 0
    for _ in range(epochs):
        before = policy.logits.detach().clone()   # where we were, in case this step leaves the fence

        loss = -surrogate(policy, *batch)         # eq. 3, negated: optimizers minimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                          # one small step, on the same batch

        # The fence (eq. 4). It is not in the loss, so it has to be checked here.
        if mean_kl(policy, qtype, old_probs) > max_kl:
            with torch.no_grad():
                policy.logits.copy_(before)       # outside the trust region: undo this step
            break                                 # and stop -- this batch has been used up
        kept += 1
    return kept
