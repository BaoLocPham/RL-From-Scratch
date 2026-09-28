"""The candidates for the slot: PPO paper §3-§4, eq. 5-8. Demo: ``python Surrogates/run_surrogates.py``.

The paper does not present PPO as one mechanism. It presents one training loop
(§5, Algorithm 1, in loop.py) with a slot for the loss, and several candidates
for that slot, compared in §6.1. In the order the Notion page climbs them:

    L^PG     eq. 2   log pi * A                          VPG: valid for 1 epoch only (loop.pg_slot)
    L^CPI    eq. 6   ratio * A                           reuses the batch, but nothing stops the ratio
    TRPO     eq. 3-4 L^CPI + a rollback after each step  needs AFTER_STEP: surgery on the loop
    KL fixed eq. 5   L^CPI - beta * KL                   one loss, but beta is a price, not a distance
    KL adapt eq. 8   eq. 5, beta re-priced to hit d_targ needs AFTER_ITERATION
    L^CLIP   eq. 7   min(ratio * A, clip(ratio) * A)     needs nothing but the slot

The paper's ablation picks L^CLIP, and that choice is what gets the name:
PPO = this loop + L^CLIP (+ a value loss and an entropy bonus, eq. 9, + GAE).
PPO/ has it as verl writes it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop import mean_kl  # noqa: E402


def ratio_of(policy, batch):
    """r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t), one per rollout."""
    qtype, action, _, old_logp = batch
    return torch.exp(policy.dist(qtype).log_prob(action) - old_logp)


def cpi_loss(policy, batch, old_probs):
    """-L^CPI (eq. 6): the ratio surrogate, with no limit at all."""
    advantage = batch[2]
    return -(ratio_of(policy, batch) * advantage).mean()


def kl_penalty_loss(policy, batch, old_probs, beta):
    """-(L^CPI - beta * KL) (eq. 5): moving is allowed, but every nat of KL costs beta."""
    advantage = batch[2]
    kl = mean_kl(policy, batch[0], old_probs)
    return -((ratio_of(policy, batch) * advantage).mean() - beta * kl)


def adapt_beta(beta, d, d_targ):
    """The paper's §4 rule, run after each iteration with d = the KL it actually moved.

    Moved too little (d < d_targ / 1.5): halve beta, moving gets cheaper.
    Moved too far    (d > d_targ * 1.5): double beta, moving gets dearer.
    Otherwise keep it. The 1.5 and 2 are the paper's heuristics.
    """
    if d < d_targ / 1.5:
        return beta / 2
    if d > d_targ * 1.5:
        return beta * 2
    return beta


def clip_loss(policy, batch, old_probs, eps=0.2):
    """-L^CLIP (eq. 7): no extra credit once the ratio has moved eps the way A wants."""
    advantage = batch[2]
    ratio = ratio_of(policy, batch)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    return -torch.min(unclipped, clipped).mean()


class AdaptiveKL:
    """L^KLPEN (eq. 8): the fixed-beta slot plus an AFTER_ITERATION hook that re-prices beta.

    One object, because the slot and the hook share beta. Use a fresh one per run.
    """

    def __init__(self, d_targ=0.01, beta=1.0):
        self.d_targ, self.beta = d_targ, beta

    def slot(self, policy, batch, old_probs):
        return kl_penalty_loss(policy, batch, old_probs, self.beta)

    def after_iteration(self, policy, batch, old_probs):
        d = float(mean_kl(policy, batch[0], old_probs).detach())   # how far this iteration moved
        self.beta = adapt_beta(self.beta, d, self.d_targ)
