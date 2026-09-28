"""The loop with a slot: PPO paper §5, Algorithm 1, on the VPG toy.

    for iteration:
        batch = collect_rollouts(policy)                  theta_old, frozen from here on
        old_probs = pi_theta_old(. | s) for the batch     taken ONCE per iteration
        for epoch in range(K):
            loss = SLOT(policy, batch, old_probs)         <- the only line the candidates change
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            AFTER_STEP (optional)                         TRPO's rollback lives here
        AFTER_ITERATION (optional)                        adaptive KL re-prices beta here

Every slot has the same signature, ``slot(policy, batch, old_probs) -> loss``:
    batch      (qtype, action, advantage, old_logp), from collect_rollouts
    old_probs  pi_theta_old(. | s_t) for every rollout, [batch, 2]
and returns a loss to MINIMISE, so each one is minus its objective.

This file is shared by the reference (surrogates.py) and your exercise
(from_scratch/surrogates.py): the loop is given, the slots are what you write.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VPG"))
sys.path.insert(0, str(ROOT / "TRPO"))
from trpo import mean_kl  # noqa: E402,F401  (re-exported: the slots use it)
from vpg import ANSWER, EASY, HARD, TOOL, Policy, collect_rollouts, pg_loss  # noqa: E402,F401

LR = 0.3                # SGD learning rate, the same as VPG/ and TRPO/
BATCH_SIZE = 16         # rollouts per iteration
EPOCHS = 50             # epochs per iteration for every slot that reuses the batch


def pg_slot(policy, batch, old_probs):
    """-L^PG (eq. 2), VPG's loss, as a slot. It ignores theta_old entirely."""
    return pg_loss(policy, *batch)


def kl_rollback(max_kl):
    """TRPO as an AFTER_STEP hook: undo the step that leaves the trust region, and stop.

    The slot for TRPO is plain L^CPI. What makes it TRPO is this hook, which has
    to run after every optimizer step -- surgery on the loop, not just the slot.
    """
    def after_step(policy, batch, old_probs, before):
        if mean_kl(policy, batch[0], old_probs) > max_kl:
            with torch.no_grad():
                policy.logits.copy_(before)
            return True                                    # stop this iteration's epochs
        return False
    return after_step


def train(slot, seed, iterations, epochs=EPOCHS, after_step=None, after_iteration=None,
          lr=LR, n=BATCH_SIZE, advantage_scale=1.0, optim=torch.optim.SGD):
    """Algorithm 1 with `slot` in it.

    Returns one row per iteration, (true J, KL moved from theta_old, epochs kept),
    and the final action probabilities.

    advantage_scale multiplies every advantage, as if rewards were measured in
    other units: the best policy is unchanged, only the numbers the loss sees grow.
    """
    torch.manual_seed(seed)                                # same seed -> same first batch for every slot
    policy = Policy()
    optimizer = optim(policy.parameters(), lr=lr)
    rows = []
    for _ in range(iterations):
        qtype, action, advantage, old_logp = collect_rollouts(policy, n)
        batch = (qtype, action, advantage * advantage_scale, old_logp)
        old_probs = policy.probs()[qtype]                  # theta_old: taken ONCE per iteration
        kept = 0
        for _ in range(epochs):
            before = policy.logits.detach().clone()        # for a hook that needs to undo this step
            loss = slot(policy, batch, old_probs)          # <- the slot
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if after_step is not None and after_step(policy, batch, old_probs, before):
                break
            kept += 1
        moved = float(mean_kl(policy, qtype, old_probs).detach())
        if after_iteration is not None:
            after_iteration(policy, batch, old_probs)
        rows.append((policy.true_reward(), moved, kept))
    return rows, policy.probs()


def is_stuck(probs):
    """Locked onto a wrong action: p(tool|HARD) near 0 or p(tool|EASY) near 1.

    A probability near 0 is almost never sampled again, so no later batch can
    bring the evidence to undo it.
    """
    return bool(probs[HARD, TOOL] < 0.05 or probs[EASY, TOOL] > 0.95)
