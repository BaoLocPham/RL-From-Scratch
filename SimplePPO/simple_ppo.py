"""PPO, the paper's algorithm, on the multi-step toy. Demo: ``python SimplePPO/run_simple_ppo.py``.

Surrogates/ ended with: PPO = the loop with L^CLIP in the slot, plus a value
loss, an entropy bonus and GAE. This file is exactly that, and nothing else --
no tokens, no masks, no verl extras:

    for iteration:
        batch = rollout(policy, critic)                   theta_old: episodes of 3 turns
        advantages, returns = compute_gae(batch)          once per batch             (eq. 11-12)
        for epoch in range(K):
            for minibatch of M steps:
                loss = -L^CLIP + c1 * L^VF - c2 * S       eq. 9, negated for descent
                zero_grad, backward, step

The one new idea since Surrogates is the critic. Rewards now arrive over
several turns, so each turn needs its own advantage -- GAE builds it from the
critic's predictions. Once every step has an advantage, the steps are
flattened into one list of samples and the rest is Surrogates' loop.

The toy (Policy, Critic, rollout) is in env.py.
"""

import torch


def compute_gae(rewards, values, gamma, lam):
    """Advantages and returns for (episodes, turns) tensors, by GAE (paper eq. 11-12).

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          the critic's surprise at turn t
        A_t     = delta_t + gamma * lam * A_{t+1}            walked backwards from the last turn

    Every episode has the same length, so there is no mask: past the last turn
    nothing follows, and V = 0 there.
    """
    with torch.no_grad():                                   # targets, computed once per batch
        nextvalues = 0                                      # V(s_{t+1}); 0 past the last turn
        lastgaelam = 0                                      # A_{t+1};    0 past the last turn
        advantages_reversed = []
        for t in reversed(range(rewards.shape[1])):         # right to left: A_t needs A_{t+1}
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t]
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values                       # the critic's target
    return advantages, returns


def policy_loss(policy, states, actions, old_logp, advantages, eps=0.2):
    """-L^CLIP (eq. 7): Surrogates' clip_loss, with a state per sample instead of a question type."""
    ratio = torch.exp(policy.dist(states).log_prob(actions) - old_logp)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def value_loss(critic, states, returns):
    """L^VF (eq. 9): 0.5 * (V(s_t) - returns_t)^2, averaged."""
    return 0.5 * ((critic(states) - returns) ** 2).mean()


def entropy_bonus(policy, states):
    """S (eq. 9): the policy's average entropy. Returned positive; the loss subtracts it."""
    return policy.dist(states).entropy().mean()


def ppo_update(policy, critic, optimizer, batch, epochs=10, minibatch_size=16, eps=0.2,
               vf_coef=0.5, entropy_coeff=0.01):
    """Algorithm 1's inner loop: `epochs` passes over ONE batch, in minibatches, on eq. 9.

    Returns each loss term averaged over every update.
    """
    # After GAE, time no longer matters: every step is one sample. Flatten (episodes, turns) -> (steps,).
    steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "advantages", "returns")}
    n = steps["states"].shape[0]
    totals, updates = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                           # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            pg = policy_loss(policy, steps["states"][idx], steps["actions"][idx], steps["old_logp"][idx],
                             steps["advantages"][idx], eps)
            vf = value_loss(critic, steps["states"][idx], steps["returns"][idx])
            ent = entropy_bonus(policy, steps["states"][idx])
            loss = pg + vf_coef * vf - entropy_coeff * ent  # eq. 9, negated

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1
            for name, value in (("pg_loss", pg), ("vf_loss", vf), ("entropy", ent)):
                totals[name] += float(value.detach())
    return {name: total / updates for name, total in totals.items()}
