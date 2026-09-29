"""GRPO, the DeepSeekMath algorithm, on the multi-step toy. Demo: ``python SimpleGRPO/run_simple_grpo.py``.

SimplePPO ended with a critic that barely paid for itself on this toy. GRPO
(DeepSeekMath, §4.1) drops it. Instead of asking a critic how good a state is,
it answers the same question several times and compares the answers with
each other:

    for iteration:
        batch = rollout(policy, G attempts per question)      theta_old
        advantages = group_advantage(batch)                   once per batch: no critic, no GAE
        for epoch in range(K):
            for minibatch of M steps:
                loss = policy_loss + beta * kl_penalty        SimplePPO's clip, plus a KL to pi_ref
                zero_grad, backward, step

Two changes from SimplePPO, and nothing else:

    advantage   (episode reward - its group's mean) / its group's std,
                the same number for every turn of the episode
    the loss    no value loss, no entropy bonus; a KL penalty to the frozen
                starting policy pi_ref instead

The clip is SimplePPO's policy_loss, imported, not rewritten. The toy is in group_env.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SimplePPO"))
from simple_ppo import policy_loss  # noqa: E402  (the clip, unchanged)


def group_advantage(rewards, group_size, eps=1e-6, scale_by_std=True):
    """GRPO's advantage for (episodes, turns) rewards, stored group by group (DeepSeekMath §4.1.2).

        A_i = (R_i - mean(R_group)) / std(R_group)       R_i: episode i's total reward

    Every turn of episode i gets the same A_i: outcome supervision says how the
    episode went, not which turn did it. `scale_by_std=False` skips the divide,
    which is Dr.GRPO.
    """
    with torch.no_grad():
        scores = rewards.sum(1)                             # one number per episode: its total reward
        groups = scores.view(-1, group_size)                # (questions, G): the attempts at one question in a row
        centred = groups - groups.mean(1, keepdim=True)     # better or worse than this question's average
        if scale_by_std:
            centred = centred / (groups.std(1, keepdim=True) + eps)    # torch.std: the sample (n - 1) std
        return centred.reshape(-1, 1).expand_as(rewards)    # the same advantage on every turn


def kl_penalty(policy, states, actions, ref_logp):
    """KL(pi_theta || pi_ref) by DeepSeekMath's estimator (eq. 4), averaged over the steps.

        pi_ref / pi_theta - log(pi_ref / pi_theta) - 1       on the action taken: k3

    Never negative, and 0 exactly where pi_theta = pi_ref.
    """
    log_ratio = ref_logp - policy.dist(states).log_prob(actions)    # log(pi_ref / pi_theta)
    return (torch.exp(log_ratio) - log_ratio - 1).mean()


def grpo_update(policy, optimizer, batch, epochs=10, minibatch_size=16, eps=0.2, beta=0.04):
    """The inner loop: `epochs` passes over ONE batch, in minibatches, on the clip plus beta * KL.

    Returns each loss term averaged over every update.
    """
    # The advantage is set per episode, so time no longer matters: every step is one sample.
    steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "ref_logp", "advantages")}
    n = steps["states"].shape[0]
    totals, updates = {"pg_loss": 0.0, "kl": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                           # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            pg = policy_loss(policy, steps["states"][idx], steps["actions"][idx], steps["old_logp"][idx],
                             steps["advantages"][idx], eps)
            kl = kl_penalty(policy, steps["states"][idx], steps["actions"][idx], steps["ref_logp"][idx])
            loss = pg + beta * kl                           # DeepSeekMath eq. 3, negated

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1
            for name, value in (("pg_loss", pg), ("kl", kl)):
                totals[name] += float(value.detach())
    return {name: total / updates for name, total in totals.items()}
