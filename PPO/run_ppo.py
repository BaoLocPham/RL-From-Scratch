"""Train a tiny policy with verl's PPO: ``python PPO/run_ppo.py``.

A miniature of a real language-model rollout, at verl's shapes. The "model" is
a table of logits, one row per response position, over a three-token vocabulary.
A response is four tokens; the reward lands on the last real one and says how
many positions matched a hidden target. A value head predicts it.

No transformer, but the tensors, the mask semantics and every function called
are the ones a verl PPO trainer uses.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from ppo import (compute_entropy_loss, compute_gae_advantage_return,  # noqa: E402
                     compute_policy_loss, compute_rewards, compute_value_loss)
else:
    sys.path.insert(0, str(HERE))
    from common import (compute_entropy_loss, compute_gae_advantage_return,  # noqa: E402
                        compute_policy_loss, compute_rewards, compute_value_loss)

torch.manual_seed(0)

RESPONSE_LENGTH = 4
VOCAB = 3
BATCH = 32
PPO_EPOCHS = 4        # gradient passes over each rollout; verl's actor.ppo_epochs
TARGET = torch.tensor([2, 0, 1, 2])          # the hidden answer, never shown
LENGTHS = torch.tensor([4, 3])               # two response lengths, to exercise the mask

policy_logits = torch.zeros(RESPONSE_LENGTH, VOCAB, requires_grad=True)
value_head = torch.zeros(RESPONSE_LENGTH, requires_grad=True)
reference_logits = torch.zeros(RESPONSE_LENGTH, VOCAB)   # frozen, for the KL term
optimizer = torch.optim.Adam([policy_logits, value_head], lr=0.1)

# Rows alternate between the two lengths; the mask is what makes them differ.
lengths = LENGTHS.repeat(BATCH // 2)
response_mask = (torch.arange(RESPONSE_LENGTH)[None, :] < lengths[:, None]).float()

print("verl-shaped PPO. response_mask has two distinct lengths:")
print(" ", response_mask[0].tolist(), "and", response_mask[1].tolist())
print("Only positions inside the mask are scored, credited or learned from.\n")
print(f"{'step':>5}{'reward':>9}{'pg_loss':>10}{'vf_loss':>9}{'entropy':>9}"
      f"{'clipfrac':>10}{'ppo_kl':>9}")

for step in range(60):
    with torch.no_grad():
        old_logits = policy_logits.detach().clone()
        probs = old_logits.softmax(-1).expand(BATCH, -1, -1)
        tokens = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(BATCH, RESPONSE_LENGTH)

        # Outcome reward: how many in-mask positions matched, parked on the last
        # real token exactly as an outcome-supervised verl batch would carry it.
        correct = ((tokens == TARGET) * response_mask).sum(-1) / lengths
        token_level_scores = torch.zeros(BATCH, RESPONSE_LENGTH)
        token_level_scores[torch.arange(BATCH), lengths - 1] = correct

        old_log_prob = old_logits.log_softmax(-1).expand(BATCH, -1, -1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_log_prob = reference_logits.log_softmax(-1).expand(BATCH, -1, -1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        # PPO puts the KL inside the reward, before GAE sees it.
        token_level_rewards = compute_rewards(token_level_scores, old_log_prob,
                                              ref_log_prob, kl_ratio=0.02)
        old_values = value_head.expand(BATCH, -1) * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, old_values, response_mask, gamma=1.0, lam=0.95)

    # PPO_EPOCHS passes over the SAME rollout. This is the point of the ratio:
    # on the first pass it is exactly 1 and the clip cannot bind, and only as
    # the policy drifts away from the one that sampled does clipping matter.
    # verl calls this actor.ppo_epochs.
    for epoch in range(PPO_EPOCHS):
        log_prob = policy_logits.log_softmax(-1).expand(BATCH, -1, -1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        vpreds = value_head.expand(BATCH, -1)

        pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask, cliprange=0.2,
            loss_agg_mode="token-mean")
        vf_loss, _ = compute_value_loss(vpreds, returns, old_values, response_mask,
                                        cliprange_value=0.2)
        entropy = compute_entropy_loss(policy_logits.expand(BATCH, -1, -1), response_mask)

        # Entropy is SUBTRACTED: maximizing it preserves exploration.
        loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % 10 == 0 or step == 59:
        print(f"{step:>5}{correct.mean():>9.3f}{pg_loss.item():>10.4f}"
              f"{vf_loss.item():>9.4f}{entropy.item():>9.4f}"
              f"{pg_clipfrac.item():>10.2f}{ppo_kl.item():>9.4f}")

print("\nlearned token distribution per position (target is 2, 0, 1, 2):")
for position, row in enumerate(policy_logits.softmax(-1).detach()):
    marker = "" if position < LENGTHS.min() else "   <- outside the short mask"
    print(f"  position {position}: {[round(v, 3) for v in row.tolist()]}"
          f"  argmax={int(row.argmax())}{marker}")
print("\nPosition 3 is inside the mask for only half the batch, so it learns from")
print("half as many samples -- it still gets there, just on less evidence.")
print("\nWatch pg_clipfrac and ppo_kl in the table above: both start at zero,")
print("because on the first of the 4 passes over a rollout the ratio is exactly")
print("1 and nothing can clip. They rise as the policy drifts away from the one")
print("that sampled, which is the drift the clip exists to bound.")
