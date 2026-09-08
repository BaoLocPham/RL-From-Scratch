"""GRPO one tensor at a time: ``python GRPO/steps_grpo.py``."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(ROOT / "from_scratch"))
    from grpo import clipped_surrogate_loss, group_relative_advantage, kl_penalty_k3
else:
    sys.path.insert(0, str(ROOT))
    from common import clipped_surrogate_loss, group_relative_advantage, kl_penalty_k3

torch.set_printoptions(precision=4, sci_mode=False)


def verifiable_reward(completion, gold):
    """The simplest real GRPO reward: exact-match correctness."""
    return 1.0 if completion.strip() == gold.strip() else 0.0


print("0. where GRPO rewards come from")
gold = ["4", "4", "4", "9", "9", "9"]
completions = ["4", "5", "4", "9", "9", "10"]
rewards = torch.tensor([verifiable_reward(completion, answer)
                        for completion, answer in zip(completions, gold)])
groups = torch.tensor([0, 0, 0, 1, 1, 1])
print("completions:", completions)
print("exact-match rewards:", rewards)
print("group-relative advantages:", group_relative_advantage(rewards, groups))

print("\n1. prompt-local normalization with differently scaled rewards")
rewards = torch.tensor([0.9, 1.0, 1.1, -5.0, 0.0, 5.0])
groups = torch.tensor([0, 0, 0, 1, 1, 1])
adv = group_relative_advantage(rewards, groups)
print("rewards:  ", rewards)
print("groups:   ", groups)
print("advantage:", adv)
for g in (0, 1):
    a = adv[groups == g]
    print(f"group {g}: mean={a.mean():.4f}, sample std={a.std():.4f}")

print("\n2. clipped policy update over a ragged completion batch")
old_logp = torch.zeros(3, 4)
ratio = torch.tensor([[1.3, 1.0, 0.7, 1.0],
                      [0.8, 1.2, 1.4, 1.0],
                      [1.1, 0.9, 1.0, 1.0]])
logp = ratio.log()
token_adv = torch.tensor([1.0, -1.0, 0.5])[:, None].expand_as(logp)
mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
print("\nratio:\n", ratio)
print("broadcast advantage:\n", token_adv)
print("mask:\n", mask)
print("clipped loss:", clipped_surrogate_loss(old_logp, logp, torch.tensor([1.0, -1.0, 0.5]), mask))

ref_logp = logp + torch.tensor([[0.05, -0.05, 0.10, 0.00],
                                [-0.10, 0.05, 0.00, 0.00],
                                [0.02, -0.02, 0.05, -0.05]])
per_token_kl = kl_penalty_k3(ref_logp, logp)
print("\n3. k3 KL penalty against a nearby frozen reference")
print("per-token KL (small and non-negative):\n", per_token_kl)
print("clipped + 0.05 * KL loss:",
      clipped_surrogate_loss(old_logp, logp, torch.tensor([1.0, -1.0, 0.5]),
                             mask, ref_logp=ref_logp, beta=0.05))
