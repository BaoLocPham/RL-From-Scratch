"""Fit tiny categorical policies from chosen/rejected pairs using DPO."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from dpo import dpo_loss, preference_accuracy
else:
    sys.path.insert(0, str(HERE))
    from common import dpo_loss, preference_accuracy

torch.manual_seed(23)
policy_logits = torch.zeros(3, 3, requires_grad=True)
reference_logits = torch.zeros_like(policy_logits)
contexts = torch.tensor([0, 0, 1, 1, 2, 2])
chosen = torch.tensor([2, 2, 0, 0, 1, 1])
rejected = torch.tensor([0, 1, 1, 2, 0, 2])
optimizer = torch.optim.Adam([policy_logits], lr=0.08)

reference_logp = reference_logits.log_softmax(-1)
ref_chosen = reference_logp[contexts, chosen]
ref_rejected = reference_logp[contexts, rejected]
for _ in range(100):
    policy_logp = policy_logits.log_softmax(-1)
    pi_chosen = policy_logp[contexts, chosen]
    pi_rejected = policy_logp[contexts, rejected]
    loss = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

with torch.no_grad():
    policy_logp = policy_logits.log_softmax(-1)
    accuracy = preference_accuracy(policy_logp[contexts, chosen],
                                   policy_logp[contexts, rejected],
                                   ref_chosen, ref_rejected)
print("final response probabilities (preferred responses are 2, 0, 1):")
for i, row in enumerate(policy_logits.softmax(-1).detach()):
    print(f"prompt {i}: {[round(value, 4) for value in row.tolist()]}")
print(f"pairwise preference accuracy: {accuracy.item():.3f}")
