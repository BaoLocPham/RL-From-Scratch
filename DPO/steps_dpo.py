"""DPO one tensor at a time: ``python DPO/steps_dpo.py``."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from dpo import *  # noqa: F403
else:
    sys.path.insert(0, str(HERE))
    from common import *  # noqa: F403

torch.set_printoptions(precision=4, sci_mode=False)


def verifiable_reward(completion, gold):
    return 1.0 if completion.strip() == gold.strip() else 0.0


print("0. where DPO chosen/rejected pairs come from")
gold = "4"
sample_a, sample_b = "4", "5"
if verifiable_reward(sample_a, gold) >= verifiable_reward(sample_b, gold):
    chosen, rejected = sample_a, sample_b
else:
    chosen, rejected = sample_b, sample_a
print("samples:", [sample_a, sample_b], "gold:", gold)
print("chosen:", chosen, "rejected:", rejected)
print("The pair is next scored under both policy and frozen reference.")

print("\n1. whole-completion log-probabilities")
logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [9.0, -9.0, 0.0]],
                       [[0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [1.0, 1.0, 1.0]]])
tokens = torch.tensor([[0, 1, 2], [2, 0, 1]])
mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
sequence_scores = sequence_log_probs(logits, tokens, mask)  # noqa: F405
print("token ids:\n", tokens)
print("mask:\n", mask)
print("summed sequence log-probabilities:", sequence_scores)

policy_chosen = torch.tensor([-1.0, -2.0, -0.8])
policy_rejected = torch.tensor([-2.0, -1.5, -1.4])
reference_chosen = torch.tensor([-1.4, -1.8, -1.0])
reference_rejected = torch.tensor([-1.8, -1.6, -1.1])
margin = preference_logit(policy_chosen, policy_rejected,  # noqa: F405
                          reference_chosen, reference_rejected, beta=0.2)
print("\nreference-corrected preference logits:", margin)
print("DPO loss:", dpo_loss(policy_chosen, policy_rejected,  # noqa: F405
                            reference_chosen, reference_rejected, beta=0.2))
print("preference accuracy:", preference_accuracy(  # noqa: F405
    policy_chosen, policy_rejected, reference_chosen, reference_rejected))
