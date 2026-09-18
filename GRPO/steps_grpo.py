"""verl's GRPO one tensor at a time: ``python GRPO/steps_grpo.py``."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from grpo import (agg_loss, compute_grpo_outcome_advantage,  # noqa: E402
                      compute_policy_loss, kl_penalty)
else:
    sys.path.insert(0, str(HERE))
    from common import (agg_loss, compute_grpo_outcome_advantage,  # noqa: E402
                        compute_policy_loss, kl_penalty)

torch.set_printoptions(precision=4, sci_mode=False)


def verifiable_reward(completion, gold):
    """The simplest real GRPO reward: exact-match correctness."""
    return 1.0 if completion.strip() == gold.strip() else 0.0


print("0. where the reward comes from, and how it is shaped")
completions = ["4", "5", "4", "9", "9", "10"]
gold = ["4", "4", "4", "9", "9", "9"]
index = np.array(["p0", "p0", "p0", "p1", "p1", "p1"])
scalar = [verifiable_reward(c, g) for c, g in zip(completions, gold)]
print("completions:", completions)
print("scalar rewards:", scalar)
token_level_rewards = torch.zeros(6, 4)
response_mask = torch.tensor([[1., 1., 1., 0.]] * 6)
token_level_rewards[:, 2] = torch.tensor(scalar)   # credited at the last real token
print("token_level_rewards (one scalar parked in a row):\n", token_level_rewards)
print("index:", index.tolist())
print("Rows sharing an index came from the same prompt. That is the group.")

print("\n1. the group replaces the critic")
advantages, returns = compute_grpo_outcome_advantage(token_level_rewards.clone(),
                                                     response_mask, index)
print("advantages:\n", advantages)
print("PPO would need a value head, GAE and a value loss to get a baseline.")
print("GRPO uses the other samples of the same prompt. No critic, and")
print("no compute_gae_advantage_return or compute_value_loss anywhere.")
print("advantages is returns:", advantages is returns,
      "-- nothing to regress under outcome supervision")

print("\n2. the baseline is prompt-local, which is the whole point")
rewards = torch.zeros(6, 4)
rewards[:, 2] = torch.tensor([0.9, 1.0, 1.1, -5.0, 0.0, 5.0])
print("scalar rewards:", rewards[:, 2].tolist())
local = compute_grpo_outcome_advantage(rewards.clone(), response_mask, index)[0]
print("per-group advantages:", local[:, 0].tolist())
flat = rewards.sum(-1)
print("if standardized across the whole batch:",
      ((flat - flat.mean()) / (flat.std() + 1e-6)).tolist())
print("Group p0's rewards sit in a narrow band and group p1's are wide. Judged")
print("globally, every p0 sample looks identical and the prompt's own signal is")
print("lost. Judged locally, p0 still ranks its three samples cleanly.")

print("\n3. the singleton special case")
alone = compute_grpo_outcome_advantage(
    torch.tensor([[0., 0., 7., 0.]]), torch.tensor([[1., 1., 1., 0.]]),
    np.array(["solo"]))[0]
print("a group of one, reward 7.0 ->", alone[0, :3].tolist())
print("mean 0 and std 1, not its own mean. Using its own mean would make the")
print("advantage identically zero and throw the sample away.")

print("\n4. one flag turns GRPO into Dr.GRPO")
grpo = compute_grpo_outcome_advantage(rewards.clone(), response_mask, index)[0]
drgrpo = compute_grpo_outcome_advantage(rewards.clone(), response_mask, index,
                                        norm_adv_by_std_in_grpo=False)[0]
print(f"{'reward':>8}{'GRPO':>10}{'Dr.GRPO':>10}")
for i in range(6):
    print(f"{rewards[i, 2]:>8.1f}{grpo[i, 0]:>10.4f}{drgrpo[i, 0]:>10.4f}")
print("GRPO divides by the group std, so a group that happened to disagree a lot")
print("produces smaller advantages than one that barely disagreed. That couples")
print("update size to group variance, which correlates with response length.")
print("Dr.GRPO drops the divide: norm_adv_by_std_in_grpo=False.")

print("\n5. one advantage per response, copied across its tokens")
print("advantages row 0:", grpo[0].tolist())
print("response_mask row 0:", response_mask[0].tolist())
print("Every token of a response carries that whole response's verdict, and")
print("padded positions are zeroed. There is no per-token credit assignment")
print("in outcome-supervised GRPO -- the reward never said which token was good.")

print("\n6. the update itself is PPO's, unchanged")
old_log_prob = torch.zeros(6, 4)
log_prob = torch.randn(6, 4, generator=torch.Generator().manual_seed(0)) * 0.1
pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
    old_log_prob, log_prob, grpo, response_mask, cliprange=0.2)
print(f"pg_loss={float(pg_loss):+.6f}  clipfrac={float(pg_clipfrac):.2f}"
      f"  ppo_kl={float(ppo_kl):+.4f}")
print("Same compute_policy_loss, same agg_loss, same clip. GRPO changes the")
print("advantage estimator and nothing else about the update.")

print("\n7. GRPO's KL sits in the loss, not in the reward")
ref_log_prob = log_prob + 0.05
per_token_kl = kl_penalty(log_prob, ref_log_prob, "k3")
beta = 0.04
total = pg_loss + beta * agg_loss(per_token_kl, response_mask, "token-mean")
print(f"per-token k3 KL (non-negative): {per_token_kl[0].tolist()}")
print(f"pg_loss {float(pg_loss):+.6f}  ->  + {beta} * KL  =  {float(total):+.6f}")
print("PPO instead subtracts its KL from the reward before GAE, so the penalty")
print("gets bootstrapped and discounted. Here it is just another loss term.")
