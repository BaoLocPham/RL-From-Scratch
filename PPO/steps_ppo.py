"""PPO one tensor at a time: ``python PPO/steps_ppo.py``."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from ppo import *  # noqa: F403
else:
    sys.path.insert(0, str(HERE))
    from common import *  # noqa: F403

torch.set_printoptions(precision=4, sci_mode=False)

print("0. where PPO rewards come from")
reward_table = torch.tensor([0.0, 0.4, 1.0])
actions_taken = torch.tensor([2, 1, 2, 0])
episode_rewards = reward_table[actions_taken].unsqueeze(0)
episode_dones = torch.tensor([[False, False, False, True]])
print("environment payout by action:", reward_table)
print("actions taken:", actions_taken)
print("rewards returned by the environment:", episode_rewards)
print("discounted returns:", discounted_returns(  # noqa: F405
    episode_rewards, episode_dones, gamma=0.9))

print("\n1. returns and GAE from a literal trajectory")
rewards = torch.tensor([[1.0, 0.5, 2.0, 9.0]])
dones = torch.tensor([[False, False, True, True]])
print("rewards:           ", rewards)
print("terminal flags:    ", dones)
print("discounted returns:", discounted_returns(rewards, dones, gamma=0.9))  # noqa: F405

values = torch.tensor([[0.2, 0.4, 0.1, 0.0, 0.0]])
advantage, targets = generalized_advantage_estimate(  # noqa: F405
    rewards, values, dones, gamma=0.9, lam=0.8)
print("\nvalues (+ bootstrap):", values)
print("GAE advantage:       ", advantage)
print("value targets:       ", targets)

old_logp = torch.zeros(2, 3)
ratios = torch.tensor([[1.3, 1.0, 0.7], [0.8, 1.2, 1.4]])
adv = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
print("\npolicy ratios:\n", ratios)
print("mask:\n", mask)
print("clipped policy loss:", clipped_policy_loss(old_logp, ratios.log(), adv, mask))  # noqa: F405

action_logits = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [9.0, -9.0]],
                              [[1.0, 1.0], [2.0, -1.0], [0.0, 0.0]]])
print("masked categorical entropy:", categorical_entropy(action_logits, mask))  # noqa: F405
