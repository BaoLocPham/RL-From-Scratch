"""Train tiny per-prompt categorical policies with GRPO."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(ROOT / "from_scratch"))
    from grpo import clipped_surrogate_loss, group_relative_advantage
else:
    sys.path.insert(0, str(ROOT))
    from common import clipped_surrogate_loss, group_relative_advantage

torch.manual_seed(7)
reward_table = torch.tensor([[0.0, 0.4, 1.0], [0.8, 0.1, -0.2], [0.2, 1.0, 0.3]])
logits = torch.zeros_like(reward_table, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.12)
group_size = 12

for step in range(80):
    old = logits.detach().clone()
    with torch.no_grad():
        actions = torch.multinomial(old.softmax(-1), group_size, replacement=True)  # (P,G)
        rewards = reward_table.gather(1, actions).reshape(-1)                        # (P*G,)
        groups = torch.arange(3).repeat_interleave(group_size)                       # (P*G,)
        adv = group_relative_advantage(rewards, groups)
        old_lp = old.log_softmax(-1).gather(1, actions).reshape(-1, 1)                # (N,1)
    lp = logits.log_softmax(-1).gather(1, actions).reshape(-1, 1)                     # (N,1)
    loss = clipped_surrogate_loss(old_lp, lp, adv, torch.ones_like(lp))
    opt.zero_grad(); loss.backward(); opt.step()

print("final action probabilities (best actions are 2, 0, 1):")
for i, row in enumerate(logits.softmax(-1).detach()):
    print(f"prompt {i}: {[round(x, 4) for x in row.tolist()]}")
