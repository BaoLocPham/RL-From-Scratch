"""Train three tiny contextual-bandit policies with repeated PPO epochs."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from ppo import normalize_advantage, ppo_loss
else:
    sys.path.insert(0, str(HERE))
    from common import normalize_advantage, ppo_loss

torch.manual_seed(17)
reward_table = torch.tensor([[0.0, 0.4, 1.0], [0.8, 0.1, -0.2], [0.2, 1.0, 0.3]])
policy_logits = torch.zeros_like(reward_table, requires_grad=True)
state_values = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.Adam([policy_logits, state_values], lr=0.08)
samples_per_context = 32

for iteration in range(70):
    with torch.no_grad():
        old_logits = policy_logits.detach().clone()
        old_value_table = state_values.detach().clone()
        actions = torch.multinomial(old_logits.softmax(-1), samples_per_context, replacement=True)
        contexts = torch.arange(3).repeat_interleave(samples_per_context)
        flat_actions = actions.reshape(-1)
        old_logp = old_logits.log_softmax(-1)[contexts, flat_actions]
        old_values = old_value_table[contexts]
        returns = reward_table[contexts, flat_actions]
        advantage = normalize_advantage(returns - old_values, torch.ones_like(returns))
    for _ in range(4):
        logits = policy_logits[contexts]
        logp = logits.log_softmax(-1).gather(1, flat_actions[:, None]).squeeze(1)
        values = state_values[contexts]
        mask = torch.ones_like(logp)
        loss = ppo_loss(old_logp, logp, advantage, values, old_values, returns,
                        logits, mask, entropy_coef=0.01)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

print("final action probabilities (best actions are 2, 0, 1):")
for i, row in enumerate(policy_logits.softmax(-1).detach()):
    print(f"context {i}: {[round(value, 4) for value in row.tolist()]}  value={state_values[i].item():.3f}")
