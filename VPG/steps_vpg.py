"""Vanilla policy gradient, one step at a time: ``python VPG/steps_vpg.py``.

Walks the notes' running example through every piece of vanilla PG, printing
each equation and the numbers substituted into it. Set ``RL_IMPL=scratch`` to
run the same walkthrough on your VPG/from_scratch/vpg.py, and
``./scripts/run_vpg.sh diff`` to compare the two outputs line by line.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
else:
    sys.path.insert(0, str(HERE))                      # the reference
from vpg import (ANSWER, EASY, HARD, MEAN_REWARD, TOOL, Policy,  # noqa: E402
                 compute_advantage, expected_reward, pg_loss, vpg_update)


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# The running example from the notes: 10 HARD questions answered by the model at
# p(tool) = 0.40. 4 called the tool and scored 3.5; 6 answered directly and scored -1.5.
QTYPE = torch.zeros(10, dtype=torch.long)
ACTION = torch.tensor([TOOL] * 4 + [ANSWER] * 6)
REWARD = torch.tensor([3.5] * 4 + [-1.5] * 6)
LR = 0.1

# ---------------------------------------------------------------- step 1
banner("STEP 1  the goal: J(theta) = E[R], the policy's expected reward")
start = Policy().probs()
for t, name in ((HARD, "HARD"), (EASY, "EASY")):
    p_ans, p_tool = start[t].tolist()
    r_ans, r_tool = MEAN_REWARD[t].tolist()
    print(f"  {name}: {p_ans:.1f} * {r_ans:.1f} + {p_tool:.1f} * {r_tool:.1f} = "
          f"{p_ans * r_ans + p_tool * r_tool:.2f}")
print(f"  J(start) = expected_reward(start probs) = {float(expected_reward(start)):.3f}")
best = torch.tensor([[0., 1.], [1., 0.]])
print(f"  J(best)  = expected_reward(always tool on HARD, answer on EASY) = {float(expected_reward(best)):.3f}")
print("  Training never sees J. It only sees rollouts, like the 10 below.")

# ---------------------------------------------------------------- step 2
banner("STEP 2  the advantage: A_t = reward_t - mean(reward)")
advantage = compute_advantage(REWARD)
print(f"  rewards: 4 tool calls at {REWARD[0]:.1f}, 6 direct answers at {REWARD[-1]:.1f}")
print(f"  mean reward = (4 * {REWARD[0]:.1f} + 6 * {REWARD[-1]:.1f}) / 10 = {REWARD.mean():.2f}")
print(f"  A(tool)   = {REWARD[0]:.1f} - {REWARD.mean():.2f} = {advantage[0]:+.2f}   (better than average)")
print(f"  A(answer) = {REWARD[-1]:.1f} - {REWARD.mean():.2f} = {advantage[-1]:+.2f}   (worse than average)")
old_logp = Policy().dist(QTYPE).log_prob(ACTION).detach()
batch = (QTYPE, ACTION, advantage, old_logp)

# ---------------------------------------------------------------- step 3
banner("STEP 3  the loss: -L^PG = -mean( log pi(a_t|s_t) * A_t )     (eq. 2)")
policy = Policy()
p_tool = policy.probs()[HARD, TOOL]
print(f"  tool calls:     log({p_tool:.2f}) = {p_tool.log():+.3f},  * A {advantage[0]:+.0f} = "
      f"{p_tool.log() * advantage[0]:+.3f}   (x4)")
print(f"  direct answers: log({1 - p_tool:.2f}) = {(1 - p_tool).log():+.3f},  * A {advantage[-1]:+.0f} = "
      f"{(1 - p_tool).log() * advantage[-1]:+.3f}   (x6)")
loss = pg_loss(policy, *batch)
print(f"  loss = pg_loss(batch) = {float(loss.detach()):+.4f}   (the value means nothing; its gradient does)")

# ---------------------------------------------------------------- step 4
banner(f"STEP 4  one update: theta <- theta - lr * grad(loss),  lr = {LR}")
loss.backward()
before = policy.logits.detach()[HARD].clone()
grad = policy.logits.grad[HARD]
print(f"  grad(loss) on HARD's logits [answer, tool] = [{grad[0]:+.3f}, {grad[1]:+.3f}]")
policy = Policy()
optimizer = torch.optim.SGD(policy.parameters(), lr=LR)
vpg_update(policy, optimizer, batch)
after = policy.logits.detach()[HARD]
print(f"  logits: [{before[0]:+.3f}, {before[1]:+.3f}] -> [{after[0]:+.3f}, {after[1]:+.3f}]")
print(f"  p(tool|HARD): {p_tool:.3f} -> {policy.probs()[HARD, TOOL]:.3f}   "
      "(the tool calls had A > 0, so they got more likely)")
print(f"  J: {float(expected_reward(start)):.3f} -> {float(expected_reward(policy.probs())):.3f}")
print("  This update was valid: the batch came from exactly this model (theta_old).")

# ---------------------------------------------------------------- step 5
banner("STEP 5  keep updating on the SAME batch: the data goes stale")
print("  The batch still says the model called the tool 4 times in 10 (p = 0.40).")
print(f"  {'update':>6} | {'model p(tool|HARD)':>18} | {'batch still says':>16}")
print(f"  {1:>6} | {policy.probs()[HARD, TOOL]:>18.3f} | {0.4:>16.2f}")
for update in range(2, 21):
    vpg_update(policy, optimizer, batch)
    if update in (2, 5, 10, 20):
        print(f"  {update:>6} | {policy.probs()[HARD, TOOL]:>18.3f} | {0.4:>16.2f}")
print("  From update 2 the model has moved but the rollouts have not. L^PG has no")
print("  memory of theta_old, so it keeps pushing on the same 4 tool calls as if")
print("  they were new evidence. ./scripts/run_vpg.sh run shows where that ends up.")
