"""Simple GRPO, one step at a time: ``python SimpleGRPO/steps_simple_grpo.py``.

Follows one group of attempts at one question through every piece of GRPO,
printing each equation with the numbers substituted into it. Set
``RL_IMPL=scratch`` to run the same walkthrough on your
SimpleGRPO/from_scratch/simple_grpo.py, and ``./scripts/run_simple_grpo.sh diff``
to compare the two outputs line by line.
"""

import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
from simple_grpo import group_advantage, grpo_update, kl_penalty, policy_loss  # noqa: E402
import group_env as env  # noqa: E402


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 3) + 0.0                    # + 0.0: no -0.0


def row(t):
    return [f(v) for v in t]


# ---------------------------------------------------------------- step 1
banner("STEP 1  one question, answered four times: a group")
print("""  A HARD question (it needs two searches). The policy attempts it 4 times. Each search
  costs 0.1; the answer is graded right (+1) or wrong (-1), like a math checker.

  attempt | turn 0  turn 1  turn 2 | searches | answer | rewards per turn   | total R
     1    | SEARCH  SEARCH  SKIP   |    2     | right  | -0.1  -0.1   1.0   |   0.8
     2    | SKIP    SKIP    SKIP   |    0     | wrong  |  0.0   0.0  -1.0   |  -1.0
     3    | SEARCH  SKIP    SKIP   |    1     | wrong  | -0.1   0.0  -1.0   |  -1.1   (one search: a coin flip, lost)
     4    | SKIP    SEARCH  SEARCH |    2     | right  |  0.0  -0.1   0.9   |   0.8

  SimplePPO asked a critic "how good is this state?" at every turn. GRPO has no critic:
  it asks how each attempt did compared with the OTHER attempts at the same question.""")
rewards = torch.tensor([[-0.1, -0.1, 1.0], [0.0, 0.0, -1.0], [-0.1, 0.0, -1.0], [0.0, -0.1, 0.9]])

# ---------------------------------------------------------------- step 2
banner("STEP 2  the group advantage: A_i = (R_i - mean(R)) / std(R)")
totals = rewards.sum(1)
mean, std = float(totals.mean()), float(totals.std())
print(f"  totals R = {row(totals)}   (outcome supervision: one number per attempt)")
print(f"  mean = {mean:.3f}")
print(f"  std  = sqrt(sum((R - mean)^2) / (4 - 1)) = {std:.4f}   (the sample std, as torch.std)")
for i, r in enumerate(totals.tolist()):
    print(f"  A_{i + 1} = ({r:+.1f} - ({mean:.3f})) / {std:.4f} = {(r - mean) / std:+.3f}")
advantages = group_advantage(rewards, 4)
print(f"  group_advantage -> (attempts, turns) =")
for i in range(4):
    print(f"    attempt {i + 1}: {row(advantages[i])}")
print("""  Every turn of an attempt carries the same number. Attempt 1's final SKIP gets +0.865
  just like its two searches: the grade says how the attempt went, not which turn did it.
  SimplePPO's GAE gave the turns of one episode different advantages ([0.208, 0.26, 0.2]);
  here there is nothing to tell them apart. Over many attempts the credit still sorts
  itself out: turns that help show up more often in the attempts that did well.""")

# ---------------------------------------------------------------- step 3
banner("STEP 3  why a group, and not the whole batch")
easy = torch.tensor([[0.0, 0.0, 1.0], [-0.1, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, -0.1, 1.0]])
both = torch.cat([rewards, easy])
print(f"  add an EASY question's group: totals {row(easy.sum(1))}   (two wasted a search, 0.9)")
batch = both.sum(1) - both.sum(1).mean()
grouped = group_advantage(both, 4, scale_by_std=False)[:, 0]
print(f"  R - the batch's mean ({f(both.sum(1).mean())}):  HARD {row(batch[:4])}   EASY {row(batch[4:])}")
print(f"  R - its group's mean:              HARD {row(grouped[:4])}   EASY {row(grouped[4:])}")
print("""  Against the batch, every EASY attempt looks good -- even the ones that wasted a search
  -- only because EASY questions are easy. Against its own group, a wasted search is
  below average. The group mean is a baseline for THIS question: what a critic's V(s_0)
  would estimate, taken from the attempts themselves.""")

# ---------------------------------------------------------------- step 4
banner("STEP 4  the std divide, and Dr.GRPO (scale_by_std=False)")
pairs = torch.tensor([[-0.1, 0.0, 1.0], [-0.1, -0.1, 1.0], [-0.1, 0.0, -1.0], [-0.1, -0.1, 1.0]])
print("  groups of two, on a HARD question: one search vs two searches")
print(f"    lucky coin: 1 search right (0.9) vs 2 searches (0.8)  GRPO {row(group_advantage(pairs[:2], 2)[:, 0])}"
      f"   Dr.GRPO {row(group_advantage(pairs[:2], 2, scale_by_std=False)[:, 0])}")
print(f"    lost coin:  1 search wrong (-1.1) vs 2 searches (0.8) GRPO {row(group_advantage(pairs[2:], 2)[:, 0])}"
      f"   Dr.GRPO {row(group_advantage(pairs[2:], 2, scale_by_std=False)[:, 0])}")
print("""  With two attempts, (R - mean) / std is always +-0.707, however far apart they are. The
  coin lands either way half the time, so GRPO's pushes between one and two searches
  cancel -- though two searches are worth 0.9 more. Dr.GRPO keeps the gap's size.""")
nearly = torch.tensor([[0.0, 0.0, 1.0]] * 3 + [[-0.1, 0.0, 1.0]])
print(f"  a group that nearly agrees, totals {row(nearly.sum(1))}:")
print(f"    GRPO {row(group_advantage(nearly, 4)[:, 0])}   Dr.GRPO "
      f"{row(group_advantage(nearly, 4, scale_by_std=False)[:, 0])}")
print("  The divide blows a 0.1 difference up to full size, and keeps pushing a nearly settled")
print("  question as hard as an open one.")

# ---------------------------------------------------------------- step 5
banner("STEP 5  a dead group: every attempt scored the same")
solved = torch.tensor([[0.0, 0.0, 1.0]] * 4)
print(f"  totals {row(solved.sum(1))}: mean 1.0, std 0 -> A = 0 / (0 + eps) = {row(group_advantage(solved, 4)[:, 0])}")
print("""  No attempt was better than another, so the group teaches nothing: its whole gradient is
  zero. Once a question is solved every group of it is dead; DAPO's dynamic sampling
  throws such groups away and samples new questions. run_simple_grpo.py counts them.""")

# ---------------------------------------------------------------- step 6
banner("STEP 6  the loss: SimplePPO's clip, plus beta * KL(pi_theta || pi_ref)")
print("  k3, DeepSeekMath's eq. 4, on the action taken: x - log x - 1, with x = pi_ref / pi_theta")
probe = env.Policy()
with torch.no_grad():
    probe.logits.copy_(torch.tensor([[0.4, 0.6]] * 12).log())
ref_probs = torch.tensor([0.6, 0.4])                  # pi_ref: SKIP 0.6, SEARCH 0.4 (the starting policy)
k_search = kl_penalty(probe, torch.tensor([0]), torch.tensor([env.SEARCH]), ref_probs[env.SEARCH:].log())
k_skip = kl_penalty(probe, torch.tensor([0]), torch.tensor([env.SKIP]), ref_probs[:env.SEARCH].log())
print(f"  pi_theta(search) 0.6, pi_ref(search) 0.4:")
k_search, k_skip = float(k_search.detach()), float(k_skip.detach())
print(f"    took SEARCH: x = 0.4/0.6 -> {k_search:.4f}      took SKIP: x = 0.6/0.4 -> {k_skip:.4f}")
exact = 0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.6)
print(f"    averaged by pi_theta: 0.6 * {k_search:.4f} + 0.4 * {k_skip:.4f} = "
      f"{0.6 * k_search + 0.4 * k_skip:.4f};  exact KL = {exact:.4f}")
print("  One action per step is all an LLM can afford to look at (TRPO's sum is over ~150k tokens).")

torch.manual_seed(0)
policy, ref_policy = env.Policy(), env.Policy()
batch = env.rollout(policy, ref_policy, 2, 8)
batch["advantages"] = group_advantage(batch["rewards"], 8)
steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "ref_logp", "advantages")}
pg = policy_loss(policy, steps["states"], steps["actions"], steps["old_logp"], steps["advantages"])
kl = kl_penalty(policy, steps["states"], steps["actions"], steps["ref_logp"])
print(f"\n  a batch of 2 questions x 8 attempts = {steps['states'].numel()} steps, at theta_old = pi_ref:")
print(f"  pg = -L^CLIP = {f(pg):+.3f}   (ratio 1: -mean(A), and each group's A sums to 0)")
print(f"  kl           = {f(kl):.3f}    (the policy has not moved from pi_ref yet)")
optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
metrics = grpo_update(policy, optimizer, batch)
print(f"  grpo_update: 10 epochs x 3 minibatches of 16 = 30 steps, mean kl {metrics['kl']:.4f}. "
      f"J {env.true_reward(env.Policy()):.3f} -> {env.true_reward(policy):.3f}")
print("  No value loss, no entropy bonus, no critic to train: one network, one loss.")

# ---------------------------------------------------------------- step 7
banner("STEP 7  the whole loop: 60 iterations of 2 questions x 8 attempts (seed 0)")
import simple_grpo as impl  # noqa: E402  (whichever implementation was imported above)
curve, policy, dead = env.train(impl, seed=0)
print(f"  true J: {curve[0]:.3f} after 1, {curve[9]:.3f} after 10, {curve[29]:.3f} after 30, "
      f"{curve[-1]:.3f} after 60.   best possible {env.BEST_J}")
print("  learned p(search):")
for line in env.describe(policy):
    print("  " + line)
print(f"  dead groups: {sum(dead[:10]) / 10:.2f} of the first ten batches, {sum(dead[-10:]) / 10:.2f} of the last ten")
print("  ./scripts/run_simple_grpo.sh run compares GRPO with Dr.GRPO, smaller groups, no KL, and PPO.")
