"""Simple PPO, one step at a time: ``python SimplePPO/steps_simple_ppo.py``.

Follows one episode of the multi-step toy through every piece of PPO, printing
each equation with the numbers substituted into it. Set ``RL_IMPL=scratch`` to
run the same walkthrough on your SimplePPO/from_scratch/simple_ppo.py, and
``./scripts/run_simple_ppo.sh diff`` to compare the two outputs line by line.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
from simple_ppo import compute_gae, entropy_bonus, policy_loss, ppo_update, value_loss  # noqa: E402
import env  # noqa: E402


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 4) + 0.0                    # + 0.0: no -0.0


# ---------------------------------------------------------------- step 1
banner("STEP 1  one episode: three decisions, the reward mostly at the end")
print("""  A HARD question. The agent searches, searches again, then skips; with two searches
  its answer is good (quality 1.0), and each search cost 0.1 when it happened.

  turn | state (HARD, turn, searches so far) | action | reward | critic V(s_t)
     0 | HARD, 0, 0                         | SEARCH |  -0.1  |  0.5
     1 | HARD, 1, 1                         | SEARCH |  -0.1  |  0.6
     2 | HARD, 2, 2                         | SKIP   |   1.0  |  0.8

  On the toy track one decision got one advantage. Here three decisions share one
  outcome: which of them deserves the credit? The critic's predictions decide.""")

# ---------------------------------------------------------------- step 2
banner("STEP 2  GAE: delta_t = r_t + gamma V(s_t+1) - V(s_t);  A_t = delta_t + gamma lam A_t+1")
rewards, values = torch.tensor([[-0.1, -0.1, 1.0]]), torch.tensor([[0.5, 0.6, 0.8]])
gamma, lam = 1.0, 0.8
print(f"  gamma {gamma}, lam {lam}. Walked backwards, because A_t needs A_t+1:")
next_v, carry = 0.0, 0.0
for t in (2, 1, 0):
    delta = float(rewards[0, t]) + gamma * next_v - float(values[0, t])
    carry = delta + gamma * lam * carry
    print(f"  t={t}: delta = {float(rewards[0, t]):+.1f} + {gamma:.0f} * {next_v:.1f} - {float(values[0, t]):.1f} = "
          f"{delta:+.2f}   A_{t} = {carry:+.3f}")
    next_v = float(values[0, t])
advantages, returns = compute_gae(rewards, values, gamma, lam)
print(f"  compute_gae -> advantages {[f(v) for v in advantages[0]]}, returns = A + V {[f(v) for v in returns[0]]}")
print("  Each surprise (delta) is how much better turn t went than the critic expected. The first")
print("  search came out even (delta 0.0: it cost 0.1, and the critic's guess rose by exactly 0.1);")
print("  the later turns were pleasant surprises, and their credit flows back to turn 0 through A.")
for lam_ in (0.0, 1.0):
    a, _ = compute_gae(rewards, values, gamma, lam_)
    what = "one-step surprise only: trusts the critic" if lam_ == 0 else "rewards from t on minus V(s_t): the toy's reward - baseline"
    print(f"  lam {lam_:.0f}: advantages {[f(v) for v in a[0]]}   ({what})")

# ---------------------------------------------------------------- step 3
banner("STEP 3  after GAE, time no longer matters: flatten to one sample per step")
torch.manual_seed(0)
policy, critic = env.Policy(), env.Critic()
batch = env.rollout(policy, critic, 16)
batch["advantages"], batch["returns"] = compute_gae(batch["rewards"], batch["values"], 1.0, 0.8)
print(f"  a batch of 16 episodes: states {tuple(batch['states'].shape)}, advantages {tuple(batch['advantages'].shape)}"
      "  (episodes, turns)")
print(f"  flattened: {batch['states'].numel()} samples, each (state, action, old log-prob, advantage, return)")
print("  From here on it is Surrogates' loop: one sample is one decision, as on the toy track.")

# ---------------------------------------------------------------- step 4
banner("STEP 4  L^CLIP per step: -min(r A, clip(r, 0.8, 1.2) A) -- your Surrogates clip")
print(f"  one step, A = +2, old p(search) = 0.40:")
one = (torch.tensor([0]), torch.tensor([env.SEARCH]), torch.tensor([0.4]).log(), torch.tensor([2.0]))
for p in (0.40, 0.44, 0.60, 0.28):
    probe = env.Policy()
    with torch.no_grad():
        probe.logits.copy_(torch.tensor([[1 - p, p]] * 12).log())
    r = p / 0.4
    note = {1.0: "at theta_old", 1.1: "inside the range: still rewarded", 1.5: "past 1.2: flat, stops",
            0.7: "moved the wrong way: pays in full"}[round(r, 1)]
    print(f"  p {p:.2f} -> ratio {r:.1f}: -min({r * 2:.1f}, {min(max(r, 0.8), 1.2) * 2:.1f}) = "
          f"{f(policy_loss(probe, *one)):+.2f}   {note}")

# ---------------------------------------------------------------- step 5
banner("STEP 5  eq. 9 on the batch: loss = pg + 0.5 * vf - 0.01 * S")
steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "advantages", "returns")}
pg = policy_loss(policy, steps["states"], steps["actions"], steps["old_logp"], steps["advantages"])
vf = value_loss(critic, steps["states"], steps["returns"])
ent = entropy_bonus(policy, steps["states"])
print(f"  pg  = -L^CLIP              = {f(pg):+.4f}   (ratio 1 at theta_old: just -mean(A))")
print(f"  vf  = 0.5 * mean((V - R)^2) = {f(vf):.4f}   (the critic starts at 0 everywhere)")
print(f"  S   = mean entropy          = {f(ent):.4f}   (p(search) 0.4 everywhere: 0.673)")
print(f"  loss = {f(pg):+.4f} + 0.5 * {f(vf):.4f} - 0.01 * {f(ent):.4f} = {f(pg + 0.5 * vf - 0.01 * ent):+.4f}")
optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=0.3)
metrics = ppo_update(policy, critic, optimizer, batch)
print(f"  ppo_update: 10 epochs x 3 minibatches of 16 = 30 steps. J {env.true_reward(env.Policy()):.3f} -> "
      f"{env.true_reward(policy):.3f}")

# ---------------------------------------------------------------- step 6
banner("STEP 6  the whole loop: 60 iterations of 16 episodes (seed 0)")
import simple_ppo as impl  # noqa: E402  (whichever implementation was imported above)
curve, policy, critic = env.train(impl, seed=0)
print(f"  true J: {curve[0]:.3f} after 1, {curve[9]:.3f} after 10, {curve[29]:.3f} after 30, "
      f"{curve[-1]:.3f} after 60.   best possible {env.BEST_J}")
print("  learned p(search):")
for row in env.describe(policy):
    print("  " + row)
v = critic.v.detach()
print(f"  critic V at the start of an episode: HARD {f(v[env.state_id(0, 0, 0)]):.2f}, "
      f"EASY {f(v[env.state_id(1, 0, 0)]):.2f}")
print("  (best play is worth 0.8 and 1.0; the critic's values are noisy estimates of the CURRENT policy,")
print("  learned from rewards with noise std 1.0)")
print("  ./scripts/run_simple_ppo.sh run compares this with PPO minus each of its pieces.")
