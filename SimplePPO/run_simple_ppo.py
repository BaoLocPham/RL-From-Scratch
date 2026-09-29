"""PPO on the multi-step toy, and what each of its pieces buys: ``python SimplePPO/run_simple_ppo.py``.

Four runs of the same loop, 20 seeds each, 60 iterations of 16 episodes, SGD lr 0.3:

    PPO             critic + GAE, the clip, 10 epochs of minibatches per batch
    1 epoch         the same, but each batch used once (the toy track's VPG habit)
    no critic       advantage = the episode's total reward - the batch's mean (no V, no GAE)
    no clip         the ratio surrogate L^CPI in place of L^CLIP

Takes about a minute. ``RL_IMPL=scratch`` runs your from_scratch code.
"""

import os
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
import env  # noqa: E402
import simple_ppo as impl  # noqa: E402

SEEDS = 20
MARKS = (9, 29, 59)                                    # after 10, 30 and 60 iterations


def outcome_advantage(batch):
    """No critic: every step of an episode gets that episode's total reward, minus the batch's mean."""
    total = batch["rewards"].sum(1, keepdim=True).expand(-1, env.TURNS)
    return total - total.mean(), total


def cpi_loss(policy, states, actions, old_logp, advantages, eps=0.2):
    """-L^CPI: the ratio with no clip."""
    ratio = torch.exp(policy.dist(states).log_prob(actions) - old_logp)
    return -(ratio * advantages).mean()


def without_clip():
    """`impl` with policy_loss swapped for L^CPI, for the duration of one ppo_update."""
    def update(*args, **kwargs):
        original, impl.policy_loss = impl.policy_loss, cpi_loss
        try:
            return impl.ppo_update(*args, **kwargs)
        finally:
            impl.policy_loss = original
    return types.SimpleNamespace(compute_gae=impl.compute_gae, ppo_update=update)


RUNS = {
    "PPO": (impl, {}, "critic + GAE, clip, 10 epochs"),
    "1 epoch": (impl, {"epochs": 1}, "each batch used once"),
    "no critic": (impl, {"advantage": outcome_advantage}, "episode reward - batch mean"),
    "no clip": (without_clip(), {}, "L^CPI instead of L^CLIP"),
}

if __name__ == "__main__":
    torch.set_num_threads(1)                           # tiny model: one thread is fastest
    print(f"""The multi-step toy: 3 turns per question, SEARCH (-0.1 at once) or SKIP, then the answer.
HARD needs two searches, EASY none. Start J {env.true_reward(env.Policy()):.3f}, best possible {env.BEST_J}.
{SEEDS} seeds, 60 iterations of 16 episodes (48 steps), SGD lr 0.3, gamma 1, lam 0.8.
""")
    print(f"  {'run':<10} | {'what changes':<30} | {'J@10':>5} | {'J@30':>5} | {'J@60':>5}")
    results = {}
    for name, (module, kwargs, what) in RUNS.items():
        curves = [env.train(module, seed, **kwargs)[0] for seed in range(SEEDS)]
        mean = [sum(c[i] for c in curves) / SEEDS for i in MARKS]
        results[name] = mean
        print(f"  {name:<10} | {what:<30} | " + " | ".join(f"{j:.3f}" for j in mean), flush=True)

    print(f"""
Reading it:
  1 epoch     the biggest gap. Reusing each batch for 10 epochs is what makes PPO
              sample-efficient; the clip is what makes that reuse safe in general.
  no critic   a small gap here ({results['PPO'][-1] - results['no critic'][-1]:+.3f} at 60). With 3 turns and a visible question
              type, the batch mean is already a decent baseline. The critic's per-step
              credit matters more as episodes get longer -- and its small gain is why
              GRPO can drop the critic for LLMs, where each response gets one reward.
  no clip     no gain here ({results['PPO'][-1] - results['no clip'][-1]:+.3f} at 60): at 10 epochs and this learning
              rate, the unclipped ratio does not run far enough to hurt. Surrogates/
              shows what happens at 50 epochs: L^CPI gets 10 of 20 runs stuck.

So on this toy, most of PPO's gain over VPG comes from reusing each batch. The
clip and the critic are what keep that reuse working as the problem grows.""")

    curve, policy, critic = env.train(impl, seed=0)
    print(f"\nOne run (seed 0), learned p(search) -- HARD should search twice, EASY never:")
    for row in env.describe(policy):
        print(row)
    print("\nNext: PPO/ -- the same algorithm for LLMs, where each step is a token and responses need masks.")
