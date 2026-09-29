"""GRPO on the multi-step toy, and what each of its choices does: ``python SimpleGRPO/run_simple_grpo.py``.

Eight runs of the same budget, 20 seeds each, 60 iterations of 16 episodes, SGD lr 0.3:

    GRPO            2 questions x 8 attempts, (R - group mean) / group std, beta 0.04, 10 epochs
    1 epoch         the same, but each batch used once
    no KL           beta 0: nothing holds the policy near pi_ref (DAPO drops it too)
    Dr.GRPO         (R - group mean), no divide by the std
    G 2             8 questions x 2 attempts, GRPO's std divide
    G 2, Dr.GRPO    8 questions x 2 attempts, no divide
    batch mean      no groups: R - the whole batch's mean (SimplePPO's "no critic")
    PPO             SimplePPO's critic + GAE, on the same right-or-wrong rewards

Takes about three minutes. ``RL_IMPL=scratch`` runs your from_scratch code.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
import group_env as env  # noqa: E402
import simple_grpo as impl  # noqa: E402

# PPO's row always runs the reference SimplePPO, loaded under its own name.
_spec = importlib.util.spec_from_file_location("simple_ppo_reference", HERE.parent / "SimplePPO" / "simple_ppo.py")
ppo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppo)

SEEDS = 20
MARKS = (9, 29, 59)                                    # after 10, 30 and 60 iterations


def dr_grpo(group_size):
    return lambda batch: impl.group_advantage(batch["rewards"], group_size, scale_by_std=False)


def batch_mean(batch):
    """No groups: every step of an episode gets its total reward, minus the whole batch's mean."""
    total = batch["rewards"].sum(1, keepdim=True)
    return (total - total.mean()).expand_as(batch["rewards"])


def train_ppo(seed, iterations=60, questions=2, group_size=8, lr=0.3):
    """SimplePPO's loop on this toy's rewards: a critic, GAE, and eq. 9. Its groups are just 16 episodes."""
    torch.manual_seed(seed)
    policy, ref_policy, critic = env.Policy(), env.Policy(), env.Critic()
    optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=lr)
    curve = []
    for _ in range(iterations):
        batch = env.rollout(policy, ref_policy, questions, group_size, critic=critic)
        batch["advantages"], batch["returns"] = ppo.compute_gae(batch["rewards"], batch["values"], 1.0, 0.8)
        ppo.ppo_update(policy, critic, optimizer, batch)
        curve.append(env.true_reward(policy))
    return curve, policy, None


RUNS = {
    "GRPO": (lambda seed: env.train(impl, seed), "G 8, std, beta 0.04"),
    "1 epoch": (lambda seed: env.train(impl, seed, epochs=1), "each batch used once"),
    "no KL": (lambda seed: env.train(impl, seed, beta=0.0), "beta 0"),
    "Dr.GRPO": (lambda seed: env.train(impl, seed, advantage=dr_grpo(8)), "no divide by the std"),
    "G 2": (lambda seed: env.train(impl, seed, questions=8, group_size=2), "8 questions x 2"),
    "G 2, Dr.GRPO": (lambda seed: env.train(impl, seed, questions=8, group_size=2, advantage=dr_grpo(2)),
                     "8 x 2, no divide"),
    "batch mean": (lambda seed: env.train(impl, seed, advantage=batch_mean), "R - batch mean, no groups"),
    "PPO": (train_ppo, "critic + GAE (SimplePPO)"),
}

if __name__ == "__main__":
    torch.set_num_threads(1)                           # tiny model: one thread is fastest
    print(f"""The multi-step toy in groups: each question answered 8 times, graded right (+1) or wrong (-1).
HARD needs two searches, EASY none. Start J {env.true_reward(env.Policy()):.3f}, best possible {env.BEST_J}.
{SEEDS} seeds, 60 iterations of 16 episodes, SGD lr 0.3, 10 epochs of minibatches of 16 steps.
"stuck": J below 0.7 after 60 iterations. "dead": groups whose attempts all scored the same.
""")
    print(f"  {'run':<12} | {'what changes':<26} | {'J@10':>5} | {'J@30':>5} | {'J@60':>5} | stuck | dead@60")
    results = {}
    for name, (run, what) in RUNS.items():
        runs = [run(seed) for seed in range(SEEDS)]
        mean = [sum(r[0][i] for r in runs) / SEEDS for i in MARKS]
        stuck = sum(r[0][-1] < 0.7 for r in runs)
        dead = f"{sum(r[2][-1] for r in runs) / SEEDS:.2f}" if runs[0][2] is not None else "  -"
        results[name] = mean
        print(f"  {name:<12} | {what:<26} | " + " | ".join(f"{j:.3f}" for j in mean) +
              f" | {stuck:>2}/{SEEDS} | {dead:>7}", flush=True)

    print(f"""
Reading it:
  1 epoch      a large gap ({results['1 epoch'][-1] - results['GRPO'][-1]:+.3f}), as in SimplePPO: reusing each batch is what
               the clip makes safe, and GRPO keeps PPO's clip.
  G 2          the std divide's worst case. With two attempts, (R - mean) / std is
               always +0.71 and -0.71, however far apart the two rewards are: a
               lucky one-search answer beating a two-search one by 0.1 counts as
               much as an unlucky one losing by 1.9. On HARD questions the two
               cancel, and the stuck runs sit between one and two searches.
               Dr.GRPO keeps the size of the gap and none get stuck ({results['G 2, Dr.GRPO'][-1]:.3f} vs {results['G 2'][-1]:.3f}).
  Dr.GRPO      at G 8 the divide does no harm ({results['Dr.GRPO'][-1] - results['GRPO'][-1]:+.3f}); it bites on small groups.
               It also explains the dead column: as a group nearly agrees, its std
               shrinks and GRPO blows the last small differences up to full size,
               so its policy sharpens until whole groups agree. Dr.GRPO's
               advantages shrink with the differences, and its groups stay alive.
  no KL        a little higher ({results['no KL'][-1] - results['GRPO'][-1]:+.3f}): the KL pulls toward the starting
               policy, and on this toy the starting policy is bad. On an LLM, pi_ref
               is the pretrained model, and staying near it is the point.
  batch mean   as good as groups here ({results['batch mean'][-1] - results['GRPO'][-1]:+.3f}): with only two question types, which
               the policy sees, one batch-wide baseline is already fair. Groups pay
               when every prompt has its own difficulty, as on an LLM's dataset.
  PPO          the critic wins on this toy ({results['PPO'][-1] - results['GRPO'][-1]:+.3f}). GRPO's case is not accuracy
               here: it is the critic it does not need -- a second LLM, its memory,
               its value loss -- when one reward arrives per response.""")

    curve, policy, dead = env.train(impl, seed=0)
    print(f"\nOne run (seed 0), learned p(search) -- HARD should search twice, EASY never:")
    for row in env.describe(policy):
        print(row)
    print(f"dead groups: {dead[0]:.2f} of the first batch, {sum(dead[-10:]) / 10:.2f} of the last ten: "
          "once a question is solved, its group agrees and goes quiet.")
    print("\nNext: GRPO/ -- the same advantage for LLMs, at verl's (batch, response_length) shapes.")
