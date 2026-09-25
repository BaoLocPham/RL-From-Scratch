"""TRPO vs vanilla PG: ``python TRPO/run_trpo.py``.

Every run collects 16 new rollouts per iteration and uses lr 0.3. The difference
is how many epochs each takes over those rollouts:
  vanilla PG, 1 epoch     1 epoch per iteration, with L^PG                  (flaw 1: slow)
  vanilla PG, 50 epochs   50 epochs per iteration, with L^PG, no constraint (flaw 2: overshoot)
  TRPO                    up to 50 epochs per iteration, with eq. 3, stopped by the KL constraint
An epoch is one pass over the iteration's 16 rollouts; with no minibatches, 1 epoch = 1 update.
The algorithm itself lives in trpo.py.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "VPG"))
sys.path.insert(0, str(HERE))
from trpo import trpo_update  # noqa: E402
from vpg import (EASY, HARD, TOOL, Policy, collect_rollouts, fresh_batch_each_update,  # noqa: E402
                 reuse_one_batch, vpg_update)

MAX_KL = 0.01           # delta: the most one iteration may change the policy's behaviour
LR = 0.3                # the same step size for every run
EPOCHS = 50             # the most epochs any run may take per iteration
SEEDS = 20


def trpo_fresh_batch_each_update(seed, iterations, n=16):
    """TRPO as intended: collect 16 NEW rollouts, run epochs on them until the KL constraint stops it, repeat.

    Returns (iteration, probabilities, true J) after every iteration, starting at
    0, and how many epochs each iteration got.
    """
    torch.manual_seed(seed)                                     # same seed -> same first batch as VPG
    policy = Policy()
    rows = [(0, policy.probs(), policy.true_reward())]
    epochs_per_batch = []
    for iteration in range(1, iterations + 1):
        batch = collect_rollouts(policy, n)                     # fresh rollouts, from the current model
        epochs_per_batch.append(trpo_update(policy, batch, max_kl=MAX_KL, lr=LR, epochs=EPOCHS))
        rows.append((iteration, policy.probs(), policy.true_reward()))
    return rows, epochs_per_batch


def vpg_many_epochs_per_iteration(seed, iterations, n=16):
    """VPG's shortcut, every iteration: 50 epochs of L^PG over its rollouts, no KL constraint.

    Returns (iteration, probabilities, true J) after every iteration, starting at 0.
    """
    torch.manual_seed(seed)
    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=LR)
    rows = [(0, policy.probs(), policy.true_reward())]
    for iteration in range(1, iterations + 1):
        batch = collect_rollouts(policy, n)
        vpg_update(policy, optimizer, batch, updates=EPOCHS)    # 50 epochs over the same 16 rollouts
        rows.append((iteration, policy.probs(), policy.true_reward()))
    return rows


def is_stuck(p):
    """Locked onto a wrong action: p(tool|HARD) near 0 or p(tool|EASY) near 1."""
    return bool(p[HARD, TOOL] < 0.05 or p[EASY, TOOL] > 0.95)


def print_terms():
    print("""Terms  (p(tool|HARD), p(tool|EASY), true J, rollout, batch, update, epoch, iteration:
        as in ./scripts/run_vpg.sh run)
  ratio         pi_theta / pi_theta_old: how much more (or less) likely an action is now than
                when the batch was collected. 1 = unchanged.
  surrogate     eq. 3, mean(ratio * A): like L^PG, but it knows which model collected the batch.
  KL            how much the policy's BEHAVIOUR has changed since theta_old. 0 = unchanged.
  KL constraint eq. 4, KL <= delta: the limit on how far one iteration may move the policy.
  delta         the most KL one iteration may use. Here 0.01.
  trust region  every policy within KL <= delta of theta_old: where the batch is still trusted.
""")


if __name__ == "__main__":
    torch.set_num_threads(1)                                    # tiny model: one thread is fastest
    print_terms()

    vpg = fresh_batch_each_update(seed=17, updates=400)
    trpo, epochs = trpo_fresh_batch_each_update(seed=17, iterations=400)

    print("=" * 86)
    print("Vanilla PG (A) vs TRPO (B): both collect 16 new rollouts per iteration, lr 0.3")
    print("=" * 86)
    print("Both start from the same unlucky batch as the VPG demo (seed 17).")
    print("  A = vanilla PG: 1 epoch per iteration, with L^PG")
    print(f"  B = TRPO:       up to {EPOCHS} epochs per iteration on eq. 3, stopped when the KL from theta_old would pass {MAX_KL}")
    print("  HARD / EASY = p(tool|HARD) / p(tool|EASY)\n")
    print(f"{'':>9} | {'':>8} | {'A: vanilla PG':^24} | {'B: TRPO':^24}")
    print(f"{'iteration':>9} | {'rollouts':>8} | {'HARD':>6} {'EASY':>6} {'true J':>9}  | "
          f"{'HARD':>6} {'EASY':>6} {'true J':>9}")
    for it in (0, 1, 5, 20, 50, 100, 400):
        _, pa, ja = vpg[it]
        _, pb, jb = trpo[it]
        print(f"{it:>9} | {16 * it:>8} | {pa[HARD, TOOL]:>6.2f} {pa[EASY, TOOL]:>6.2f} {ja:>9.3f}  | "
              f"{pb[HARD, TOOL]:>6.2f} {pb[EASY, TOOL]:>6.2f} {jb:>9.3f}")

    shortcut_j = reuse_one_batch(seed=17, updates=100)[-1][2]   # the VPG demo's 100 epochs, no constraint
    print(f"""
Reading it:
  iteration 1   both read the same unlucky batch (it says search hurts HARD questions).
                TRPO takes {epochs[0]} epochs on it before the KL constraint stops it, so it steps
                further the wrong way: J {vpg[1][2]:.3f} (A) vs {trpo[1][2]:.3f} (B). The constraint caps that
                damage -- VPG's shortcut of 100 epochs with no constraint fell to {shortcut_j:.3f}.
  5 - 20        the next batches are fresh and mostly honest. TRPO gets many safe epochs
                out of each one, so it recovers first and pulls ahead: J {trpo[20][2]:.3f} vs {vpg[20][2]:.3f}.
  400           both end near the best J 0.90: {vpg[400][2]:.3f} (A) vs {trpo[400][2]:.3f} (B).

  Epochs per iteration over the first 800 rollouts:  vanilla PG 1,  TRPO {sum(epochs[:50]) / 50:.1f} on average.
                That is the gain: more learning from each expensive batch, without
                leaving the trust region.""")

    # Not one lucky seed: 20 runs of each, with the same rollouts, lr and epoch limit.
    marks = (10, 25, 50)                                        # iterations: 160, 400, 800 rollouts
    runs = {
        "VPG, 1 epoch per iteration": ([fresh_batch_each_update(seed, marks[-1]) for seed in range(SEEDS)],
                                       "slow"),
        f"VPG, {EPOCHS} epochs per iteration": ([vpg_many_epochs_per_iteration(seed, marks[-1])
                                              for seed in range(SEEDS)], "overshoots (no KL constraint)"),
        f"TRPO, <={EPOCHS} epochs per iteration": ([trpo_fresh_batch_each_update(seed, marks[-1])[0]
                                                 for seed in range(SEEDS)], "fast and safe (KL constraint)"),
    }
    print(f"""
{"=" * 86}
Is TRPO better than VPG?  {SEEDS} seeds each, lr {LR}, the same rollouts for every run
{"=" * 86}""")
    print(f"  {'':<32} {'J@160':>6} {'J@400':>6} {'J@800':>6} {'stuck':>7}")
    for name, (rs, verdict) in runs.items():
        mean_j = [sum(r[m][2] for r in rs) / SEEDS for m in marks]
        stuck = sum(is_stuck(r[marks[-1]][1]) for r in rs)
        print(f"  {name:<32} " + " ".join(f"{j:>6.3f}" for j in mean_j) + f" {stuck:>4}/{SEEDS}   {verdict}")
    print("""
  vs VPG, 1 epoch      TRPO learns more from every iteration's rollouts (many safe
                       epochs instead of one), so it is ahead at every budget: it
                       fixes flaw 1, slowness.
  vs VPG, 50 epochs    50 epochs with no KL constraint is quick at first, then locks
                       misleading batches in and gets stuck. TRPO allows the same 50
                       epochs, but the KL constraint stops each iteration early: it
                       fixes flaw 2, overshoot, and ends highest with no run stuck.
  stuck  = ended with p(tool|HARD) < 0.05 or p(tool|EASY) > 0.95: locked onto a wrong action.""")

    print("""
TRPO's flaw: the KL constraint is not part of the loss. After every epoch, something outside
the loss has to measure the KL and undo the step if it went too far -- "subject to"
is not something you can call .backward() on. PPO builds the limit INTO the loss by
clipping the ratio, so plain SGD is enough.""")
