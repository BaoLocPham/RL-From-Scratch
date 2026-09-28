"""The toy's Table 1: every candidate for the slot, same loop, same budget. ``python Surrogates/run_surrogates.py``.

The PPO paper, §6.1, runs one training loop with each candidate objective in
its slot and compares them. This does the same on the VPG toy:

    1. The main table: 20 seeds, 800 rollouts each, SGD at lr 0.3 (as in VPG/ and
       TRPO/), 50 epochs per iteration for every slot that reuses the batch.
    2. Units: why beta is the odd one out. With Adam (the paper's optimizer), a
       run's steps do not depend on the size of the advantages, so multiplying
       every reward by 10 changes nothing -- unless the slot mixes reward with KL.

Takes about three minutes. Also saves the learning curves to surrogates.png.

Columns (VPG/README.md defines true J and stuck):
  J@160     true J after 160 rollouts (likewise J@400, J@800), mean of the seeds
  stuck     runs that ended locked onto a wrong action (p(tool|HARD) < 0.05 or
            p(tool|EASY) > 0.95); they cannot recover
  KL/iter   how far one iteration moved the policy, in nats: the trust region actually used
  hooks     what the slot needed besides itself: AFTER_STEP (surgery on the loop)
            or AFTER_ITERATION (between updates)
"""

import sys
from functools import partial
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loop import BATCH_SIZE, EPOCHS, LR, ROOT, is_stuck, kl_rollback, pg_slot, train  # noqa: E402
from surrogates import AdaptiveKL, clip_loss, cpi_loss, kl_penalty_loss  # noqa: E402

TOTAL_ROLLOUTS = 800
ITERATIONS = TOTAL_ROLLOUTS // BATCH_SIZE
SEEDS = 20
BEST_REWARD = 0.9


def fixed(slot, epochs=EPOCHS, after_step=None):
    """A candidate with no state: the same slot and hook every run."""
    return lambda: dict(slot=slot, epochs=epochs, after_step=after_step)


def adaptive(d_targ=0.01):
    """A fresh AdaptiveKL per run: its beta is state."""
    def make():
        a = AdaptiveKL(d_targ=d_targ)
        return dict(slot=a.slot, epochs=EPOCHS, after_iteration=a.after_iteration)
    return make


# name -> (make one run's settings, objective, hooks it needs)
CANDIDATES = {
    "L^PG, 1 epoch":         (fixed(pg_slot, epochs=1), "eq. 2  (VPG)", "none"),
    "L^PG, 50 epochs":       (fixed(pg_slot), "eq. 2, batch reused", "none"),
    "L^CPI":                 (fixed(cpi_loss), "eq. 6, no limit", "none"),
    "TRPO, delta 0.01":      (fixed(cpi_loss, after_step=kl_rollback(0.01)), "eq. 6 + rollback", "AFTER_STEP"),
    "KL penalty, beta 0.3":  (fixed(partial(kl_penalty_loss, beta=0.3)), "eq. 5", "none"),
    "KL penalty, beta 3":    (fixed(partial(kl_penalty_loss, beta=3)), "eq. 5", "none"),
    "KL penalty, beta 10":   (fixed(partial(kl_penalty_loss, beta=10)), "eq. 5", "none"),
    "KL adaptive, d 0.01":   (adaptive(), "eq. 8", "AFTER_ITERATION"),
    "L^CLIP, eps 0.2":       (fixed(clip_loss), "eq. 7", "none"),
}


def evaluate(make, seeds=SEEDS, **kwargs):
    """Mean J curve, stuck count and mean KL/iter over `seeds` runs."""
    curves, stuck, kls = [], 0, []
    for seed in range(seeds):
        rows, probs = train(seed=seed, iterations=ITERATIONS, **make(), **kwargs)
        curves.append([r[0] for r in rows])
        kls.append(sum(r[1] for r in rows) / len(rows))
        stuck += is_stuck(probs)
    curve = [sum(c[i] for c in curves) / seeds for i in range(ITERATIONS)]
    return curve, stuck, sum(kls) / seeds


def main_table():
    marks = [ITERATIONS // 5 - 1, ITERATIONS // 2 - 1, ITERATIONS - 1]    # 160, 400, 800 rollouts
    print(f"1. The toy's Table 1: {TOTAL_ROLLOUTS} rollouts per run, batch {BATCH_SIZE}, SGD lr {LR}, "
          f"{SEEDS} seeds.\n   Start J 0.60, best {BEST_REWARD}.\n")
    print(f"  {'slot':<21} | {'objective':<20} | {'J@160':>5} | {'J@400':>5} | {'J@800':>5} | "
          f"{'stuck':>5} | {'KL/iter':>7} | hooks")
    curves = {}
    for name, (make, objective, hooks) in CANDIDATES.items():
        curve, stuck, kl = evaluate(make)
        curves[name] = curve
        print(f"  {name:<21} | {objective:<20} | " + " | ".join(f"{curve[i]:>5.3f}" for i in marks)
              + f" | {stuck:>2}/{SEEDS} | {kl:>7.4f} | {hooks}", flush=True)
    print("""
  L^PG, 1 epoch    correct but slow: every expensive batch buys one small update.
  L^PG, 50 epochs  overshoots: L^PG has no memory of theta_old, so a misleading batch
  and L^CPI        gets locked in. L^CPI remembers theta_old but has no brake, and does
                   worse still. (The paper's L^CPI row scores -0.39: below a random policy.)
  TRPO             the brake as a hook: safe, but the loop itself had to change.
  KL penalty       the brake as a fee, inside the loss. beta 0.3 barely brakes, beta 10
                   brakes so hard it learns slowly; which beta is right depends on the
                   size of the rewards (table 2).
  KL adaptive      re-prices beta between iterations. It cannot stop the update in
                   progress, and in quiet stretches it halves beta toward 0, so the next
                   misleading batch meets almost no brake: the stuck runs.
  L^CLIP           the brake inside the loss, in behaviour units: needs nothing but the
                   slot, and ends highest with no run stuck. The paper agrees (0.82 vs
                   0.74 for adaptive KL) and uses it for everything after §6.1.""")
    return curves


def units_table(seeds=10, lr=0.01):
    print(f"""
{"=" * 90}
2. Units: multiply every reward by 10.  Adam lr {lr}, {seeds} seeds, 50 epochs per iteration.
{"=" * 90}
  With Adam a step's size does not depend on the size of the gradient, so scaling all
  rewards should change nothing. It does not -- except where the loss mixes reward with KL.
""")
    rows = {"TRPO, delta 0.01": fixed(cpi_loss, after_step=kl_rollback(0.01)),
            "L^CLIP, eps 0.2": fixed(clip_loss),
            "KL penalty, beta 30": fixed(partial(kl_penalty_loss, beta=30)),
            "KL adaptive, d 0.01": adaptive()}
    print(f"  {'slot':<21} | {'rewards x1: J@800':>17} {'KL/iter':>8} | {'rewards x10: J@800':>18} {'KL/iter':>8}")
    for name, make in rows.items():
        out = [evaluate(make, seeds=seeds, lr=lr, optim=torch.optim.Adam, advantage_scale=scale)
               for scale in (1.0, 10.0)]
        print(f"  {name:<21} | {out[0][0][-1]:>17.3f} {out[0][2]:>8.4f} | {out[1][0][-1]:>18.3f} {out[1][2]:>8.4f}",
              flush=True)
    print("""
  TRPO, L^CLIP     identical: delta and eps are DISTANCES (KL, ratio), which do not care
                   what units the reward is in.
  beta 30          a PRICE, reward per nat: x10 rewards make every nat 10x cheaper, so
                   the same beta goes from barely moving to moving freely -- it behaves
                   exactly like beta 3 on the original rewards.
  adaptive         re-prices beta, so KL/iter stays within about 2x of d_targ at either
                   scale where fixed beta moved 50x. That is the fix the paper's §4 proposes.""")


def save_figure(curves):
    """True J vs rollouts used, one line per slot. Skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")                                 # write a file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return
    rollouts = [BATCH_SIZE * (i + 1) for i in range(ITERATIONS)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in curves.items():
        style = "-" if name.startswith(("L^CLIP", "TRPO", "L^PG, 1")) else "--"
        ax.plot(rollouts, curve, style, label=name, linewidth=2 if name.startswith("L^CLIP") else 1.2)
    ax.axhline(BEST_REWARD, color="gray", linestyle=":", linewidth=1, label="best possible")
    ax.set_xlabel("rollouts used")
    ax.set_ylabel(f"true reward J (mean of {SEEDS} seeds)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = HERE / "surrogates.png"
    fig.savefig(path, dpi=120)
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    torch.set_num_threads(1)                                  # tiny model: one thread is fastest
    curves = main_table()
    units_table()
    save_figure(curves)
    print("""
So PPO is this loop with L^CLIP in the slot -- plus, in the paper's §5, a value-function loss
to train the critic behind A, an entropy bonus, and GAE advantages (eq. 9-12). PPO/ has
all of it the way verl writes it, for (batch, response_length) tensors.""")
