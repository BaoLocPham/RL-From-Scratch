"""Vanilla policy gradient's flaw, side by side: ``python VPG/run_vpg.py``.

Both runs start from the same unlucky batch and take the same updates. The only
difference: new rollouts for every update (vanilla PG as intended: correct, but
slow), or updating again and again on the first 16 rollouts (cheap, but it
overshoots). The algorithm itself lives in vpg.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpg import EASY, HARD, TOOL, fresh_batch_each_update, print_terms, reuse_one_batch  # noqa: E402

torch.set_num_threads(1)                                    # tiny model: one thread is fastest
print_terms()

# Both runs start from the SAME first batch (seed 17, an unlucky one) and take the
# SAME number of updates with the same lr. The only difference: does each update
# get new rollouts (A), or keep using the first 16 (B)?
fresh = fresh_batch_each_update(seed=17, updates=400)
same = reuse_one_batch(seed=17, updates=100)

print("=" * 86)
print("Vanilla PG: new rollouts for every update (A) vs. reusing the first 16 rollouts (B)")
print("=" * 86)
print("Both start from the same unlucky batch (seed 17) and take the same updates, lr 0.3.")
print("  A = vanilla PG as intended: collect 16 NEW rollouts before every update")
print("  B = the shortcut: collect 16 rollouts once, keep updating on them")
print("  HARD / EASY = p(tool|HARD) / p(tool|EASY)\n")
print(f"{'':>6} | {'A: new rollouts every update':^34} | {'B: the same 16 rollouts':^34}")
print(f"{'update':>6} | {'rollouts':>8} {'HARD':>6} {'EASY':>6} {'true J':>8}   | "
      f"{'rollouts':>8} {'HARD':>6} {'EASY':>6} {'true J':>8}")
for update in (0, 1, 5, 20, 50, 100):
    _, pa, ja = fresh[update]
    _, pb, jb = same[update]
    print(f"{update:>6} | {16 * update:>8} {pa[HARD, TOOL]:>6.2f} {pa[EASY, TOOL]:>6.2f} {ja:>8.3f}   | "
          f"{16 if update else 0:>8} {pb[HARD, TOOL]:>6.2f} {pb[EASY, TOOL]:>6.2f} {jb:>8.3f}")
_, pa, ja = fresh[400]
print(f"{400:>6} | {16 * 400:>8} {pa[HARD, TOOL]:>6.2f} {pa[EASY, TOOL]:>6.2f} {ja:>8.3f}   | {'':>8}")

print("""
Reading it:
  update 1   identical. Same batch, same model: both are one valid vanilla PG update.
         The batch was unlucky -- by chance it said search hurts HARD questions --
         so both step slightly the wrong way (J 0.600 -> 0.592).
  A          the next batches are new and mostly honest, so they correct that step.
         J climbs to 0.87 after 100 updates and 0.89 after 400. Correct, but SLOW:
         every update costs 16 new, expensive rollouts (6,400 in total).
  B          only 16 rollouts in total -- cheap -- but updates 2-100 reread rollouts
         from a model that no longer exists. The unlucky batch is never corrected,
         only repeated: both types end on the wrong action and J falls to 0.37.""")
# Not a cherry-picked seed: count how often B on a random first batch ends below the start.
worse = sum(reuse_one_batch(seed, updates=100)[-1][2] < 0.6 for seed in range(30))
print(f"             Over 30 random batches, B ended below the starting J 0.60 in {worse}/30.\n")
print("The flaw: vanilla PG is either correct-but-slow (A) or cheap-but-wrong (B).")
print("L^PG has no idea how far is too far from theta_old, so it cannot be both.")
print("TRPO and PPO fix exactly that.")
