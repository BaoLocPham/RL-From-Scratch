"""VPG vs TRPO vs PPO, same rollout budget: ``python Comparison/run_compare.py``.

The story of the PPO paper, Sections 2-3, on one toy (VPG/vpg.py):

    VPG, each batch once   correct, but slow: one small update per iteration's rollouts
    VPG, batch reused      faster at first, then overshoots: acts on a stale reading
    TRPO                   reuses the batch safely, but its KL constraint sits outside the loss
    PPO                    TRPO's safety with VPG's simplicity: a clipped ratio

Every method gets the same number of rollouts (the expensive part), so the
comparison is fair. The true reward J is exact -- the environment is known --
and averaged over 20 seeds. Also saves the learning curves to vpg_trpo_ppo.png.

Columns (the rest are as in VPG/README.md, "Terms in the VPG logs"):
  J@160     true J after 160 rollouts (likewise J@400, J@800)
  stuck     runs that ended with p(tool|HARD) < 0.05 or p(tool|EASY) > 0.95, locked
            onto a wrong action. They cannot recover: an action with probability
            near 0 is almost never tried again, so no new evidence arrives.
  KL/iter     how much one iteration changed the policy's behaviour (0 = unchanged);
              TRPO allows at most 0.01
  step limit  what stops a method moving too far from theta_old in one iteration:
              none for VPG reused, the KL constraint (checked after every update)
              for TRPO, and PPO's clip, inside the loss
  An epoch is one pass over an iteration's 16 rollouts; with no minibatches,
  1 epoch = 1 update.
"""

import sys
from pathlib import Path

import torch

# All three methods share one toy, from VPG/; TRPO's update lives in TRPO/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VPG"))
sys.path.insert(0, str(ROOT / "TRPO"))
from vpg import BEST_REWARD, EASY, HARD, TOOL, Policy, collect_rollouts, vpg_update  # noqa: E402
from trpo import trpo_update  # noqa: E402

TOTAL_ROLLOUTS = 800    # rollout budget per run -- the same for every method
BATCH_SIZE = 16         # rollouts per iteration, so 50 iterations per run
LR = 0.3                # SGD learning rate, the same for all three methods
REUSE = 50              # epochs per iteration for "VPG reused", TRPO (at most) and PPO
MAX_KL = 0.01           # TRPO's delta: how far one iteration may move the policy
SEEDS = 20              # every number is averaged over this many independent runs


# ---------------------------------------------------------------- PPO (paper Section 3)
def ppo_loss(policy, qtype, action, advantage, old_logp, eps=0.2):
    """-L^CLIP: TRPO's eq. 3 ratio, but no extra credit once it leaves [1 - eps, 1 + eps].

    ratio = pi_theta / pi_theta_old, how far this action's probability has moved
    since the batch was collected. For A > 0, min(...) caps the reward for
    raising the probability at 1 + eps; for A < 0, it caps lowering it at 1 - eps.
    Past the cap the gradient is exactly zero, so the model stops moving on this
    batch -- the trust region, drawn with a clamp instead of a KL constraint.
    """
    ratio = torch.exp(policy.dist(qtype).log_prob(action) - old_logp)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    return -torch.min(unclipped, clipped).mean()               # negated: optimizers minimise


def ppo_update(policy, optimizer, batch, updates):
    """Identical to vpg_update -- only the loss line differs. First-order, plain SGD."""
    for _ in range(updates):
        loss = ppo_loss(policy, *batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------- the four methods
# name -> (how to update on one iteration's rollouts, what it does, what limits the step)
METHODS = {
    "VPG, each batch once": (lambda pol, opt, b: vpg_update(pol, opt, b, updates=1),
                             "L^PG, 1 epoch/iteration", "none needed"),
    "VPG, batch reused":    (lambda pol, opt, b: vpg_update(pol, opt, b, updates=REUSE),
                             f"L^PG, {REUSE} epochs/iteration", "none (!)"),
    "TRPO":                 (lambda pol, opt, b: trpo_update(pol, b, max_kl=MAX_KL, lr=LR, epochs=REUSE),
                             f"eq. 3, <={REUSE} epochs/iteration", "KL constraint"),
    "PPO":                  (lambda pol, opt, b: ppo_update(pol, opt, b, updates=REUSE),
                             f"L^CLIP, {REUSE} epochs/iteration", "clip, in the loss"),
}


def kl(p_old, p_new):
    """Exact KL(pi_old || pi_new), averaged over both question types."""
    return (0.5 * (p_old * (p_old / p_new).log()).sum(dim=-1)).sum().item()


def train(update, seed):
    """One run: collect -> update -> discard, until the rollout budget is spent.

    Returns the true J after every batch, whether the run got stuck, and the
    mean KL each iteration moved the policy (how far it went from theta_old).
    """
    torch.manual_seed(seed)                                   # same seed -> same first batch for every method
    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=LR)   # TRPO makes its own, same lr
    curve, kls = [], []
    for _ in range(TOTAL_ROLLOUTS // BATCH_SIZE):
        batch = collect_rollouts(policy, BATCH_SIZE)          # fresh batch from the current model
        p_old = policy.probs()                                # theta_old, before this batch's update(s)
        update(policy, optimizer, batch)
        kls.append(kl(p_old, policy.probs()))                 # how far this batch moved the model
        curve.append(policy.true_reward())
    # "Stuck": locked onto a wrong action. A probability near 0 is almost never
    # sampled again, so no future batch can bring the evidence to undo it.
    p = policy.probs()
    stuck = p[HARD, TOOL] < 0.05 or p[EASY, TOOL] > 0.95
    return curve, bool(stuck), sum(kls) / len(kls)


def main():
    torch.set_num_threads(1)                                  # tiny model: one thread is fastest
    n = TOTAL_ROLLOUTS // BATCH_SIZE
    marks = [n // 5 - 1, n // 2 - 1, n - 1]                   # after 20%, 50%, 100% of the budget
    print(f"{TOTAL_ROLLOUTS} rollouts per run, batch {BATCH_SIZE}, lr {LR}, {SEEDS} seeds. "
          f"Start J 0.60, best {BEST_REWARD}.\n")
    print(f"{'method':<21} | {'what it does':<29} | {'J@160':>5} | {'J@400':>5} | {'J@800':>5} | "
          f"{'stuck':>5} | {'KL/iter':>8} | {'step limit':<18}")
    curves = {}
    for name, (update, what, limit) in METHODS.items():
        runs = [train(update, seed) for seed in range(SEEDS)]
        curve = [sum(r[0][i] for r in runs) / SEEDS for i in range(n)]   # mean J after each batch
        stuck = sum(r[1] for r in runs)
        mean_kl = sum(r[2] for r in runs) / SEEDS
        curves[name] = curve
        print(f"{name:<21} | {what:<29} | " + " | ".join(f"{curve[i]:>5.2f}" for i in marks)
              + f" | {stuck:>2}/{SEEDS} | {mean_kl:>8.4f} | {limit:<18}")

    print("""
  VPG once      correct but slow: every expensive batch buys one small update.
  VPG reused    overshoots: L^PG keeps acting on a reading taken at theta_old, so a
                misleading batch gets locked in (the 'stuck' runs, the largest KL/iter).
  TRPO          the KL constraint keeps each iteration's move small (KL/iter <= delta):
                safe, but it is outside the loss -- it must be checked after every update.
  PPO           the clip builds the same limit into the loss: plain SGD, safe AND simple.""")
    save_figure(curves)


def save_figure(curves):
    """True J vs rollouts used, one line per method. Skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")                                 # write a file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return
    rollouts = [BATCH_SIZE * (i + 1) for i in range(TOTAL_ROLLOUTS // BATCH_SIZE)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in curves.items():
        ax.plot(rollouts, curve, label=name)
    ax.axhline(BEST_REWARD, color="gray", linestyle="--", linewidth=1, label="best possible")
    ax.set_xlabel("rollouts used")
    ax.set_ylabel(f"true reward J (mean of {SEEDS} seeds)")
    ax.legend()
    fig.tight_layout()
    path = Path(__file__).resolve().parent / "vpg_trpo_ppo.png"
    fig.savefig(path, dpi=120)
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
