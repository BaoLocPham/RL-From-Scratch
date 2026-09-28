"""TRPO, one step at a time: ``python TRPO/steps_trpo.py``.

Walks the notes' running example through every piece of TRPO, printing each
equation and the numbers substituted into it. Set ``RL_IMPL=scratch`` to run
the same walkthrough on your TRPO/from_scratch/trpo.py, and
``./scripts/run_trpo.sh diff`` to compare the two outputs line by line.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "VPG"))           # the toy, always the reference
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
else:
    sys.path.insert(0, str(HERE))                      # the reference
from trpo import mean_kl, surrogate, trpo_update  # noqa: E402
from vpg import (ANSWER, HARD, TOOL, Policy, collect_rollouts,  # noqa: E402
                 pg_loss, reuse_one_batch)


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def policy_at(p_tool_hard):
    """A policy whose HARD row is (1 - p, p); the EASY row stays at the start, 0.40."""
    policy = Policy()
    with torch.no_grad():
        policy.logits[HARD] = torch.tensor([1 - p_tool_hard, p_tool_hard]).log()
    return policy


# The running example from the notes: 10 HARD questions answered by the model at
# p(tool) = 0.40. 4 called the tool (A = +3); 6 answered directly (A = -2).
QTYPE = torch.zeros(10, dtype=torch.long)
ACTION = torch.tensor([TOOL] * 4 + [ANSWER] * 6)
ADV = torch.tensor([3.] * 4 + [-2.] * 6)
OLD_LOGP = torch.tensor([0.4] * 4 + [0.6] * 6).log()
BATCH = (QTYPE, ACTION, ADV, OLD_LOGP)
MAX_KL = 0.01
LR = 0.02               # small, so several epochs fit inside the trust region

# ---------------------------------------------------------------- step 1
banner("STEP 1  the ratio surrogate: mean( pi_theta / pi_theta_old * A )     (eq. 3)")
print("  At theta_old the model IS the one that collected the batch, so every ratio is 1:")
print(f"  surrogate(theta_old) = mean(1 * A) = (4 * 3 + 6 * -2) / 10 = "
      f"{round(float(surrogate(Policy(), *BATCH).detach()), 4) + 0.0:+.4f}")   # + 0.0: no -0.0
moved = policy_at(0.48)
print("  After the model moves to p(tool|HARD) = 0.48, the ratio remembers theta_old:")
print(f"  tool calls:     0.48 / 0.40 = {0.48 / 0.40:.3f},  * A +3 = {0.48 / 0.40 * 3:+.3f}   (x4)")
print(f"  direct answers: 0.52 / 0.60 = {0.52 / 0.60:.3f},  * A -2 = {0.52 / 0.60 * -2:+.3f}   (x6)")
print(f"  surrogate(0.48) = {float(surrogate(moved, *BATCH).detach()):+.4f}")

# ---------------------------------------------------------------- step 2
banner("STEP 2  at theta_old, eq. 3 and L^PG have the same gradient")
policy = Policy()
surrogate(policy, *BATCH).backward()
grad_surr = policy.logits.grad[HARD].clone()
policy = Policy()
(-pg_loss(policy, *BATCH)).backward()                  # pg_loss is -L^PG, so negate it back
grad_pg = policy.logits.grad[HARD]
print(f"  grad surrogate on HARD's logits [answer, tool] = [{grad_surr[0]:+.3f}, {grad_surr[1]:+.3f}]")
print(f"  grad L^PG      on HARD's logits [answer, tool] = [{grad_pg[0]:+.3f}, {grad_pg[1]:+.3f}]")
print("  The same first step. The difference comes after the model moves: eq. 3 divides")
print("  by pi_theta_old, so it knows the batch is old. L^PG does not.")

# ---------------------------------------------------------------- step 3
banner("STEP 3  the KL: mean( sum_x P(x) * (ln P(x) - ln Q(x)) ),  P = old, Q = new   (eq. 4)")
old_probs = Policy().probs()[QTYPE]
for p in (0.45, 0.48):
    kl = float(mean_kl(policy_at(p), QTYPE, old_probs).detach())
    answer = 0.6 * (torch.tensor(0.6).log() - torch.tensor(1 - p).log())
    tool = 0.4 * (torch.tensor(0.4).log() - torch.tensor(p).log())
    verdict = "inside" if kl <= MAX_KL else "OUTSIDE"
    print(f"  0.40 -> {p:.2f}:  answer 0.60 * (ln 0.60 - ln {1 - p:.2f}) = {answer:+.4f}")
    print(f"                tool   0.40 * (ln 0.40 - ln {p:.2f}) = {tool:+.4f}")
    print(f"                KL = {kl:.4f}   -> {verdict} the trust region (delta = {MAX_KL})")
print("  KL uses BOTH actions at every rollout, not just the one taken.")

# ---------------------------------------------------------------- step 4
banner(f"STEP 4  trpo_update: epochs on the same batch until the KL passes {MAX_KL},  lr = {LR}")
print(f"  {'epoch':>5} | {'p(tool|HARD)':>12} | {'KL from theta_old':>17} |")
for epoch in range(1, 8):
    policy = Policy()
    trpo_update(policy, BATCH, max_kl=float("inf"), lr=LR, epochs=epoch)   # no fence: just to measure
    kl = float(mean_kl(policy, QTYPE, old_probs).detach())
    note = "kept" if kl <= MAX_KL else "over delta: undone, stop"
    print(f"  {epoch:>5} | {policy.probs()[HARD, TOOL]:>12.4f} | {kl:>17.5f} | {note}")
    if kl > MAX_KL:
        break
policy = Policy()
kept = trpo_update(policy, BATCH, max_kl=MAX_KL, lr=LR)
print(f"  trpo_update(max_kl={MAX_KL}) kept {kept} epochs: p(tool|HARD) 0.4000 -> "
      f"{policy.probs()[HARD, TOOL]:.4f}")
print("  Between 0.45 and 0.48, as on the Notion page: more than 1 epoch from one batch,")
print("  but never outside the trust region.")

# ---------------------------------------------------------------- step 5
banner("STEP 5  the unlucky batch that broke VPG (seed 17), up to 100 epochs, lr 0.3")
torch.manual_seed(17)
policy = Policy()
batch = collect_rollouts(policy, 16)
start_j = policy.true_reward()
kept = trpo_update(policy, batch, max_kl=MAX_KL, lr=0.3, epochs=100)
vpg_j = reuse_one_batch(seed=17, updates=100)[-1][2]
print(f"  {'':<24} | {'epochs used':>11} | {'true J':>6}")
print(f"  {'start':<24} | {'':>11} | {start_j:>6.3f}")
print(f"  {'VPG, no KL constraint':<24} | {100:>11} | {vpg_j:>6.3f}")
print(f"  {'TRPO, max_kl 0.01':<24} | {kept:>11} | {policy.true_reward():>6.3f}")
print("  The batch misleads both. The KL fence stops TRPO after a few epochs, so a bad")
print("  batch costs little. ./scripts/run_trpo.sh run shows the next, fresh batches repair it.")
