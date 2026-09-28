"""The slot's candidates, one step at a time: ``python Surrogates/steps_surrogates.py``.

Walks the notes' running example through every candidate for the slot, printing
each equation and the numbers substituted into it. Set ``RL_IMPL=scratch`` to
run the same walkthrough on your Surrogates/from_scratch/surrogates.py, and
``./scripts/run_surrogates.sh diff`` to compare the two outputs line by line.
"""

import os
import sys
from functools import partial
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # loop.py, always the reference
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your slots
from surrogates import adapt_beta, clip_loss, cpi_loss, kl_penalty_loss  # noqa: E402
from loop import (ANSWER, HARD, TOOL, Policy, collect_rollouts, kl_rollback,  # noqa: E402
                  mean_kl, pg_slot)


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def policy_at(p_tool_hard):
    """A policy whose HARD row is (1 - p, p); the EASY row stays at the start, 0.40."""
    policy = Policy()
    with torch.no_grad():
        policy.logits[HARD] = torch.tensor([1 - p_tool_hard, p_tool_hard]).log()
    return policy


def value(x):
    return round(float(x.detach()), 4) + 0.0          # + 0.0: no -0.0


# The running example: 10 HARD questions at p(tool) = 0.40, 4 tool calls (A = +3), 6 answers (A = -2).
QTYPE = torch.zeros(10, dtype=torch.long)
BATCH = (QTYPE, torch.tensor([TOOL] * 4 + [ANSWER] * 6), torch.tensor([3.] * 4 + [-2.] * 6),
         torch.tensor([0.4] * 4 + [0.6] * 6).log())
OLD_PROBS = Policy().probs()[QTYPE]

# ---------------------------------------------------------------- step 1
banner("STEP 1  one loop, one slot: every candidate starts in the same direction")
print("""  for iteration:  collect a batch at theta_old
      for epoch in range(K):
          loss = SLOT(policy, batch, old_probs)      <- the only line that changes
          zero_grad, backward, step
          AFTER_STEP     (TRPO's rollback only)
      AFTER_ITERATION    (adaptive beta only)""")
print("  gradient of each slot on HARD's logits [answer, tool], at theta_old:")
slots = {"L^PG  (VPG)": pg_slot, "L^CPI": cpi_loss, "KL penalty, beta 3": partial(kl_penalty_loss, beta=3),
         "L^CLIP, eps 0.2": clip_loss}
for name, slot in slots.items():
    policy = Policy()
    slot(policy, BATCH, OLD_PROBS).backward()
    g = policy.logits.grad[HARD]
    print(f"    {name:<20} [{g[0]:+.3f}, {g[1]:+.3f}]")
print("  Identical: the ratio is 1, the KL's gradient is 0 and the clip is inactive at theta_old.")
print("  They differ only AFTER the model moves -- which is exactly when reusing the batch matters.")

# ---------------------------------------------------------------- step 2
banner("STEP 2  L^CPI = mean( r * A ),  r = pi / pi_old      (eq. 6): reuse with no brake")
for p in (0.48, 0.70, 0.99):
    r_tool, r_answer = p / 0.4, (1 - p) / 0.6
    print(f"  p(tool) {p:.2f}: r = {r_tool:.3f} (tool), {r_answer:.3f} (answer)   "
          f"L^CPI = {-value(cpi_loss(policy_at(p), BATCH, OLD_PROBS)):+.4f}")
print("  It keeps rising all the way to 'always call the tool': nothing in it says stop.")

# ---------------------------------------------------------------- step 3
banner("STEP 3  the fee: L^CPI - beta * KL      (eq. 5)")
print(f"  {'':<12} | {'p(tool) 0.45':>13} | {'p(tool) 0.48':>13} | better move")
kls = {p: value(mean_kl(policy_at(p), QTYPE, OLD_PROBS)) for p in (0.45, 0.48)}
print(f"  {'KL':<12} | {kls[0.45]:>13.4f} | {kls[0.48]:>13.4f} |")
for beta in (0.3, 3, 30):
    obj = {p: -value(kl_penalty_loss(policy_at(p), BATCH, OLD_PROBS, beta)) for p in (0.45, 0.48)}
    better = "0.48, the bigger move" if obj[0.48] > obj[0.45] else "0.45, the smaller move"
    print(f"  {f'beta {beta}':<12} | {obj[0.45]:>13.4f} | {obj[0.48]:>13.4f} | {better}")
print("  Same evidence, different winner: beta, not the data, decides how far to go. And beta is")
print("  a PRICE in reward per nat: double every advantage and beta 30 behaves like beta 15.")

# ---------------------------------------------------------------- step 4
banner("STEP 4  adaptive beta: re-price after each iteration      (eq. 8)")
d_targ = 0.01
print(f"  d_targ {d_targ}: keep beta while d is in [{d_targ / 1.5:.4f}, {d_targ * 1.5:.4f}]")
beta = 1.0
for d in (0.030, 0.030, 0.012, 0.004, 0.004):
    new = adapt_beta(beta, d, d_targ)
    verdict = "too far -> double" if new > beta else "too timid -> halve" if new < beta else "inside -> keep"
    print(f"  moved d = {d:.3f}: {verdict:<18} beta {beta:g} -> {new:g}")
    beta = new
print("  The rule acts BETWEEN iterations: it prices the next update, it cannot stop this one.")

# ---------------------------------------------------------------- step 5
banner("STEP 5  the clip: min( r * A, clip(r, 0.8, 1.2) * A )      (eq. 7)")
for action, advantage, old_p, ratios in ((TOOL, 3., 0.4, (1.0, 1.1, 1.5, 0.7)),
                                         (ANSWER, -2., 0.6, (1.0, 0.9, 0.6, 1.3))):
    name = "tool call" if action == TOOL else "direct answer"
    print(f"  one {name}, A = {advantage:+.0f}:")
    for r in ratios:
        p_action = old_p * r
        p_tool = p_action if action == TOOL else 1 - p_action
        policy = policy_at(p_tool)
        one = (QTYPE[:1], torch.tensor([action]), torch.tensor([advantage]), torch.tensor([old_p]).log())
        loss = clip_loss(policy, one, OLD_PROBS[:1])
        loss.backward()
        moving = float(policy.logits.grad.abs().sum()) > 1e-6
        clipped = min(max(r, 0.8), 1.2) * advantage
        print(f"    r {r:.1f}: r*A {r * advantage:+.2f}, clipped {clipped:+.2f}, min {-value(loss):+.2f}  "
              f"-> {'still moving' if moving else 'flat: gradient 0, stops here'}")
print("  The clip only ever removes CREDIT (flat once moved 0.2 the helpful way), never a penalty.")

# ---------------------------------------------------------------- step 6
banner("STEP 6  every candidate on the batch that broke VPG (seed 17), 50 epochs, lr 0.3")
print(f"  {'slot':<24} | {'epochs kept':>11} | {'true J':>6} | {'KL moved':>8}")
torch.manual_seed(17)
print(f"  {'start':<24} | {'':>11} | {Policy().true_reward():>6.3f} |")
candidates = {"L^PG (VPG, reused)": (pg_slot, None), "L^CPI": (cpi_loss, None),
              "TRPO, delta 0.01": (cpi_loss, kl_rollback(0.01)),
              "KL penalty, beta 0.3": (partial(kl_penalty_loss, beta=0.3), None),
              "KL penalty, beta 3": (partial(kl_penalty_loss, beta=3), None),
              "KL penalty, beta 10": (partial(kl_penalty_loss, beta=10), None),
              "L^CLIP, eps 0.2": (clip_loss, None)}
for name, (slot, after_step) in candidates.items():
    torch.manual_seed(17)
    policy = Policy()
    batch = collect_rollouts(policy, 16)
    old_probs = policy.probs()[batch[0]]
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
    kept = 0
    for _ in range(50):
        before = policy.logits.detach().clone()
        loss = slot(policy, batch, old_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if after_step is not None and after_step(policy, batch, old_probs, before):
            break
        kept += 1
    print(f"  {name:<24} | {kept:>11} | {policy.true_reward():>6.3f} | {value(mean_kl(policy, batch[0], old_probs)):>8.4f}")
print("  One misleading batch, seven slots. The fixed limits (delta, eps) brake by distance; beta")
print("  brakes by price. ./scripts/run_surrogates.sh run shows 20 seeds of whole training runs.")
