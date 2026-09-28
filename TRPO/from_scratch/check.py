"""Staged grader for ``trpo.py``. Expected constants are reference outputs."""

import inspect
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trpo as sol

# The running example from the notes: 10 rollouts on HARD questions, collected at
# p(tool) = 0.40 -- 4 tool calls that went well (A = +3), 6 direct answers (A = -2).
QTYPE = torch.zeros(10, dtype=torch.long)
ACTION = torch.tensor([1] * 4 + [0] * 6)
ADV = torch.tensor([3.] * 4 + [-2.] * 6)
OLD_LOGP = torch.tensor([0.4] * 4 + [0.6] * 6).log()        # what the collecting model gave them
BATCH = (QTYPE, ACTION, ADV, OLD_LOGP)

SURR_048 = 0.40                                     # eq. 3 after moving to p(tool|HARD) = 0.48
LPG_048 = -0.09605130                               # L^PG there: log pi instead of the ratio
LOG_RATIO_048 = 0.39050689                          # mean(log ratio * A): forgot the exp
INV_RATIO_048 = -0.38461536                         # mean(pi_old / pi * A): ratio upside down
GRAD_HARD = [-1.2, 1.2]                             # d(surrogate)/d logits[HARD] at theta_old = L^PG's

KL_045, KL_048 = 0.00509366, 0.01293190             # the two moves on the Notion page
KL_080 = 0.38190851                                 # 0.40 -> 0.80
KL_080_REVERSED = 0.33479530                        # KL(new || old) for the same move
KL_MIXED = 0.33950669                               # 2 HARD + 3 EASY rollouts, both rows moved
KL_MIXED_SUM = 1.69753337                           # the same, summed over the batch instead of averaged
KL_MIXED_NO_P = 0.75068367                          # the same, without the P(x) weight

KEPT_002 = 5                                        # running example, lr 0.02, max_kl 0.01
AFTER_002 = [[-0.63276142, -0.79435486], [-0.51082557, -0.91629070]]
AFTER_002_NO_LIMIT_8 = [[-0.70749676, -0.71961951], [-0.51082557, -0.91629070]]
SEED17_KEPT, SEED17_J = 6, 0.55858028               # the unlucky batch, lr 0.3, up to 100 epochs
VPG_SEED17_J = 0.37252286                           # VPG's 100 epochs on it, no KL constraint


STAGE = 0                                           # the stage being graded, for messages


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open TRPO/from_scratch/trpo.py, fill the lines "
                f"marked 'TODO stage {STAGE}', and try them with `python TRPO/from_scratch/trpo.py`")


def has_blank(fn):
    """A `...` left on a 'TODO stage N' line. Bare `...` and `if ...:` never raise, so look."""
    for line in inspect.getsource(fn).splitlines():
        if f"TODO stage {STAGE}" in line:
            code = line.split("#")[0].strip().rstrip(":").strip()
            if code.endswith("..."):
                return True
    return False


def need(condition, message):
    if not condition:
        raise Fail(message)


def call(fn, *args, **kwargs):
    if has_blank(fn):
        raise unfinished()
    try:
        value = fn(*args, **kwargs)
    except Exception as exc:
        if "llipsis" in str(exc):
            raise unfinished() from exc
        raise Fail(f"raised {type(exc).__name__}: {exc}") from exc
    if value is Ellipsis:
        raise unfinished()
    return value


def num(x):
    """A float, whether you returned a tensor (with or without gradient) or a number."""
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def close(actual, expected, atol=1e-5):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def policy_at(p_tool_hard, p_tool_easy=0.4):
    policy = sol.Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.tensor([[1 - p_tool_hard, p_tool_hard],
                                          [1 - p_tool_easy, p_tool_easy]]).log())
    return policy


def stage_1():
    policy = sol.Policy()
    at_old = call(sol.surrogate, policy, *BATCH)
    need(isinstance(at_old, torch.Tensor) and at_old.requires_grad,
         "the surrogate must carry gradient: take logp from policy.dist(...), and do not "
         "use .item() or torch.no_grad()")
    need(at_old.numel() == 1, f"return one number (the mean), got shape {tuple(at_old.shape)}")

    moved = call(sol.surrogate, policy_at(0.48), *BATCH).detach()
    if close(moved, 0.0):
        raise Fail("got 0 after the model moved: the ratio stayed 1. Divide by pi_theta_old from "
                   "the batch (old_logp), not by the current policy")
    if close(moved, -SURR_048):
        raise Fail("got -0.40: that is the loss. surrogate() is the objective to go UP; "
                   "stage 3 puts the minus sign in front")
    if close(moved, LPG_048):
        raise Fail(f"got {num(moved):.4f}: that is L^PG (log pi * A, eq. 2). "
                   "Eq. 3 uses the ratio pi_theta / pi_theta_old")
    if close(moved, LOG_RATIO_048):
        raise Fail(f"got {num(moved):.4f}: that is the LOG of the ratio. Exponentiate the "
                   "difference of log-probs: exp(log a - log b) = a / b")
    if close(moved, INV_RATIO_048):
        raise Fail(f"got {num(moved):.4f}: the ratio is upside down. New over old: "
                   "exp(logp - old_logp)")
    need(close(moved, SURR_048), f"at p(tool|HARD) = 0.48 expected {SURR_048:.4f}, got {num(moved):.4f}\n"
         "  tool: 0.48/0.40 = 1.200 * 3,  answer: 0.52/0.60 = 0.867 * -2,  averaged over 10")
    need(close(at_old.detach(), 0.0), f"at theta_old every ratio is 1, so the surrogate is mean(A) = 0, "
         f"got {num(at_old):.4f}")

    other = call(sol.surrogate, policy_at(0.48), QTYPE, ACTION, ADV, torch.full((10,), -0.1)).detach()
    need(not close(other, moved),
         "the value did not change when old_logp changed. Eq. 3 divides by pi_theta_old: "
         "that is how it remembers which model collected the batch")

    at_old.backward()
    need(close(policy.logits.grad[sol.HARD], torch.tensor(GRAD_HARD)),
         f"gradient on the HARD logits at theta_old should be {GRAD_HARD} (the same as L^PG's), "
         f"got {policy.logits.grad[sol.HARD].tolist()}")


def stage_2():
    old = sol.Policy().probs()[QTYPE]
    same = call(sol.mean_kl, sol.Policy(), QTYPE, old)
    need(torch.as_tensor(same).numel() == 1, f"return one number (the mean), got shape "
         f"{tuple(torch.as_tensor(same).shape)}")
    need(close(same, 0.0), f"an unchanged policy has KL 0, got {num(same):.5f}")

    got = call(sol.mean_kl, policy_at(0.48), QTYPE, old)
    far = call(sol.mean_kl, policy_at(0.80), QTYPE, old)
    qtype = torch.tensor([0, 0, 1, 1, 1])
    mixed = call(sol.mean_kl, policy_at(0.80, 0.10), qtype, sol.Policy().probs()[qtype])
    if close(far, KL_080_REVERSED):
        raise Fail("that is KL(new || old). Eq. 4 is KL(old || new): weight by the OLD "
                   "probabilities, P = old_probs")
    if close(mixed, KL_MIXED_SUM):
        raise Fail("that is the SUM over the batch; eq. 4 averages over it (.mean())")
    if close(mixed, KL_MIXED_NO_P):
        raise Fail("each action's log-ratio must be weighted by its old probability P(x)")

    need(close(got, KL_048), f"0.40 -> 0.48 should give {KL_048:.5f}, got {num(got):.5f}\n"
         "  answer: 0.60 * (ln 0.60 - ln 0.52),  tool: 0.40 * (ln 0.40 - ln 0.48),  add them")
    need(close(call(sol.mean_kl, policy_at(0.45), QTYPE, old), KL_045),
         f"0.40 -> 0.45 should give {KL_045:.5f}")
    need(close(far, KL_080), f"0.40 -> 0.80 should give {KL_080:.5f}, got {num(far):.5f}")
    need(close(mixed, KL_MIXED), f"a mixed batch (2 HARD, 3 EASY) should give {KL_MIXED:.5f}, "
         f"got {num(mixed):.5f}\n  use policy.logits[qtype], so each rollout gets its own question's row")


def stage_3():
    policy = sol.Policy()
    start = policy.logits.detach().clone()
    kept = call(sol.trpo_update, policy, BATCH, max_kl=0.01, lr=0.02)
    need(isinstance(kept, int), f"return the number of epochs kept, an int; got {kept!r}")
    after = policy.logits.detach()
    if close(after, start) and kept == 0:
        raise Fail("the model did not move and 0 epochs were kept. With lr 0.02 the first "
                   "step's KL is 0.0003, well inside 0.01: is the KL check the right way round?")
    need(policy.probs()[sol.HARD, sol.TOOL] > 0.4,
         "p(tool|HARD) went DOWN, but the tool calls had A = +3. The loss is MINUS the surrogate")
    if kept == 50:
        raise Fail("all 50 epochs were kept: the KL constraint never fired. old_probs must be taken "
                   "ONCE, before the loop -- it is the centre of the trust region, and a centre that "
                   "follows the policy is always at distance 0")
    final_kl = num(sol.mean_kl(policy, QTYPE, sol.Policy().probs()[QTYPE]))
    if final_kl > 0.01:
        raise Fail(f"the policy ended outside the trust region (KL {final_kl:.4f} > 0.01): the step "
                   "that broke the constraint was not undone. Copy the logits with "
                   ".detach().clone() -- a plain .detach() shares memory and moves with them")
    if kept == KEPT_002 + 1:
        raise Fail(f"kept {kept} epochs, expected {KEPT_002}: the step that was undone should not count")
    need(kept == KEPT_002, f"running example, lr 0.02, max_kl 0.01: expected {KEPT_002} epochs kept, got {kept}")
    need(close(after, torch.tensor(AFTER_002)), f"expected logits {AFTER_002}, got {after.tolist()}")

    policy = sol.Policy()
    kept = call(sol.trpo_update, policy, BATCH, max_kl=1e9, lr=0.02, epochs=8)
    need(kept == 8 and close(policy.logits.detach(), torch.tensor(AFTER_002_NO_LIMIT_8)),
         "with no limit (max_kl = 1e9) every one of the 8 epochs should be kept, "
         f"ending at {AFTER_002_NO_LIMIT_8}; got {kept} epochs, {policy.logits.detach().tolist()}.\n"
         "  Is old_probs taken ONCE, before the loop? It is the centre of the trust region")

    policy = sol.Policy()
    kept = call(sol.trpo_update, policy, BATCH, max_kl=1e-6, lr=0.02)
    need(kept == 0 and close(policy.logits.detach(), start),
         "with max_kl = 1e-6 even the first step breaks the constraint: 0 epochs, logits unchanged")


def stage_4():
    """No new code: your trpo_update on the unlucky batch that broke VPG (seed 17)."""
    torch.manual_seed(17)
    policy = sol.Policy()
    batch = sol.collect_rollouts(policy, 16)
    start_j = policy.true_reward()
    kept = sol.trpo_update(policy, batch, max_kl=0.01, lr=0.3, epochs=100)
    need(kept == SEED17_KEPT and close(policy.true_reward(), SEED17_J, atol=1e-4),
         f"seed 17, up to 100 epochs: expected {SEED17_KEPT} kept and J {SEED17_J:.3f}, "
         f"got {kept} and {policy.true_reward():.3f}")
    print("  the unlucky batch from ./scripts/run_vpg.sh run (seed 17), up to 100 epochs, lr 0.3:")
    print(f"  {'':<28} | {'epochs used':>11} | {'true J':>6}")
    print(f"  {'start':<28} | {'':>11} | {start_j:>6.3f}")
    print(f"  {'VPG, no KL constraint':<28} | {100:>11} | {VPG_SEED17_J:>6.3f}")
    print(f"  {'your TRPO, max_kl 0.01':<28} | {kept:>11} | {policy.true_reward():>6.3f}")
    print("  The batch is misleading either way; the KL fence stops TRPO after a few epochs, "
          "so it does far less damage.")


STAGES = [
    ("the ratio surrogate, eq. 3", stage_1),
    ("the KL, eq. 4", stage_2),
    ("the update behind the KL fence", stage_3),
    ("the fix, with your code", stage_4),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "from_scratch/trpo.py`" not in str(exc):
            print("\n  Tip: `python TRPO/from_scratch/trpo.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all TRPO stages pass")
