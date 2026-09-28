"""Staged grader for ``surrogates.py``. Expected constants are reference outputs."""

import inspect
import sys
import traceback
from functools import partial
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import surrogates as sol
from loop import Policy, collect_rollouts, mean_kl, train  # noqa: E402

# The running example from the notes: 10 rollouts on HARD questions, collected at
# p(tool) = 0.40 -- 4 tool calls that went well (A = +3), 6 direct answers (A = -2).
QTYPE = torch.zeros(10, dtype=torch.long)
ACTION = torch.tensor([1] * 4 + [0] * 6)
ADV = torch.tensor([3.] * 4 + [-2.] * 6)
OLD_LOGP = torch.tensor([0.4] * 4 + [0.6] * 6).log()
BATCH = (QTYPE, ACTION, ADV, OLD_LOGP)
OLD_PROBS = torch.tensor([[0.6, 0.4]] * 10)

GRAD_AT_OLD = [1.2, -1.2]                           # d(loss)/d logits[HARD] at theta_old, for every slot
CPI_048 = -0.40                                     # loss at p(tool|HARD) = 0.48
LPG_048 = 0.09605130                                # -L^PG there: log pi instead of the ratio
LOG_RATIO_048 = -0.39050689                         # forgot the exp
INV_RATIO_048 = 0.38461536                          # ratio upside down
PEN_048_B3 = -0.36120433                            # -(0.40 - 3 * 0.0129)
PEN_048_B3_PLUS = -0.43879569                       # -(0.40 + 3 * 0.0129): the KL rewarded
PEN_080_B03, PEN_080_B10 = -1.88542795, 1.81908463  # p(tool) = 0.80: cheap vs dear moving
CLIP = {0.48: -0.40, 0.60: -0.48, 0.28: 0.60, 0.80: -0.48, 0.20: 1.0}   # loss at each p(tool|HARD)
CLIP_NO_MIN_028 = 0.48                              # clipped term alone at p = 0.28
CLIP_MAX_060 = -1.0                                 # max instead of min at p = 0.60

# Stage 5: 50 epochs on the unlucky seed-17 batch, lr 0.3 -> (true J, KL moved)
SEED17 = {"L^CPI": (0.44337857, 0.21466628), "KL penalty, beta 0.3": (0.47234333, 0.12274474),
          "KL penalty, beta 3": (0.56419355, 0.00712964), "KL penalty, beta 10": (0.58843249, 0.00071407),
          "L^CLIP, eps 0.2": (0.53923631, 0.02446184)}
ADAPTIVE_SEED0 = (0.88584727, 2.0)                  # 50 iterations: final true J, final beta


STAGE = 0                                           # the stage being graded, for messages


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open Surrogates/from_scratch/surrogates.py, fill the "
                f"lines marked 'TODO stage {STAGE}', and try them with "
                "`python Surrogates/from_scratch/surrogates.py`")


def has_blank(fn):
    """A `...` left on a 'TODO stage N' line. Bare `...` and `if ...:` never raise, so look."""
    fn = getattr(fn, "func", fn)                    # a functools.partial: grade the function inside
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
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def close(actual, expected, atol=1e-5):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def policy_at(p_tool_hard):
    policy = Policy()
    with torch.no_grad():
        policy.logits[0] = torch.tensor([1 - p_tool_hard, p_tool_hard]).log()
    return policy


def grad_at_old(slot):
    """The slot's gradient on HARD's logits at theta_old."""
    policy = Policy()
    loss = call(slot, policy, BATCH, OLD_PROBS)
    need(isinstance(loss, torch.Tensor) and loss.requires_grad,
         "the loss must carry gradient: build it from policy.dist(...), and do not use .item() "
         "or torch.no_grad()")
    need(loss.numel() == 1, f"return one number (the mean), got shape {tuple(loss.shape)}")
    loss.backward()
    return policy.logits.grad[0]


def stage_1():
    ratio = call(sol.ratio_of, Policy(), BATCH)
    if close(ratio, torch.zeros(10)):
        raise Fail("got 0 at theta_old: that is the LOG of the ratio. Exponentiate the difference")
    need(close(ratio, torch.ones(10)), f"at theta_old every ratio is 1, got {torch.as_tensor(ratio).tolist()}")
    ratio = call(sol.ratio_of, policy_at(0.48), BATCH)
    if close(ratio[0], 0.4 / 0.48):
        raise Fail("the ratio is upside down: new over old, exp(logp - old_logp)")
    if close(ratio[0], torch.tensor(1.2).log()):
        raise Fail("that is the LOG of the ratio; exponentiate it")
    need(close(ratio, torch.tensor([1.2] * 4 + [0.52 / 0.6] * 6)),
         f"at p(tool) = 0.48 the ratios are 1.2 (tool) and 0.867 (answer), got {torch.as_tensor(ratio).tolist()}")

    got = call(sol.cpi_loss, policy_at(0.48), BATCH, OLD_PROBS)
    if close(got, -CPI_048):
        raise Fail("got +0.40: that is L^CPI itself. A slot returns a LOSS, minus the objective")
    if close(got, LPG_048):
        raise Fail("that is -L^PG (log pi * A). L^CPI uses the ratio")
    need(close(got, CPI_048), f"at p(tool) = 0.48 expected {CPI_048}, got {num(got):.4f}")
    need(close(grad_at_old(sol.cpi_loss), torch.tensor(GRAD_AT_OLD)),
         f"at theta_old the gradient should be {GRAD_AT_OLD}, the same as L^PG's")


def stage_2():
    got = call(sol.kl_penalty_loss, policy_at(0.48), BATCH, OLD_PROBS, beta=3)
    if close(got, PEN_048_B3_PLUS):
        raise Fail("the KL is being rewarded: SUBTRACT beta * KL from the objective (then negate)")
    if close(got, -PEN_048_B3):
        raise Fail("got +0.3612: that is the objective. A slot returns minus it")
    if close(got, CPI_048):
        raise Fail("got L^CPI's -0.40: the beta * KL term is missing")
    need(close(got, PEN_048_B3), f"at p(tool) = 0.48, beta 3: expected {PEN_048_B3:.4f}, got {num(got):.4f}\n"
         "  -(0.40 - 3 * 0.0129): L^CPI minus beta times mean_kl(policy, qtype, old_probs)")
    need(close(call(sol.kl_penalty_loss, policy_at(0.48), BATCH, OLD_PROBS, beta=0), CPI_048),
         "with beta = 0 the penalty should vanish and leave exactly L^CPI")
    need(close(call(sol.kl_penalty_loss, policy_at(0.80), BATCH, OLD_PROBS, beta=0.3), PEN_080_B03) and
         close(call(sol.kl_penalty_loss, policy_at(0.80), BATCH, OLD_PROBS, beta=10), PEN_080_B10),
         f"at p(tool) = 0.80: beta 0.3 should give {PEN_080_B03:.4f} and beta 10 {PEN_080_B10:.4f}")
    need(close(grad_at_old(partial(sol.kl_penalty_loss, beta=3)), torch.tensor(GRAD_AT_OLD)),
         f"at theta_old the KL term has zero gradient, so the slot's gradient should be {GRAD_AT_OLD}")


def stage_3():
    cases = [(0.003, 0.5, "too timid"), (0.013, 1.0, "inside the band"), (0.03, 2.0, "too far"),
             (0.01 / 1.5, 1.0, "exactly on the lower edge"), (0.015, 1.0, "exactly on the upper edge"),
             (0.0, 0.5, "not moving at all"), (1.0, 2.0, "far out")]
    got = [call(sol.adapt_beta, 1.0, d, 0.01) for d, _, _ in cases]
    if close(got[0], 2.0) and close(got[2], 0.5):
        raise Fail("halving and doubling are swapped: moving too far should make moving DEARER (double)")
    if close(got[2], 1.5) or close(got[0], 1 / 1.5):
        raise Fail("1.5 is the band's width; the step is 2: halve or double")
    if close(got[1], 0.5) or close(got[1], 2.0):
        raise Fail("d = 0.013 is inside the band [d_targ / 1.5, d_targ * 1.5] = [0.0067, 0.015]: keep beta")
    for (d, expected, why), value in zip(cases, got):
        need(close(value, expected), f"adapt_beta(1.0, d={d:.4g}, d_targ=0.01) ({why}): expected {expected}, "
             f"got {value}")
    need(close(call(sol.adapt_beta, 3.0, 0.12, 0.1), 3.0) and close(call(sol.adapt_beta, 3.0, 0.02, 0.1), 1.5),
         "the band must follow d_targ: with d_targ = 0.1 it is [0.067, 0.15], so d = 0.12 is inside "
         "and d = 0.02 below")


def stage_4():
    for p, expected in CLIP.items():
        got = call(sol.clip_loss, policy_at(p), BATCH, OLD_PROBS)
        if p == 0.28 and close(got, CLIP_NO_MIN_028):
            raise Fail("the clipped term alone throws away the penalty for moving the WRONG way. "
                       "Take min(unclipped, clipped)")
        if p == 0.60 and close(got, CLIP_MAX_060):
            raise Fail("that is the max; L^CLIP is the pessimistic MIN of the two")
        if close(got, -expected) and expected != 0:
            raise Fail(f"got {num(got):.4f}: that is L^CLIP itself. A slot returns minus it")
        need(close(got, expected), f"at p(tool) = {p}: expected loss {expected}, got {num(got):.4f}")
    need(close(grad_at_old(sol.clip_loss), torch.tensor(GRAD_AT_OLD)),
         f"inside the range the clip does nothing: the gradient at theta_old should be {GRAD_AT_OLD}")

    tool_only = (QTYPE[:1], ACTION[:1], ADV[:1], OLD_LOGP[:1])       # one tool call, A = +3
    for p, flat in ((0.60, True), (0.28, False)):
        policy = policy_at(p)
        call(sol.clip_loss, policy, tool_only, OLD_PROBS[:1]).backward()
        moving = float(policy.logits.grad.abs().sum()) > 1e-6
        need(moving != flat, f"one tool call (A = +3) at ratio {p / 0.4:.1f}: "
             + ("past 1 + eps its gradient must be exactly 0 -- the model stops there" if flat
                else "moved the wrong way, so its gradient must still pull it back"))


def stage_5():
    """No new code: your slots on the batch that broke VPG, then adaptive beta over a full run."""
    print("  the unlucky batch (seed 17), 50 epochs, lr 0.3, no hooks -- only the slot differs:")
    print(f"  {'slot':<22} | {'true J':>6} | {'KL moved':>8}")
    torch.manual_seed(17)
    print(f"  {'start':<22} | {Policy().true_reward():>6.3f} |")
    slots = {"L^CPI": sol.cpi_loss, "KL penalty, beta 0.3": partial(sol.kl_penalty_loss, beta=0.3),
             "KL penalty, beta 3": partial(sol.kl_penalty_loss, beta=3),
             "KL penalty, beta 10": partial(sol.kl_penalty_loss, beta=10), "L^CLIP, eps 0.2": sol.clip_loss}
    for name, slot in slots.items():
        torch.manual_seed(17)
        policy = Policy()
        batch = collect_rollouts(policy, 16)
        old_probs = policy.probs()[batch[0]]
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
        for _ in range(50):
            loss = slot(policy, batch, old_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        j, kl = policy.true_reward(), num(mean_kl(policy, batch[0], old_probs))
        need(close(j, SEED17[name][0], atol=1e-4), f"{name}: expected J {SEED17[name][0]:.3f}, got {j:.3f}")
        print(f"  {name:<22} | {j:>6.3f} | {kl:>8.4f}")
    print("  The batch misleads every slot. L^CPI has no brake; beta decides how hard the fee brakes\n"
          "  (0.3 barely, 10 almost freezes it); the clip brakes softly, with a fixed eps.")

    adaptive = sol.AdaptiveKL()
    rows, _ = train(adaptive.slot, 0, 50, after_iteration=adaptive.after_iteration)
    need(close(rows[-1][0], ADAPTIVE_SEED0[0], atol=1e-4) and close(adaptive.beta, ADAPTIVE_SEED0[1]),
         f"adaptive KL, seed 0, 50 iterations: expected J {ADAPTIVE_SEED0[0]:.3f} and final beta "
         f"{ADAPTIVE_SEED0[1]}, got {rows[-1][0]:.3f} and {adaptive.beta}")
    print(f"\n  adaptive beta, seed 0, 50 iterations: J {rows[-1][0]:.3f}, beta 1.0 -> {adaptive.beta:g}. "
          "The full comparison,\n  20 seeds of every slot, is ./scripts/run_surrogates.sh run")


STAGES = [
    ("L^CPI, the ratio with no limit (eq. 6)", stage_1),
    ("the fixed KL penalty (eq. 5)", stage_2),
    ("the adaptive beta rule (eq. 8)", stage_3),
    ("L^CLIP (eq. 7)", stage_4),
    ("every slot in the loop, with your code", stage_5),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "from_scratch/surrogates.py`" not in str(exc):
            print("\n  Tip: `python Surrogates/from_scratch/surrogates.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all Surrogates stages pass")
