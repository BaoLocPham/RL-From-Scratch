"""Staged grader for ``vpg.py``. Expected constants are reference outputs."""

import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vpg as sol

# The running example from the notes: 10 rollouts on HARD questions, collected at
# p(tool) = 0.40 -- 4 tool calls that went well (A = +3), 6 direct answers (A = -2).
QTYPE = torch.zeros(10, dtype=torch.long)
ACTION = torch.tensor([1] * 4 + [0] * 6)
ADV = torch.tensor([3.] * 4 + [-2.] * 6)
OLD_LOGP = torch.tensor([0.4] * 4 + [0.6] * 6).log()        # what the collecting model gave them

LOSS = 0.48655814                                   # -L^PG on that batch
GRAD_HARD = [1.2, -1.2]                             # d(-L^PG)/d logits[HARD]
AFTER_1 = [[-0.63082558, -0.79629070], [-0.51082557, -0.91629070]]    # logits, SGD lr 0.1
AFTER_3 = [[-0.87082559, -0.55629069], [-0.51082557, -0.91629070]]
AFTER_3_NO_ZERO_GRAD = [[-1.23082566, -0.19629067], [-0.51082557, -0.91629070]]
SEED17_J = {1: 0.59247243, 100: 0.37252286}         # true J on the unlucky batch


STAGE = 0                                           # the stage being graded, for messages


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open VPG/from_scratch/vpg.py, fill the lines "
                f"marked 'TODO stage {STAGE}', and try them with `python VPG/from_scratch/vpg.py`")


def need(condition, message):
    if not condition:
        raise Fail(message)


def call(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except TypeError as exc:
        if "llipsis" in str(exc):
            raise unfinished() from exc
        raise Fail(f"raised TypeError: {exc}") from exc
    except Exception as exc:
        raise Fail(f"raised {type(exc).__name__}: {exc}") from exc
    if value is Ellipsis:
        raise unfinished()
    return value


def close(actual, expected, atol=1e-5):
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def stage_1():
    start = torch.tensor([[0.6, 0.4], [0.6, 0.4]])
    got = call(sol.expected_reward, start)
    need(torch.as_tensor(got).numel() == 1, f"return one number (J), got shape {tuple(torch.as_tensor(got).shape)}")
    if close(got, 1.2):
        raise Fail("got 1.20: you added HARD and EASY; they are 50/50, so average them")
    if close(got, 0.62):
        raise Fail("got 0.62: MEAN_REWARD is indexed [type][action] -- it looks transposed")
    need(close(got, 0.6), f"the start policy should give J = 0.60, got {float(got):.4f}\n"
         "  HARD: 0.6*0.2 + 0.4*1.0 = 0.52,  EASY: 0.6*0.8 + 0.4*0.5 = 0.68")
    need(close(call(sol.expected_reward, torch.tensor([[0., 1.], [1., 0.]])), 0.9),
         "the best policy (always tool on HARD, always answer on EASY) should give J = 0.90")
    need(close(call(sol.expected_reward, torch.tensor([[0.5, 0.5], [0.9, 0.1]])), 0.685),
         "a mixed policy should give J = 0.685; weight each reward by its probability")


def stage_2():
    reward = torch.tensor([1., 2., 3., 6.])
    got = call(sol.compute_advantage, reward)
    if close(got, reward):
        raise Fail("returned the reward unchanged; subtract the batch-mean baseline")
    if close(got, (reward - reward.mean()) / reward.std()):
        raise Fail("that is whitened (divided by the std); only subtract the mean here")
    need(close(got, torch.tensor([-2., -1., 0., 3.])),
         f"expected [-2, -1, 0, 3], got {torch.as_tensor(got).tolist()}")
    need(abs(float(torch.as_tensor(got).mean())) < 1e-6, "advantages should average to zero")


def stage_3():
    policy = sol.Policy()
    loss = call(sol.pg_loss, policy, QTYPE, ACTION, ADV, OLD_LOGP)
    need(isinstance(loss, torch.Tensor) and loss.requires_grad,
         "the loss must carry gradient: take logp from policy.dist(...), and do not "
         "use .item() or torch.no_grad()")
    if close(loss, -LOSS):
        raise Fail(f"got {float(loss.detach()):.4f}: that is L^PG itself; optimizers minimise, so negate it")
    if close(loss, 0.0):
        raise Fail("got 0: that is the ratio pi/pi_old (eq. 3, TRPO / PPO), which is 1 at theta_old. "
                   "L^PG uses log pi and never looks at old_logp")
    need(close(loss, LOSS), f"expected {LOSS:.6f}, got {float(loss.detach()):.6f}\n"
         "  -mean(log pi(a|s) * A) over the 10 rollouts")
    other = call(sol.pg_loss, sol.Policy(), QTYPE, ACTION, ADV, torch.full((10,), -0.1))
    need(close(other, LOSS),
         "the loss changed when only old_logp changed. L^PG never looks at old_logp -- "
         "a ratio with it is eq. 3 (TRPO / PPO), not eq. 2")
    loss.backward()
    need(close(policy.logits.grad[sol.HARD], torch.tensor(GRAD_HARD)),
         f"gradient on the HARD logits should be {GRAD_HARD}, got {policy.logits.grad[sol.HARD].tolist()}")


def loss_is_unfinished():
    """Stage 4's first TODO is `loss = ...`; bare `...` lines never raise, so look for it."""
    import inspect
    return "loss = ...  " in inspect.getsource(sol.vpg_update)


def stage_4():
    batch = (QTYPE, ACTION, ADV, OLD_LOGP)
    policy = sol.Policy()
    start = policy.logits.detach().clone()
    call(sol.vpg_update, policy, torch.optim.SGD(policy.parameters(), lr=0.1), batch)
    after = policy.logits.detach()
    if close(after, start):
        if loss_is_unfinished():
            raise unfinished()
        raise Fail("the model did not move: call optimizer.step() after loss.backward()")
    need(policy.probs()[sol.HARD, sol.TOOL] > 0.4,
         "p(tool|HARD) went DOWN, but the tool calls had A = +3; check the loss sign")
    need(close(after, torch.tensor(AFTER_1)), f"one update: expected logits {AFTER_1}, got {after.tolist()}")

    policy = sol.Policy()
    call(sol.vpg_update, policy, torch.optim.SGD(policy.parameters(), lr=0.1), batch, updates=3)
    after = policy.logits.detach()
    if close(after, torch.tensor(AFTER_1)):
        raise Fail("updates=3 moved exactly as far as updates=1; loop `updates` times")
    if close(after, torch.tensor(AFTER_3_NO_ZERO_GRAD)):
        raise Fail("gradients piled up across updates; call optimizer.zero_grad() before backward()")
    need(close(after, torch.tensor(AFTER_3)), f"three updates: expected logits {AFTER_3}, got {after.tolist()}")


def stage_5():
    """No new code: your four pieces, run on the unlucky batch from VPG/vpg.py."""
    torch.manual_seed(17)
    policy = sol.Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
    batch = sol.collect_rollouts(policy, 16)
    rows = [(0, policy.probs(), policy.true_reward())]
    for update in range(1, 101):
        sol.vpg_update(policy, optimizer, batch)
        rows.append((update, policy.probs(), policy.true_reward()))
    for update, expected in SEED17_J.items():
        need(close(rows[update][2], expected, atol=1e-4),
             f"after {update} update(s) on the seed-17 batch J should be {expected:.3f}, got {rows[update][2]:.3f}")
    print("  your implementation, one batch reused (seed 17):")
    print(f"  {'update':>6} | {'p(tool|HARD)':>12} | {'p(tool|EASY)':>12} | {'true J':>6}")
    for update, p, j in rows:
        if update in (0, 1, 20, 100):
            print(f"  {update:>6} | {p[sol.HARD, sol.TOOL]:>12.2f} | {p[sol.EASY, sol.TOOL]:>12.2f} | {j:>6.3f}")
    print("  update 1 is vanilla PG; updates 2-100 act on a stale reading and drag J below the start.")


STAGES = [
    ("expected reward J", stage_1),
    ("advantage with a mean baseline", stage_2),
    ("the L^PG loss", stage_3),
    ("the update loop", stage_4),
    ("the flaw, with your code", stage_5),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "from_scratch/vpg.py`" not in str(exc):
            print(f"\n  Tip: `python VPG/from_scratch/vpg.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all VPG stages pass")
