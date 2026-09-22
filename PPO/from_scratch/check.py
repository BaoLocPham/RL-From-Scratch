"""Staged grader for ``ppo.py``. Expected constants are reference outputs."""

import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ppo as sol

R = torch.tensor([[0., 0., 1., 0.], [0., -1., 0., 0.]])
V = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
M = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])

GAE_ADV = torch.tensor([[0.71058786, 0.72572333, 0.75388032, 0.12816931],
                        [-1.06693876, -1.12325275, 0.12816931, 0.12816931]])
GAE_RET = torch.tensor([[0.75156754, 0.86849999, 1.0, 0.40000001],
                        [-0.83700001, -1.0, 0.30000001, 0.20000000]])
AGG = {"token-mean": 3.0, "seq-mean-token-sum": 7.5,
       "seq-mean-token-mean": 3.25, "seq-mean-token-sum-norm": 3.75}
POLICY = (0.29502711, 0.20000000, -0.40000000, 0.0)
POLICY_DUAL = -0.04200168
POLICY_DUAL_LOWER = 0.40000000
POLICY_ASYM = 0.27502713
VALUE = (0.54818946, 0.60000002)
ENTROPY = 0.81268251
KL = {"k1": [0.20000005, -0.09999999, -1.0, 0.0],
      "abs": [0.20000005, 0.09999999, 1.0, 0.0],
      "k2": [0.02000001, 0.00499999, 0.5, 0.0],
      "k3": [0.01873076, 0.00517094, 0.71828175, 0.0]}


class Fail(Exception):
    pass


def need(condition, message):
    if not condition:
        raise Fail(message)


def call(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except TypeError as exc:
        if "llipsis" in str(exc):
            raise Fail(f"a TODO is still an ellipsis: {exc}") from exc
        raise Fail(f"raised TypeError: {exc}") from exc
    except Exception as exc:
        raise Fail(f"raised {type(exc).__name__}: {exc}") from exc
    need(value is not None, "returned None; finish this stage's TODO")
    return value


def close(actual, expected, atol=1e-5):
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def stage_1():
    values = torch.tensor([[1., 2., 100.]])
    mask = torch.tensor([[1., 1., 0.]])
    got = call(sol.masked_mean, values, mask)
    if close(got, 34.333332):
        raise Fail("averaged over every position; the masked-out 100. must not count")
    need(close(got, 1.5), f"expected 1.5, got {float(got)}")
    need(torch.isfinite(torch.as_tensor(call(sol.masked_mean, values, torch.zeros(1, 3)))).all(),
         "an all-zero mask must stay finite; verl adds 1e-8 rather than clamping")

    spread = torch.tensor([[1., 2., 3., 100.]])
    mask4 = torch.tensor([[1., 1., 1., 0.]])
    got = call(sol.masked_var, spread, mask4)
    if close(got, 0.6666667):
        raise Fail("that is the population variance; unbiased=True applies n/(n-1) "
                   "over the number of MASKED positions")
    need(close(got, 1.0), f"expected 1.0, got {float(got)}")
    need(close(call(sol.masked_var, spread, mask4, unbiased=False), 0.6666667),
         "unbiased=False must skip the Bessel correction")

    got = call(sol.masked_whiten, spread, mask4)
    need(close(got[0, :3], torch.tensor([-1.0, 0.0, 1.0])),
         f"masked positions should whiten to [-1, 0, 1], got {got[0, :3].tolist()}")
    shifted = call(sol.masked_whiten, spread, mask4, shift_mean=False)
    need(close(shifted[0, :3], torch.tensor([1.0, 2.0, 3.0])),
         "shift_mean=False re-adds the original mean after scaling")


def stage_2():
    got = call(sol.compute_gae_advantage_return, R, V, M, 0.9, 0.95)
    need(isinstance(got, tuple) and len(got) == 2,
         "return the pair (advantages, returns)")
    advantages, returns = got
    if close(returns, GAE_ADV + V, atol=1e-3) and not close(advantages, GAE_ADV):
        raise Fail("returns look right but advantages do not; whiten the advantage "
                   "AFTER computing returns, not before")
    need(close(advantages, GAE_ADV),
         f"wrong advantages.\n  expected {GAE_ADV.tolist()}\n  got      {advantages.tolist()}\n"
         "  a masked-out position must CARRY the running values through, not reset "
         "them to zero -- response_mask is not a `dones` flag")
    need(close(returns, GAE_RET),
         f"wrong returns.\n  expected {GAE_RET.tolist()}\n  got      {returns.tolist()}\n"
         "  returns are advantage + values, taken before whitening")


def stage_3():
    loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
    for mode, expected in AGG.items():
        got = call(sol.agg_loss, loss, M, mode)
        if mode == "seq-mean-token-sum-norm" and close(got, 7.5):
            raise Fail("seq-mean-token-sum-norm divided by the number of responses; "
                       "Dr.GRPO's divisor is the padded width, loss_mask.shape[-1]")
        need(close(got, expected),
             f"{mode}: expected {expected}, got {float(got)}")
    try:
        sol.agg_loss(loss, M, "nonsense")
    except ValueError:
        pass
    except Fail:
        raise
    except Exception as exc:
        raise Fail(f"an unknown mode should raise ValueError, raised {type(exc).__name__}") from exc
    else:
        raise Fail("an unknown loss_agg_mode must raise ValueError")


def stage_4():
    old = torch.zeros(2, 4)
    # Row 1 carries a negative advantage and a ratio above clip_ratio_c, which is
    # the only situation where the dual clip does anything at all.
    logp = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
    advantages = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
    got = call(sol.compute_policy_loss, old, logp, advantages, M, cliprange=0.2)
    need(isinstance(got, tuple) and len(got) == 4,
         "return (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)")
    pg_loss, pg_clipfrac, ppo_kl, _ = got
    if close(pg_loss, -POLICY[0]):
        raise Fail("the sign is flipped; the surrogate is negated so gradient DESCENT "
                   "increases the objective")
    need(close(pg_loss, POLICY[0]),
         f"wrong pg_loss: expected {POLICY[0]:.8f}, got {float(pg_loss):.8f}")
    need(close(pg_clipfrac, POLICY[1]), f"wrong pg_clipfrac: got {float(pg_clipfrac)}")
    if close(ppo_kl, -POLICY[2]):
        raise Fail("ppo_kl has the wrong sign; it is the masked mean of the NEGATED "
                   "log-ratio")
    need(close(ppo_kl, POLICY[2]), f"wrong ppo_kl: got {float(ppo_kl)}")

    dual_loss, _, _, dual_lower = call(sol.compute_policy_loss, old, logp, advantages,
                                       M, cliprange=0.2, clip_ratio_c=1.5)
    if close(dual_loss, POLICY[0]):
        raise Fail("clip_ratio_c=1.5 changed nothing; row 1 has a negative advantage "
                   "and a ratio above 1.5, so the dual clip must floor its loss")
    need(close(dual_loss, POLICY_DUAL),
         f"wrong dual-clip loss: expected {POLICY_DUAL:.8f}, got {float(dual_loss):.8f}")
    need(close(dual_lower, POLICY_DUAL_LOWER),
         f"wrong pg_clipfrac_lower: expected {POLICY_DUAL_LOWER:.2f}, got "
         f"{float(dual_lower):.2f}; it counts only positions where the dual clip "
         "bound and the advantage is negative")

    asym = call(sol.compute_policy_loss, old, logp, advantages, M, cliprange=0.2,
                cliprange_low=0.2, cliprange_high=0.3)[0]
    need(close(asym, POLICY_ASYM),
         f"wrong asymmetric-clip loss: expected {POLICY_ASYM:.8f}, got {float(asym):.8f}; "
         "cliprange_low and cliprange_high must override cliprange independently")

    positive = torch.ones(2, 4)
    a = call(sol.compute_policy_loss, old, logp, positive, M, cliprange=0.2,
             clip_ratio_c=1.5)[0]
    b = call(sol.compute_policy_loss, old, logp, positive, M, cliprange=0.2,
             clip_ratio_c=3.0)[0]
    need(close(a, b), "with every advantage positive, clip_ratio_c must not matter")


def stage_5():
    x = torch.tensor([[0., 5., -5.]])
    got = call(sol.clip_by_value, x, torch.full((1, 3), -1.0), torch.full((1, 3), 1.0))
    need(close(got, torch.tensor([[0., 1., -1.]])), f"wrong clip: {got.tolist()}")

    vpreds = torch.tensor([[0.2, 0.3, 0.4, 0.], [0.6, 0.5, 0., 0.]])
    got = call(sol.compute_value_loss, vpreds, GAE_RET, V, M, 0.05)
    need(isinstance(got, tuple) and len(got) == 2, "return (vf_loss, vf_clipfrac)")
    vf_loss, vf_clipfrac = got
    if close(vf_loss, VALUE[0] * 2):
        raise Fail("missing the 0.5 factor, which is applied after aggregation")
    need(close(vf_loss, VALUE[0]),
         f"wrong vf_loss: expected {VALUE[0]:.8f}, got {float(vf_loss):.8f}; keep the "
         "LARGER of the clipped and unclipped squared errors, and clip around `values`")
    need(close(vf_clipfrac, VALUE[1]), f"wrong vf_clipfrac: got {float(vf_clipfrac)}")


def stage_6():
    uniform = torch.zeros(1, 1, 4)
    got = call(sol.entropy_from_logits, uniform)
    need(close(got, torch.full((1, 1), 1.3862944)),
         f"a uniform distribution over 4 has entropy ln(4)=1.3863, got {got.tolist()}")
    peaked = torch.tensor([[[50., 0., 0., 0.]]])
    need(float(call(sol.entropy_from_logits, peaked)) < 1e-6,
         "a near-deterministic distribution has entropy about zero")

    logits = torch.tensor([[[2., 0., -1.], [0., 1., 0.], [1., 1., 1.], [0., 0., 0.]],
                           [[3., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
    got = call(sol.compute_entropy_loss, logits, M)
    if close(got, -ENTROPY):
        raise Fail("entropy is returned POSITIVE; the trainer subtracts it")
    need(close(got, ENTROPY), f"expected {ENTROPY:.8f}, got {float(got):.8f}")


def stage_7():
    logprob = torch.tensor([[-1.0, -0.5, -2.0, 0.]])
    ref = torch.tensor([[-1.2, -0.4, -1.0, 0.]])
    for kind, expected in KL.items():
        got = call(sol.kl_penalty, logprob, ref, kind)
        need(close(got, torch.tensor([expected])),
             f"kl_penalty {kind!r}: expected {expected}, got {got.tolist()}")
    need(bool((call(sol.kl_penalty, logprob, ref, "k3") >= 0).all()),
         "k3 must be non-negative on every token; check the direction of the "
         "subtraction, it is the opposite of k1's")
    need(close(call(sol.kl_penalty, logprob, ref, "low_var_kl"), torch.tensor([KL["k3"]])),
         "'low_var_kl' and 'k3' are the same estimator")

    got = call(sol.compute_rewards, R, torch.zeros(2, 4), torch.full((2, 4), -0.1), 0.2)
    expected = torch.tensor([[-0.02, -0.02, 0.98, -0.02], [-0.02, -1.02, -0.02, -0.02]])
    if close(got, R + 0.02):
        raise Fail("the KL was added, not subtracted; check which way round the "
                   "log-ratio goes given it is subtracted from the score")
    need(close(got, expected), f"expected {expected.tolist()}, got {got.tolist()}")


STAGES = [
    ("masked mean, variance and whitening", stage_1),
    ("GAE advantage and returns", stage_2),
    ("loss aggregation modes", stage_3),
    ("dual-clip policy loss", stage_4),
    ("clipped value loss", stage_5),
    ("entropy", stage_6),
    ("KL estimators and reward folding", stage_7),
]

for number, (name, stage) in enumerate(STAGES, 1):
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all PPO stages pass")
