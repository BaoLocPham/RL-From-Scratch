"""Staged grader for ``ppo.py``. Expected constants are reference outputs."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ppo as sol
import task  # noqa: E402

R = torch.tensor([[0., 0., 1., 0.], [0., -1., 0., 0.]])
V = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
M = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])

LOGPROBS = [-1.09861231, -0.16984603]
GAE_RAW = torch.tensor([[0.65156752, 0.66850001, 0.69999999, 0.0], [-1.33700001, -1.39999998, 0.0, 0.0]])
CLIP_BATCH = 0.29502711                             # core ppo_clip_loss on the stage 9 batch (eps 0.2)
CLIP_BATCH_MIN = -0.19197340                        # the same with min instead of max
VLOSS = 0.53023881                                  # core value_loss on the stage 10 batch
VLOSS_NO_HALF, VLOSS_UNMASKED = 1.06047761, 0.34952426
GAE_ADV = torch.tensor([[0.71058786, 0.72572333, 0.75388032, 0.12816931],
                        [-1.06693876, -1.12325275, 0.12816931, 0.12816931]])
GAE_RET = torch.tensor([[0.75156754, 0.86849999, 1.0, 0.40000001],
                        [-0.83700001, -1.0, 0.30000001, 0.20000000]])
# A hole in the MIDDLE of a response (a tool's output at position 1), gamma = lam = 1.
HOLE_R, HOLE_V, HOLE_M = (torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[0.1, 0.5, 0.2, 0.3]]),
                          torch.tensor([[1., 0., 1., 1.]]))
HOLE_RET = [1.0, 1.29999995, 1.0, 1.0]
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


def _reference():
    """PPO/common.py, loaded under its own name. Stage 8 compares whole batches against it."""
    spec = importlib.util.spec_from_file_location("ppo_reference", HERE.parent / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = 0                                           # the stage being graded, for messages


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open PPO/from_scratch/ppo.py, fill the lines marked "
                f"'TODO stage {STAGE}', and try them with `python PPO/from_scratch/ppo.py`")


def has_blank(fn):
    """A `...` left on a 'TODO stage N' line. Bare `...` never raises, so look."""
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
        raise Fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}") from exc
    if value is Ellipsis:
        raise unfinished()
    need(value is not None, f"{fn.__name__} returned None")
    return value


def num(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def close(actual, expected, atol=1e-5):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def stage_1():
    logits = torch.tensor([[[0., 0., 0.], [2., 0., -1.]]])
    got = call(sol.logprobs_from_logits, logits, torch.tensor([[1, 0]]))
    need(tuple(got.shape) == (1, 2), f"one log-prob per position: expected shape (1, 2), got {tuple(got.shape)}. "
         "gather the sampled token's entry, then squeeze the vocabulary axis away")
    if close(got, [0.33333331, 0.84379470]):
        raise Fail("those are probabilities; use log_softmax, not softmax")
    if close(got, [0.0, 2.0]):
        raise Fail("those are the raw logits of the sampled tokens; normalise with log_softmax first")
    need(close(got, LOGPROBS), f"expected {LOGPROBS}, got {got.tolist()}")

    values = torch.tensor([[1., 2., 100.]])
    mask = torch.tensor([[1., 1., 0.]])
    got = call(sol.masked_mean, values, mask)
    if close(got, 34.333332):
        raise Fail("averaged over every position; the masked-out 100. must not count")
    need(close(got, 1.5), f"masked_mean: expected 1.5, got {num(got)}")
    need(torch.isfinite(torch.as_tensor(call(sol.masked_mean, values, torch.zeros(1, 3)))).all(),
         "an all-zero mask must stay finite; verl adds 1e-8 to the count rather than clamping")
    need(close(call(sol.masked_mean, torch.tensor([[1., 2.], [3., 4.]]), torch.ones(2, 2), axis=-1), [1.5, 3.5]),
         "masked_mean(..., axis=-1) should average each row separately: pass axis to both sums")


def stage_2():
    got = call(sol.compute_gae, R, V, M, 0.9, 0.95)
    need(isinstance(got, tuple) and len(got) == 2, "return the pair (advantages, returns)")
    advantages, returns = got
    need(close(advantages, GAE_RAW),
         f"wrong advantages.\n  expected {GAE_RAW.tolist()}\n  got      {advantages.tolist()}\n"
         "  row 0 by hand: A_2 = 0.70, A_1 = 0.07 + 0.855 * 0.70 = 0.6685, A_0 = 0.08 + 0.855 * 0.6685 = 0.6516")
    need(close(returns, GAE_RET),
         f"advantages are right but returns are not: returns = advantages + values.\n"
         f"  expected {GAE_RET.tolist()}\n  got      {returns.tolist()}")

    _, hole_returns = call(sol.compute_gae, HOLE_R, HOLE_V, HOLE_M, 1.0, 1.0)
    if close(hole_returns[0, 2:], [1.0, 1.0]) and not close(hole_returns[0, 0], 1.0):
        raise Fail("a masked position in the MIDDLE reset the carry to zero, so the first token lost part of "
                   "its credit for the final reward (return below 1.0). response_mask is not a "
                   "`dones` flag: where it is 0, keep nextvalues and lastgaelam as they were")
    need(close(hole_returns, [HOLE_RET]),
         f"a response with a tool-output token in the middle (mask [1, 0, 1, 1]): expected returns "
         f"{HOLE_RET}, got {hole_returns.tolist()}")


def stage_3():
    one = torch.ones(1, 1)
    got = call(sol.ppo_clip_loss, torch.zeros(1, 1), torch.tensor([[1.35]]).log(), one, one)
    need(torch.as_tensor(got).numel() == 1, "return one number: the masked mean over the tokens")
    if close(got, -1.35):
        raise Fail("one token with A = +1 at ratio 1.35 should be clipped to -1.2: take the MAX of the two "
                   "(the minus sign turned Surrogates' min into a max)")
    if close(got, 1.2):
        raise Fail("the sign is flipped; the loss is -A * ratio, so descent raises the objective")
    need(close(got, -1.2), f"one token, A +1, ratio 1.35: expected -1.2, got {num(got):.4f}")
    need(close(call(sol.ppo_clip_loss, torch.zeros(1, 1), torch.tensor([[0.7]]).log(), one, one), -0.7),
         "A +1 at ratio 0.7 moved the WRONG way: it should pay in full, -0.7, not the clipped -0.8")

    old = torch.zeros(2, 4)
    logp = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
    advantages = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
    got = call(sol.ppo_clip_loss, old, logp, advantages, M)
    if close(got, CLIP_BATCH_MIN):
        raise Fail("that is the min; once the minus sign is inside, the pessimistic choice is the MAX")
    need(close(got, CLIP_BATCH), f"a batch of two responses: expected {CLIP_BATCH:.6f}, got {num(got):.6f}. "
         "Average over the mask with your masked_mean")
    loss = call(sol.ppo_clip_loss, old, logp.clone().requires_grad_(), advantages, M)
    need(loss.requires_grad, "the loss must carry gradient back to log_prob: no .item() or torch.no_grad()")


def stage_4():
    one = torch.ones(1, 1)
    got = call(sol.value_loss, torch.tensor([[0.2]]), torch.tensor([[0.75]]), one)
    if close(got, 0.3025):
        raise Fail("missing the 0.5")
    need(close(got, 0.15125), f"one token, prediction 0.2, return 0.75: expected 0.15125, got {num(got):.5f}")
    vpreds = torch.tensor([[0.2, 0.3, 0.4, 0.], [0.6, 0.5, 0., 0.]])
    got = call(sol.value_loss, vpreds, GAE_RET, M)
    if close(got, VLOSS_UNMASKED):
        raise Fail("the padded positions were counted; average over the mask")
    need(close(got, VLOSS), f"a batch of two responses: expected {VLOSS:.6f}, got {num(got):.6f}")


def stage_5():
    got = call(sol.entropy_from_logits, torch.zeros(1, 1, 4))
    need(close(got, torch.full((1, 1), 1.3862944)),
         f"a uniform distribution over 4 has entropy ln 4 = 1.3863, got {got.tolist()}")
    need(num(call(sol.entropy_from_logits, torch.tensor([[[50., 0., 0., 0.]]]))) < 1e-6,
         "a near-deterministic distribution has entropy about zero")
    logits = torch.tensor([[[2., 0., -1.], [0., 1., 0.], [1., 1., 1.], [0., 0., 0.]],
                           [[3., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
    got = call(sol.entropy_bonus, logits, M)
    if close(got, -ENTROPY):
        raise Fail("entropy is returned POSITIVE; the loss subtracts it (stage 6)")
    need(close(got, ENTROPY), f"entropy_bonus: expected {ENTROPY:.8f}, got {num(got):.8f}")


def moved_model():
    """A model some way from the start (and from the reference), so nothing is trivially zero."""
    model = task.TokenModel()
    with torch.no_grad():
        model.policy_logits.copy_(torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.0, -1.0],
                                                [0.0, 0.8, 0.2], [-0.3, 0.3, 0.9]]))
        model.value_head.copy_(torch.tensor([0.2, 0.3, 0.4, 0.5]))
    return model


def check_loop(ref, advantage_name, update_name, verl):
    """Shared by stages 6 and 12: one update against the reference, then a whole training run."""
    torch.manual_seed(0)
    batch = task.rollout(moved_model(), 32, ref.logprobs_from_logits)
    mine = call(getattr(sol, advantage_name), {k: v.clone() for k, v in batch.items()})
    theirs = getattr(ref, advantage_name)({k: v.clone() for k, v in batch.items()})
    for key in ("advantages", "returns"):
        need(key in mine, f"{advantage_name} must add batch[{key!r}]")
        need(close(mine[key], theirs[key]), f"{advantage_name}: batch[{key!r}] differs from the reference")

    def one_update(update):
        torch.manual_seed(1)                                        # the same shuffles for both
        model = moved_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        prepared = getattr(ref, advantage_name)({k: v.clone() for k, v in batch.items()})
        return model, update(model, optimizer, prepared)

    model, metrics = one_update(lambda *a: call(getattr(sol, update_name), *a))
    expected, _ = one_update(getattr(ref, update_name))
    need(isinstance(metrics, dict) and "pg_loss" in metrics, f"{update_name} must return the metrics dict")
    need(not close(model.value_head, torch.tensor([0.2, 0.3, 0.4, 0.5])),
         "the critic did not move. Is vf_coef * vf_loss in the loss, and is the value loss's TARGET "
         "mb['returns']?")
    need(not close(model.policy_logits, moved_model().policy_logits),
         "the policy did not move: is pg_loss in the loss, built from your log_prob (which carries gradient)?")
    if not close(model.policy_logits, expected.policy_logits, atol=1e-4):
        raise Fail("after one update the policy differs from the reference. Check eq. 9's signs, "
                   "loss = pg_loss + vf_coef * vf_loss - entropy_coeff * entropy (the entropy bonus is "
                   "SUBTRACTED), and that the policy loss gets mb['old_log_prob'] -- the FROZEN one -- "
                   "your log_prob and mb['advantages'], in that order")
    need(close(model.value_head, expected.value_head, atol=1e-4),
         "the policy matches but the critic does not: the value loss compares model.values(m) "
         "with mb['returns']")

    history, model = task.train(sol, verl=verl)
    reference, _ = task.train(ref, verl=verl)
    print("  your PPO on the token task (target 2, 0, 1, 2), 40 iterations:")
    print("  " + task.header(verl))
    for i in (0, 5, 10, 20, 39):
        print("  " + task.row(i, history[i]))
    need(abs(history[-1]["reward"] - reference[-1]["reward"]) < 1e-4,
         f"final reward {history[-1]['reward']:.3f}, the reference reaches {reference[-1]['reward']:.3f}")
    print(f"  learned tokens: {model.policy_logits.argmax(-1).tolist()}   (target {task.TARGET.tolist()})")


def stage_6():
    check_loop(_reference(), "compute_advantage", "ppo_update", verl=False)
    print("\n  PART 1 DONE: that is core PPO, the paper's Algorithm 1 with eq. 9. Part 2 (stages 7-12)")
    print("  is verl's production extras -- do it before GRPO, which imports agg_loss, compute_policy_loss")
    print("  and kl_penalty from it.")


def stage_7():
    spread = torch.tensor([[1., 2., 3., 100.]])
    mask4 = torch.tensor([[1., 1., 1., 0.]])
    got = call(sol.masked_var, spread, mask4)
    if close(got, 0.6666667):
        raise Fail("that is the population variance; unbiased=True applies n/(n-1) "
                   "over the number of MASKED positions")
    if close(got, 0.8888889):
        raise Fail("n/(n-1) must count the REAL positions (mask.sum()), not the padded width")
    need(close(got, 1.0), f"masked_var: expected 1.0, got {num(got)}")
    need(close(call(sol.masked_var, spread, mask4, unbiased=False), 0.6666667),
         "unbiased=False must skip the Bessel correction")

    got = call(sol.masked_whiten, spread, mask4)
    need(close(got[0, :3], torch.tensor([-1.0, 0.0, 1.0])),
         f"masked positions should whiten to [-1, 0, 1], got {got[0, :3].tolist()}")
    shifted = call(sol.masked_whiten, spread, mask4, shift_mean=False)
    need(close(shifted[0, :3], torch.tensor([1.0, 2.0, 3.0])),
         "shift_mean=False re-adds the original mean after scaling")


    got = call(sol.compute_gae_advantage_return, R, V, M, 0.9, 0.95)
    need(isinstance(got, tuple) and len(got) == 2, "return the pair (advantages, returns)")
    advantages, returns = got
    if close(returns, GAE_ADV + V, atol=1e-3):
        raise Fail("returns were built from the WHITENED advantage; take them from compute_gae as they are")
    need(close(returns, GAE_RET), "returns should be compute_gae's, untouched")
    need(close(advantages, GAE_ADV),
         f"advantages should be compute_gae's, whitened over the mask.\n"
         f"  expected {GAE_ADV.tolist()}\n  got      {advantages.tolist()}")


def stage_8():
    loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
    for mode, expected in AGG.items():
        got = call(sol.agg_loss, loss, M, mode)
        if mode == "seq-mean-token-sum-norm" and close(got, 7.5):
            raise Fail("seq-mean-token-sum-norm divided by the number of responses; "
                       "Dr.GRPO's divisor is the padded width, loss_mask.shape[-1]")
        if close(got, loss.mean()) or close(got, 214.0 / 8):
            raise Fail(f"{mode}: the padding (the 100s) was counted; multiply by the mask first")
        need(close(got, expected), f"{mode}: expected {expected}, got {num(got)}")
    try:
        sol.agg_loss(loss, M, "nonsense")
    except ValueError:
        pass
    except Exception as exc:
        raise Fail(f"an unknown mode should raise ValueError, raised {type(exc).__name__}") from exc
    else:
        raise Fail("an unknown loss_agg_mode must raise ValueError")


    logits = torch.tensor([[[2., 0., -1.], [0., 1., 0.], [1., 1., 1.], [0., 0., 0.]],
                           [[3., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
    need(close(call(sol.compute_entropy_loss, logits, M), ENTROPY),
         f"compute_entropy_loss with token-mean should equal entropy_bonus, {ENTROPY:.6f}")


def stage_9():
    one = torch.ones(1, 1)
    got = call(sol.compute_policy_loss, torch.zeros(1, 1), torch.tensor([[1.35]]).log(), one, one, cliprange=0.2)
    need(isinstance(got, tuple) and len(got) == 4, "return (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)")
    if close(got[0], -1.35):
        raise Fail("one token with A = +1 at ratio 1.35 should be clipped to -1.2: take the MAX of "
                   "pg_losses1 and pg_losses2 (the minus sign turned Surrogates' min into a max)")
    if close(got[0], 1.2):
        raise Fail("the sign is flipped; the loss is -A * ratio, so descent raises the objective")
    need(close(got[0], -1.2), f"one token, A +1, ratio 1.35: expected -1.2, got {num(got[0]):.4f}")

    old = torch.zeros(2, 4)
    # Row 1 carries a negative advantage and a ratio above clip_ratio_c, which is
    # the only situation where the dual clip does anything at all.
    logp = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
    advantages = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
    pg_loss, pg_clipfrac, ppo_kl, _ = call(sol.compute_policy_loss, old, logp, advantages, M, cliprange=0.2)
    need(close(pg_loss, POLICY[0]), f"wrong pg_loss: expected {POLICY[0]:.8f}, got {num(pg_loss):.8f}")
    need(close(pg_clipfrac, POLICY[1]), f"wrong pg_clipfrac: got {num(pg_clipfrac)}")
    if close(ppo_kl, -POLICY[2]):
        raise Fail("ppo_kl has the wrong sign; it is the masked mean of MINUS the log-ratio")
    need(close(ppo_kl, POLICY[2]), f"wrong ppo_kl: got {num(ppo_kl)}")

    dual_loss, _, _, dual_lower = call(sol.compute_policy_loss, old, logp, advantages, M,
                                       cliprange=0.2, clip_ratio_c=1.5)
    if close(dual_loss, POLICY[0]):
        raise Fail("clip_ratio_c=1.5 changed nothing; row 1 has a negative advantage and a ratio "
                   "above 1.5, so the dual clip must floor its loss")
    need(close(dual_loss, POLICY_DUAL),
         f"wrong dual-clip loss: expected {POLICY_DUAL:.8f}, got {num(dual_loss):.8f}")
    need(close(dual_lower, POLICY_DUAL_LOWER),
         f"wrong pg_clipfrac_lower: expected {POLICY_DUAL_LOWER:.2f}, got {num(dual_lower):.2f}")

    asym = call(sol.compute_policy_loss, old, logp, advantages, M, cliprange=0.2,
                cliprange_low=0.2, cliprange_high=0.3)[0]
    need(close(asym, POLICY_ASYM),
         f"wrong asymmetric-clip loss: expected {POLICY_ASYM:.8f}, got {num(asym):.8f}; "
         "clamp to [1 - cliprange_low, 1 + cliprange_high]")
    positive = torch.ones(2, 4)
    a = call(sol.compute_policy_loss, old, logp, positive, M, cliprange=0.2, clip_ratio_c=1.5)[0]
    b = call(sol.compute_policy_loss, old, logp, positive, M, cliprange=0.2, clip_ratio_c=3.0)[0]
    need(close(a, b), "with every advantage positive, clip_ratio_c must not matter: the dual clip "
         "applies only where A < 0")


def stage_10():
    x = torch.tensor([[0., 5., -5.]])
    got = call(sol.clip_by_value, x, torch.full((1, 3), -1.0), torch.full((1, 3), 1.0))
    need(close(got, torch.tensor([[0., 1., -1.]])), f"clip_by_value: expected [[0, 1, -1]], got {got.tolist()}")

    one = torch.ones(1, 1)
    got = call(sol.compute_value_loss, torch.tensor([[0.2]]), torch.tensor([[0.75]]), torch.tensor([[0.1]]), one, 0.05)
    need(isinstance(got, tuple) and len(got) == 2, "return (vf_loss, vf_clipfrac)")
    if close(got[0], 0.15125):
        raise Fail("kept the SMALLER squared error; keep the larger, the pessimistic one")
    if close(got[0], 0.36):
        raise Fail("missing the 0.5, applied after aggregation")
    need(close(got[0], 0.18), f"one token (old 0.1, target 0.75, new 0.2, cliprange 0.05): expected 0.18, "
         f"got {num(got[0]):.4f}")

    vpreds = torch.tensor([[0.2, 0.3, 0.4, 0.], [0.6, 0.5, 0., 0.]])
    vf_loss, vf_clipfrac = call(sol.compute_value_loss, vpreds, GAE_RET, V, M, 0.05)
    need(close(vf_loss, VALUE[0]), f"wrong vf_loss: expected {VALUE[0]:.8f}, got {num(vf_loss):.8f}; "
         "clip around `values` (the old prediction), not around `returns`")
    need(close(vf_clipfrac, VALUE[1]), f"wrong vf_clipfrac: got {num(vf_clipfrac)}")


def stage_11():
    logprob = torch.tensor([[-1.0, -0.5, -2.0, 0.]])
    ref = torch.tensor([[-1.2, -0.4, -1.0, 0.]])
    for kind, expected in KL.items():
        got = call(sol.kl_penalty, logprob, ref, kind)
        need(close(got, torch.tensor([expected])), f"kl_penalty {kind!r}: expected {expected}, got {got.tolist()}")
    need(bool((call(sol.kl_penalty, logprob, ref, "k3") >= 0).all()),
         "k3 must be non-negative on every token")
    need(close(call(sol.kl_penalty, logprob, ref, "low_var_kl"), torch.tensor([KL["k3"]])),
         "'low_var_kl' and 'k3' are the same estimator")

    got = call(sol.compute_rewards, R, torch.zeros(2, 4), torch.full((2, 4), -0.1), 0.2)
    expected = torch.tensor([[-0.02, -0.02, 0.98, -0.02], [-0.02, -1.02, -0.02, -0.02]])
    if close(got, R + 0.02):
        raise Fail("the KL was added, not subtracted: the log-ratio is old_log_prob - ref_log_prob")
    need(close(got, expected), f"compute_rewards: expected {expected.tolist()}, got {got.tolist()}")


def stage_12():
    check_loop(_reference(), "verl_compute_advantage", "verl_ppo_update", verl=True)
    print("\n  PART 2 DONE: verl's PPO. The same loop; every extra sits inside a function you wrote.")


STAGES = [
    ("(core) one decision -> T tokens: log-probs and the masked mean", stage_1),
    ("(core) GAE, the advantage (eq. 11-12)", stage_2),
    ("(core) L^CLIP per token (eq. 7)", stage_3),
    ("(core) L^VF, the value loss", stage_4),
    ("(core) S, the entropy bonus", stage_5),
    ("(core) the loop, Algorithm 1 with eq. 9", stage_6),
    ("(verl) whitening the advantages", stage_7),
    ("(verl) four ways to average over tokens", stage_8),
    ("(verl) asymmetric and dual clip, with metrics", stage_9),
    ("(verl) the clipped value loss", stage_10),
    ("(verl) KL to a reference model, in the reward", stage_11),
    ("(verl) the same loop with verl's functions", stage_12),
]

CORE_ONLY = "core" in sys.argv[1:]                 # `check.py core`: Part 1 only
if CORE_ONLY:
    STAGES = STAGES[:6]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "from_scratch/ppo.py`" not in str(exc):
            print("\n  Tip: `python PPO/from_scratch/ppo.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all core PPO stages pass (Part 2, verl's extras: run without `core`)" if CORE_ONLY else "all PPO stages pass")
