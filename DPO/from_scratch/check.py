"""Staged grader for ``dpo.py``. Hand-worked cases, plus the reference on seeded random inputs."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dpo as sol  # noqa: E402


def _reference():
    """DPO/common.py, loaded under its own name."""
    spec = importlib.util.spec_from_file_location("dpo_reference", HERE.parent / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()
STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open DPO/from_scratch/dpo.py and fill the lines marked "
                f"'TODO stage {STAGE}'")


def has_blank(fn):
    """A `...` left on a 'TODO stage N' line. A bare `...` statement never raises, so look."""
    for line in inspect.getsource(fn).splitlines():
        if f"TODO stage {STAGE}" in line and line.split("#")[0].strip().endswith("..."):
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
        if "-100" in str(exc) and "out of bounds" in str(exc):
            raise Fail("gather hit a -100 label. -100 marks an unscored position, not a token: replace the -100s "
                       "with 0 before gather (labels.clamp(min=0)), and let the mask zero those positions") from exc
        raise Fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}") from exc
    return value


def close(actual, expected, atol=1e-4):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    actual = torch.as_tensor(actual, dtype=torch.float32)
    expected = torch.as_tensor(expected, dtype=torch.float32)
    return actual.shape == expected.shape and torch.allclose(actual, expected, atol=atol)


def stage_1():
    rewards = torch.zeros(4, 3)
    rewards[[0, 1, 2, 3], [2, 1, 2, 2]] = torch.tensor([1.0, 0.0, 0.5, 0.5])
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.], [1., 1., 1.], [1., 1., 1.]])
    got = call(sol.compute_onlinedpo_pref, rewards, mask)
    need(isinstance(got, torch.Tensor) and got.dtype == torch.bool,
         "return a bool tensor: True on each pair's chosen response")
    need(tuple(got.shape) == (4,), f"one entry per RESPONSE, shape (bs,) = (4,); got {tuple(got.shape)}")
    if got.tolist() == [True, False, False, True]:
        raise Fail("on a tie (the second pair: 0.5 and 0.5) the SECOND response was chosen. torch.argmax returns "
                   "the first maximum, so the recipe chooses the first response")
    need(got.tolist() == [True, False, True, False],
         f"scores [1.0, 0.0, 0.5, 0.5] should give [True, False, True, False]; got {got.tolist()}")

    outside = torch.tensor([[0., 0., 0.], [0., 0., 5.]])      # response b's reward sits on a padded position
    got = call(sol.compute_onlinedpo_pref, outside, torch.tensor([[1., 1., 1.], [1., 1., 0.]]))
    need(got.tolist() == [True, False], "a reward on a masked-out position must not count: multiply by "
         "response_mask before summing")

    torch.manual_seed(0)
    rewards = torch.randn(10, 5)
    mask = (torch.rand(10, 5) > 0.3).float()
    need(torch.equal(call(sol.compute_onlinedpo_pref, rewards, mask), REF.compute_onlinedpo_pref(rewards, mask)),
         "wrong on 5 random pairs: pairs are rows (0, 1), (2, 3), ...: view the scores as (-1, 2)")
    try:
        sol.compute_onlinedpo_pref(torch.zeros(3, 2), torch.ones(3, 2))
        raise Fail("an odd batch cannot be split into pairs: keep the ValueError at the top")
    except ValueError:
        pass


def stage_2():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 5, 5, generator=generator, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 1, 2], [-100, -100, 4, 3, 0],
                           [-100, -100, 2, 2, 1], [-100, -100, 1, 2, 2]])
    expected = REF.get_batch_logps(logits, labels)
    got = call(sol.get_batch_logps, logits, labels)
    need(isinstance(got, torch.Tensor), "return a tensor")
    need(tuple(got.shape) == (4,), f"one log-prob per sequence, shape (bs,) = (4,); got {tuple(got.shape)}")
    if not close(got, expected):
        logp = F.log_softmax(logits.detach(), -1)
        scored = labels != -100
        unshifted = (logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * scored).sum(-1)
        if close(got, unshifted):
            raise Fail("no shift: the logits at position t predict the token at t + 1, so use logits[:, :-1] with "
                       "labels[:, 1:]")
        if close(got, REF.get_batch_logps(logits, labels, average_log_prob=True)):
            raise Fail("that is the mean per token. With average_log_prob=False, SUM the scored positions")
        raise Fail(f"expected {[round(x, 4) for x in expected.tolist()]}, got {[round(x, 4) for x in got.tolist()]}\n"
                   "  per position t < seq_len - 1: log_softmax(logits[:, t])[labels[:, t + 1]], summed where "
                   "labels[:, t + 1] != -100")
    need(got.requires_grad, "the log-probs must carry gradient to the logits")
    mean = call(sol.get_batch_logps, logits, labels, average_log_prob=True)
    need(close(mean, REF.get_batch_logps(logits, labels, average_log_prob=True)),
         "average_log_prob=True: divide each sum by its number of scored tokens (clamped to at least 1)")
    empty = call(sol.get_batch_logps, logits[:1], torch.full((1, 5), -100), average_log_prob=True)
    need(torch.isfinite(empty).all() and close(empty, [0.0]),
         "a sequence with no scored token must give 0, not nan: clamp the count to at least 1")


def stage_3():
    g = torch.Generator().manual_seed(1)
    pc, pr = (-5 * torch.rand(6, generator=g)).requires_grad_(), (-5 * torch.rand(6, generator=g)).requires_grad_()
    rc, rr = -5 * torch.rand(6, generator=g), -5 * torch.rand(6, generator=g)

    got = call(sol.compute_online_dpo_loss, pc, pr, rc, rr, 0.1)
    need(isinstance(got, torch.Tensor) and got.numel() == 1, "return one number, the mean over the pairs")
    expected = REF.compute_online_dpo_loss(pc, pr, rc, rr, 0.1)
    if not close(got, expected):
        logits = (pc - pr) - (rc - rr)
        wrong = {
            "the logits are upside down: (log pi(chosen) - log pi(rejected)) minus the same under pi_ref":
                -F.logsigmoid(-0.1 * logits).mean(),
            "beta is missing: -logsigmoid(beta * logits)": -F.logsigmoid(logits).mean(),
            "pi_ref is missing: subtract ref_logratios (unless reference_free)": -F.logsigmoid(0.1 * (pc - pr)).mean(),
        }
        for message, value in wrong.items():
            if close(got, value):
                raise Fail(message)
        raise Fail(f"sigmoid, beta 0.1: expected {float(expected):.5f}, got {float(got):.5f}")
    need(got.requires_grad, "the loss must carry gradient")

    for kwargs, what in (({"label_smoothing": 0.2}, "label_smoothing 0.2: (1 - eps) * -logsigmoid(beta * logits) "
                          "+ eps * -logsigmoid(-beta * logits)"),
                         ({"loss_type": "ipo"}, "ipo: mean((logits - 1 / (2 * beta)) ** 2)"),
                         ({"reference_free": True}, "reference_free: ref_logratios is zeros_like(pi_logratios)"),
                         ({"loss_type": "ipo", "reference_free": True}, "ipo, reference_free")):
        got = call(sol.compute_online_dpo_loss, pc, pr, rc, rr, 0.1, **kwargs)
        need(close(got, REF.compute_online_dpo_loss(pc, pr, rc, rr, 0.1, **kwargs)), f"wrong for {what}")

    far = call(sol.compute_online_dpo_loss, torch.tensor([-1000.0]), torch.tensor([-1.0]), torch.tensor([-1.0]),
               torch.tensor([-1.0]), 1.0)
    need(torch.isfinite(far).all(), "a pair the policy gets badly wrong gave a non-finite loss: use F.logsigmoid, "
         "not torch.log(torch.sigmoid(...))")
    try:
        sol.compute_online_dpo_loss(pc, pr, rc, rr, 0.1, loss_type="hinge")
        raise Fail("an unsupported loss_type must raise ValueError (keep the else branch)")
    except ValueError:
        pass


STAGES = [
    ("compute_onlinedpo_pref: pairs from rewards", stage_1),
    ("get_batch_logps: log pi of a whole response", stage_2),
    ("compute_online_dpo_loss: DPO, label smoothing, IPO", stage_3),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all DPO stages pass")
