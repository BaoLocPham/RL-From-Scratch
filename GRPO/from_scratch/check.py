"""Staged grader for ``grpo.py``. Expected constants are reference outputs."""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grpo as sol

ADV = torch.tensor([-0.9999900460, 0.0, 0.9999900460,
                    -0.9999998212, 0.0, 0.9999998212])
SINGLETON = torch.tensor([6.9999933243])
LOSS = -0.3222221434
K3 = torch.tensor([[0.3678793907, 0.0, 0.1487212181, 4.3890562057]])
NAIVE_LOG_RATIO = torch.tensor([[1.0, 0.0, -0.5, -2.0]])
LOSS_WITH_KL = -0.3221295178


class Fail(Exception):
    pass


def need(condition, message):
    if not condition:
        raise Fail(message)


def call(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except Exception as exc:
        raise Fail(f"raised {type(exc).__name__}: {exc}") from exc
    need(value is not None, "returned None; finish this stage's TODO")
    return value


def close(actual, expected, atol=1e-5):
    return torch.allclose(torch.as_tensor(actual), torch.as_tensor(expected), atol=atol)


def stage_1():
    rewards = torch.tensor([0.9, 1.0, 1.1, -5.0, 0.0, 5.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    got = call(sol.group_relative_advantage, rewards, groups)
    need(tuple(got.shape) == (6,), f"expected shape (6,), got {tuple(got.shape)}")
    if not close(got, ADV):
        global_answer = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        if close(got, global_answer):
            raise Fail("normalized the whole batch; GRPO's baseline is prompt-local")
        raise Fail(f"wrong per-group advantages: got {got.tolist()}")
    equal = call(sol.group_relative_advantage, torch.tensor([2.0, 2.0, 2.0]),
                 torch.tensor([4, 4, 4]))
    need(torch.isfinite(equal).all() and close(equal, torch.zeros(3)),
         "an identical-reward group must be finite and near zero; keep the epsilon")
    one = call(sol.group_relative_advantage, torch.tensor([7.0]), torch.tensor([9]))
    need(close(one, SINGLETON),
         "singleton handling differs from Agent0: its special-case mean is zero and std is one")


def stage_2():
    old = torch.zeros(3, 4)
    ratio = torch.tensor([[1.3, 1.0, 0.7, 1.0],
                          [0.8, 1.2, 1.4, 1.0],
                          [1.1, 0.9, 1.0, 1.0]])
    advantage = torch.tensor([1.0, -1.0, 0.5])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    got = call(sol.clipped_surrogate_loss, old, ratio.log(), advantage, mask)
    need(torch.as_tensor(got).ndim == 0, "loss must be one scalar")
    need(close(got, LOSS),
         f"wrong clipped loss: expected {LOSS:.10f}, got {float(got):.10f}; check negative advantages and masked denominator")
    ragged = call(sol.clipped_surrogate_loss, torch.zeros(2, 3),
                  torch.tensor([[1.1, 0.9, 1.0], [0.8, 1.4, 1.0]]).log(),
                  torch.tensor([1.0, -1.0]),
                  torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32))
    need(torch.as_tensor(ragged).ndim == 0,
         "advantage must broadcast down each completion row, not across tokens")


def stage_3():
    ref_logp = torch.tensor([[-1.0, 0.0, 0.5, 2.0]])
    logp = torch.zeros_like(ref_logp)
    got = call(sol.kl_penalty_k3, ref_logp, logp)
    need(tuple(got.shape) == (1, 4), f"expected per-token shape (1, 4), got {tuple(got.shape)}")
    if close(got, NAIVE_LOG_RATIO):
        raise Fail("returned the naive log-ratio estimator; it can be negative on one sample")
    need(close(got, K3), f"wrong k3 values: expected {K3.tolist()}, got {got.tolist()}")
    need(bool((got >= 0).all()), "k3 must stay non-negative on every sampled token")


def stage_4():
    old = torch.zeros(3, 4)
    ratio = torch.tensor([[1.3, 1.0, 0.7, 1.0],
                          [0.8, 1.2, 1.4, 1.0],
                          [1.1, 0.9, 1.0, 1.0]])
    logp = ratio.log()
    advantage = torch.tensor([1.0, -1.0, 0.5])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    ref_logp = logp + torch.tensor([[0.05, -0.05, 0.10, 0.00],
                                    [-0.10, 0.05, 0.00, 0.00],
                                    [0.02, -0.02, 0.05, -0.05]])
    got = call(sol.clipped_surrogate_loss, old, logp, advantage, mask,
               ref_logp=ref_logp, beta=0.05)
    need(close(got, LOSS_WITH_KL),
         f"wrong clipped-plus-KL loss: expected {LOSS_WITH_KL:.10f}, got {float(got):.10f}")
    no_weight = call(sol.clipped_surrogate_loss, old, logp, advantage, mask,
                     ref_logp=ref_logp, beta=0.0)
    need(close(no_weight, LOSS), "beta=0 must preserve the original clipped-only loss")
    no_reference = call(sol.clipped_surrogate_loss, old, logp, advantage, mask, beta=0.05)
    need(close(no_reference, LOSS), "without a reference, preserve the original clipped-only loss")


STAGES = [
    ("group-relative advantage", stage_1),
    ("clipped surrogate", stage_2),
    ("non-negative k3 KL", stage_3),
    ("clipped surrogate plus KL", stage_4),
]

for number, (name, stage) in enumerate(STAGES, 1):
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
print("all GRPO stages pass")
