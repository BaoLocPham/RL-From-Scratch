"""Staged PPO grader. Expected constants were printed by ``../common.py``."""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppo as sol

RETURNS = torch.tensor([[3.0699999332, 2.2999999523, 2.0, 9.0]])
GAE = torch.tensor([[2.2817599773, 1.5579999685, 1.8999999762, 9.0]])
TARGETS = torch.tensor([[2.4817600250, 1.9579999447, 2.0, 9.0]])
NORMALIZED = torch.tensor([[-0.1690308452, 0.5070925355, 0.0],
                           [-1.5212775469, 1.1832158566, 0.0]])
POLICY_LOSS = -0.1800000072
VALUE_LOSS = 0.1189999953
ENTROPY = 0.3884986937
TOTAL_LOSS = -0.1243849993


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


REWARDS = torch.tensor([[1.0, 0.5, 2.0, 9.0]])
DONES = torch.tensor([[False, False, True, True]])
VALUES_BOOTSTRAP = torch.tensor([[0.2, 0.4, 0.1, 0.0, 0.0]])
OLD_LOGP = torch.zeros(2, 3)
RATIOS = torch.tensor([[1.3, 1.0, 0.7], [0.8, 1.2, 1.4]])
ADVANTAGE = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
MASK = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
VALUES = torch.tensor([[1.4, 0.5, 0.0], [0.2, 0.9, 0.0]])
OLD_VALUES = torch.tensor([[1.0, 0.4, 0.0], [0.3, 0.7, 0.0]])
VALUE_TARGETS = torch.tensor([[1.1, 1.0, 0.0], [0.8, 0.2, 0.0]])
LOGITS = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [9.0, -9.0]],
                       [[1.0, 1.0], [2.0, -1.0], [0.0, 0.0]]])


def stage_1():
    got = call(sol.discounted_returns, REWARDS, DONES, 0.9)
    need(close(got, RETURNS),
         f"expected {RETURNS.tolist()}, got {got.tolist()}; a terminal step must block later rewards")


def stage_2():
    got = call(sol.generalized_advantage_estimate, REWARDS, VALUES_BOOTSTRAP, DONES, 0.9, 0.8)
    need(isinstance(got, (tuple, list)) and len(got) == 2,
         "return an (advantage, value_target) pair")
    advantage, targets = got
    need(close(advantage, GAE), f"wrong GAE values: expected {GAE.tolist()}, got {advantage.tolist()}")
    need(close(targets, TARGETS), f"wrong value targets: expected {TARGETS.tolist()}, got {targets.tolist()}")


def stage_3():
    raw = torch.tensor([[1.0, 2.0, 99.0], [-1.0, 3.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
    got = call(sol.normalize_advantage, raw, mask)
    need(close(got, NORMALIZED),
         f"expected {NORMALIZED.tolist()}, got {got.tolist()}; padding must not enter mean/std")


def stage_4():
    got = call(sol.clipped_policy_loss, OLD_LOGP, RATIOS.log(), ADVANTAGE, MASK)
    need(torch.as_tensor(got).ndim == 0, "policy loss must be scalar")
    need(close(got, POLICY_LOSS),
         f"expected {POLICY_LOSS:.10f}, got {float(got):.10f}; inspect clipping for negative advantages")


def stage_5():
    got = call(sol.clipped_value_loss, VALUES, OLD_VALUES, VALUE_TARGETS, MASK)
    need(close(got, VALUE_LOSS),
         f"expected {VALUE_LOSS:.10f}, got {float(got):.10f}; use the more conservative error")


def stage_6():
    entropy = call(sol.categorical_entropy, LOGITS, MASK)
    need(close(entropy, ENTROPY),
         f"expected entropy {ENTROPY:.10f}, got {float(entropy):.10f}; ignore padded logits")
    total = call(sol.ppo_loss, OLD_LOGP, RATIOS.log(), ADVANTAGE, VALUES,
                 OLD_VALUES, VALUE_TARGETS, LOGITS, MASK)
    need(close(total, TOTAL_LOSS),
         f"expected combined loss {TOTAL_LOSS:.10f}, got {float(total):.10f}")


STAGES = [
    ("discounted returns", stage_1), ("GAE", stage_2),
    ("advantage normalization", stage_3), ("policy clipping", stage_4),
    ("value clipping", stage_5), ("entropy and total objective", stage_6),
]
for number, (name, stage) in enumerate(STAGES, 1):
    try:
        stage(); print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}"); raise SystemExit(1)
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
print("all PPO stages pass")
