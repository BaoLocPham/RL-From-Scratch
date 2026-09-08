"""Staged DPO grader. Expected constants were printed by ``../common.py``."""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpo as sol

SEQUENCE_LOGP = torch.tensor([-0.7212907076, -2.3670706749])
PREFERENCE_LOGIT = torch.tensor([0.1200000048, -0.0600000136, 0.0999999866])
DPO_LOSS = 0.6676465869
ACCURACY = 0.6666666865


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


LOGITS = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [9.0, -9.0, 0.0]],
                       [[0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [1.0, 1.0, 1.0]]])
TOKENS = torch.tensor([[0, 1, 2], [2, 0, 1]])
MASK = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
POLICY_CHOSEN = torch.tensor([-1.0, -2.0, -0.8])
POLICY_REJECTED = torch.tensor([-2.0, -1.5, -1.4])
REFERENCE_CHOSEN = torch.tensor([-1.4, -1.8, -1.0])
REFERENCE_REJECTED = torch.tensor([-1.8, -1.6, -1.1])


def stage_1():
    got = call(sol.sequence_log_probs, LOGITS, TOKENS, MASK)
    need(tuple(got.shape) == (2,), f"expected one score per completion, got shape {tuple(got.shape)}")
    need(close(got, SEQUENCE_LOGP),
         f"expected {SEQUENCE_LOGP.tolist()}, got {got.tolist()}; sum valid tokens only")


def stage_2():
    got = call(sol.preference_logit, POLICY_CHOSEN, POLICY_REJECTED,
               REFERENCE_CHOSEN, REFERENCE_REJECTED, 0.2)
    need(close(got, PREFERENCE_LOGIT),
         f"expected {PREFERENCE_LOGIT.tolist()}, got {got.tolist()}; retain the reference correction")


def stage_3():
    got = call(sol.dpo_loss, POLICY_CHOSEN, POLICY_REJECTED,
               REFERENCE_CHOSEN, REFERENCE_REJECTED, 0.2)
    need(torch.as_tensor(got).ndim == 0, "DPO loss must be scalar")
    need(close(got, DPO_LOSS), f"expected {DPO_LOSS:.10f}, got {float(got):.10f}")


def stage_4():
    got = call(sol.preference_accuracy, POLICY_CHOSEN, POLICY_REJECTED,
               REFERENCE_CHOSEN, REFERENCE_REJECTED)
    need(close(got, ACCURACY), f"expected accuracy {ACCURACY:.10f}, got {float(got):.10f}")


STAGES = [
    ("sequence log-probabilities", stage_1), ("preference logit", stage_2),
    ("DPO loss", stage_3), ("preference accuracy", stage_4),
]
for number, (name, stage) in enumerate(STAGES, 1):
    try:
        stage(); print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}"); raise SystemExit(1)
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
print("all DPO stages pass")
