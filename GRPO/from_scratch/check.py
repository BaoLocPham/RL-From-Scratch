"""Staged grader for ``grpo.py``. Expected constants are reference outputs."""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import grpo as sol
except Exception as exc:  # the PPO exercise is imported by grpo.py
    print("could not import grpo.py -- it imports your PPO/from_scratch/ppo.py, "
          f"so finish that one first.\n  {type(exc).__name__}: {exc}")
    raise SystemExit(1)

R = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.],
                  [0., 0., 0.5, 0.], [0., -1., 0., 0.]])
M = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.],
                  [1., 1., 1., 1.], [1., 1., 0., 0.]])
INDEX = np.array(["a", "a", "a", "b"])

ADV = torch.tensor([[0.99999797, 0.99999797, 0.99999797, 0.],
                    [-0.99999797, -0.99999797, 0., 0.],
                    [0., 0., 0., 0.],
                    [-0.99999905, -0.99999905, 0., 0.]])
DRGRPO = torch.tensor([[0.5, 0.5, 0.5, 0.],
                       [-0.5, -0.5, 0., 0.],
                       [0., 0., 0., 0.],
                       [-1.0, -1.0, 0., 0.]])


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
    got = call(sol.compute_grpo_outcome_advantage, R.clone(), M, INDEX)
    need(isinstance(got, tuple) and len(got) == 2,
         "return the pair (advantages, returns)")
    advantages, returns = got
    need(tuple(advantages.shape) == (4, 4),
         f"advantages must be (bs, response_length) = (4, 4), got {tuple(advantages.shape)}")

    if close(advantages[2], torch.tensor([0., 0., 0., 0.])) and not close(advantages, ADV):
        pass  # row 2 is legitimately zero; fall through to the full comparison

    flat = R.sum(dim=-1)
    whole_batch = (flat - flat.mean()) / (flat.std() + 1e-6)
    if close(advantages[:, 0], whole_batch):
        raise Fail("standardized the whole batch; GRPO's baseline is the prompt "
                   "group, so rows sharing an index compare only against each other")

    need(close(advantages, ADV),
         f"wrong advantages.\n  expected {ADV.tolist()}\n  got      {advantages.tolist()}")
    need(advantages is returns or close(returns, advantages),
         "with outcome supervision there is nothing to regress, so return the same "
         "tensor for both")

    padded = advantages * (1 - M)
    need(float(padded.abs().sum()) == 0.0,
         "masked-out positions must be zero; multiply through by response_mask")

    # Row 3 is alone in group "b": mean 0, std 1, so its advantage is its own score.
    need(close(advantages[3, 0], -0.99999905),
         "a singleton group takes mean=0 and std=1, not its own mean -- its own mean "
         "would zero the advantage and waste the sample")

    got = call(sol.compute_grpo_outcome_advantage, R.clone(), M, INDEX,
               norm_adv_by_std_in_grpo=False)[0]
    if close(got, ADV):
        raise Fail("norm_adv_by_std_in_grpo=False changed nothing; Dr.GRPO subtracts "
                   "the group mean WITHOUT dividing by the group std")
    need(close(got, DRGRPO),
         f"wrong Dr.GRPO advantages.\n  expected {DRGRPO.tolist()}\n  got      {got.tolist()}")

    equal = call(sol.compute_grpo_outcome_advantage,
                 torch.tensor([[1., 0.], [1., 0.], [1., 0.]]),
                 torch.ones(3, 2), np.array(["z", "z", "z"]))[0]
    need(torch.isfinite(equal).all() and close(equal, torch.zeros(3, 2)),
         "a group whose rewards are all equal has zero std; the epsilon must keep "
         "that finite and near zero")

    population = torch.tensor([1.0, 0.0, 0.5])
    pop_std = float(population.std(unbiased=False))
    sample_std = float(population.std(unbiased=True))
    guess = (1.0 - population.mean()) / (pop_std + 1e-6)
    if close(advantages[0, 0], guess) and abs(pop_std - sample_std) > 1e-6:
        raise Fail("that is the population std; torch.std defaults to the sample "
                   "(n-1) std, which is what verl uses")


def stage_2():
    for name in ("agg_loss", "compute_policy_loss", "kl_penalty"):
        need(hasattr(sol, name),
             f"{name} should be importable from grpo.py -- verl keeps it beside the "
             "GRPO advantage in one core_algos.py, and a GRPO trainer calls it")
    loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
    mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
    need(close(call(sol.agg_loss, loss, mask, "token-mean"), 3.0),
         "the agg_loss re-exported here is your PPO one; make PPO stage 3 pass first")
    k3 = call(sol.kl_penalty, torch.tensor([[-1.0]]), torch.tensor([[-1.2]]), "k3")
    need(close(k3, torch.tensor([[0.01873076]])),
         "the kl_penalty re-exported here is your PPO one; make PPO stage 7 pass first")


STAGES = [
    ("group-relative outcome advantage", stage_1),
    ("the pieces GRPO shares with PPO", stage_2),
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
print("all GRPO stages pass")
