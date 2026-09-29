"""Staged grader for ``simple_grpo.py``. Hand-worked constants, plus the reference on seeded random batches."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import simple_grpo as sol  # noqa: E402
import group_env as env  # noqa: E402  (simple_grpo.py put SimpleGRPO/ on the path)

# The worked example: one HARD question, four attempts (totals 0.8, -1.0, -1.1, 0.8).
R = torch.tensor([[-0.1, -0.1, 1.0], [0.0, 0.0, -1.0], [-0.1, 0.0, -1.0], [0.0, -0.1, 0.9]])
ADV = [0.8653927, -0.8186147, -0.9121707, 0.8653927]
DR = [0.925, -0.875, -0.975, 0.925]
# Two questions of three attempts: totals [0.8, -1.0, 0.9] and [1.0, 0.9, 1.0].
R2 = torch.tensor([[-0.1, -0.1, 1.0], [0.0, 0.0, -1.0], [-0.1, 0.0, 1.0],
                   [0.0, 0.0, 1.0], [0.0, -0.1, 1.0], [0.0, 0.0, 1.0]])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference():
    """SimpleGRPO/simple_grpo.py, with the reference SimplePPO clip (not yours) inside it."""
    module = _load("simple_grpo_reference", HERE.parent / "simple_grpo.py")
    module.policy_loss = _load("simple_ppo_reference", ROOT / "SimplePPO" / "simple_ppo.py").policy_loss
    return module


STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open SimpleGRPO/from_scratch/simple_grpo.py, fill the lines "
                f"marked 'TODO stage {STAGE}', and try them with `python SimpleGRPO/from_scratch/simple_grpo.py`")


def has_blank(fn):
    """A `...` left on a 'TODO stage N' line. Bare `...` never raises, so look."""
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
        raise Fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}") from exc
    if value is Ellipsis:
        raise unfinished()
    return value


def num(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def close(actual, expected, atol=1e-4):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    actual = torch.as_tensor(actual, dtype=torch.float32)
    expected = torch.as_tensor(expected, dtype=torch.float32)
    return actual.shape == expected.shape and torch.allclose(actual, expected, atol=atol)


def random_steps(seed):
    """A policy and 10 flattened steps with a reference log-prob, all random but reproducible."""
    torch.manual_seed(seed)
    policy = env.Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.randn(12, 2))
    states, actions = torch.randint(0, 12, (10,)), torch.randint(0, 2, (10,))
    ref_logp = torch.rand(10).clamp(0.05, 0.95).log()
    return policy, states, actions, ref_logp


def stage_1():
    got = call(sol.group_advantage, R, 4)
    need(isinstance(got, torch.Tensor), "return a tensor of advantages")
    need(tuple(got.shape) == (4, 3), f"advantages should be (episodes, turns) = (4, 3), one per step; got "
         f"{tuple(got.shape)}. Expand the per-episode column to the shape of rewards")
    turn0 = got[:, 0]
    if close(turn0, (R[:, -1] - R[:, -1].mean()) / (R[:, -1].std() + 1e-6)):
        raise Fail("these use only the last turn's reward: sum each episode's rewards (the search costs too)")
    pop = R.sum(1).std(unbiased=False)
    if close(turn0, (R.sum(1) - R.sum(1).mean()) / pop, atol=1e-3):
        raise Fail("that is the population std (divide by n). GRPO uses torch.std's default, the sample std "
                   "(divide by n - 1): expected 1.0689 for this group, not 0.9257")
    need(close(turn0, ADV), f"wrong advantages.\n  expected {[round(a, 3) for a in ADV]}\n  "
         f"got      {[round(a, 3) for a in turn0.tolist()]}\n"
         "  by hand: totals [0.8, -1.0, -1.1, 0.8], mean -0.125, sample std 1.0689")
    need(close(got, torch.tensor(ADV).unsqueeze(1).expand(4, 3)),
         "turn 0 is right, but every turn of an episode should carry the same advantage")

    dr = call(sol.group_advantage, R, 4, scale_by_std=False)
    if close(dr[:, 0], ADV):
        raise Fail("scale_by_std=False still divides: Dr.GRPO only subtracts the group mean")
    need(close(dr[:, 0], DR), f"Dr.GRPO: expected {DR}, got {[round(a, 3) for a in dr[:, 0].tolist()]}")

    two = call(sol.group_advantage, R2, 3)
    totals = R2.sum(1)
    if close(two[:, 0], (totals - totals.mean()) / (totals.std() + 1e-6)):
        raise Fail("with two questions, each attempt must be compared with its OWN group only: view the "
                   "totals as (questions, group_size) and take the mean and std of each row")
    ref = _reference()
    need(close(two, ref.group_advantage(R2, 3)), "two groups of three: differs from the reference. Reshape with "
         "group_size (not a fixed number), and take the mean and std along dim 1 with keepdim=True")
    need(not two.requires_grad, "the advantage is a target: compute it under torch.no_grad()")


def stage_2():
    policy, states, actions, ref_logp = random_steps(1)
    ref = _reference()
    expected = ref.kl_penalty(policy, states, actions, ref_logp)
    got = call(sol.kl_penalty, policy, states, actions, ref_logp)
    need(isinstance(got, torch.Tensor) and got.requires_grad,
         "the penalty must carry gradient: take log pi_theta from policy.dist(states).log_prob(actions)")
    need(got.numel() == 1, f"return one number (the mean), got shape {tuple(got.shape)}")
    logp = policy.dist(states).log_prob(actions)
    backwards = logp - ref_logp
    if close(got, (torch.exp(backwards) - backwards - 1).mean()):
        raise Fail("the ratio is upside down: eq. 4 uses x = pi_ref / pi_theta, so log x = ref_logp - log pi_theta")
    if close(got, (ref_logp - logp).mean()) or close(got, (logp - ref_logp).mean()):
        raise Fail("that is just the log-ratio (k1), which can be negative. Eq. 4 is x - log x - 1, with x = pi_ref / pi_theta")
    need(close(got, expected), f"expected {num(expected):.5f}, got {num(got):.5f}\n"
         "  mean(exp(log_ratio) - log_ratio - 1), with log_ratio = ref_logp - log pi_theta(a|s)")
    at_ref = call(sol.kl_penalty, policy, states, actions, logp.detach())
    need(close(at_ref, 0.0), "where pi_theta = pi_ref the penalty must be exactly 0")


def stage_3():
    ref = _reference()
    policy, states, actions, old_logp = random_steps(2)
    advantages = torch.randn(10)
    try:
        mine = sol.policy_loss(policy, states, actions, old_logp, advantages)
        ok = isinstance(mine, torch.Tensor) and close(mine, ref.policy_loss(policy, states, actions, old_logp, advantages))
    except Exception:
        ok = False
    if not ok:
        raise Fail("stage 3 uses YOUR policy_loss from SimplePPO/from_scratch/simple_ppo.py, and it is not right "
                   "yet. Finish SimplePPO first: ./scripts/run_simple_ppo.sh check")

    def one_update(update, **kwargs):
        torch.manual_seed(7)
        policy, ref_policy = env.Policy(), env.Policy()
        batch = env.rollout(policy, ref_policy, 2, 8)
        batch["advantages"] = ref.group_advantage(batch["rewards"], 8)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
        with torch.no_grad():                            # start away from pi_ref, so the KL has a gradient
            policy.logits.add_(0.5 * torch.randn(12, 2))
        metrics = update(policy, optimizer, batch, **kwargs)
        return policy, metrics

    policy, metrics = one_update(lambda *a, **k: call(sol.grpo_update, *a, **k))
    expected, _ = one_update(ref.grpo_update)
    need(isinstance(metrics, dict) and "kl" in metrics, "grpo_update must return the metrics dict")
    need(metrics["kl"] > 0, "the KL term is 0: call kl_penalty on this minibatch's states, actions and ref_logp")
    if not close(policy.logits, expected.logits):
        no_kl, _ = one_update(ref.grpo_update, beta=0.0)
        minus, _ = one_update(ref.grpo_update, beta=-0.04)
        if close(policy.logits, no_kl.logits):
            raise Fail("the KL is computed but not in the loss: loss = pg + beta * kl")
        if close(policy.logits, minus.logits):
            raise Fail("the KL is subtracted: it is a penalty, so the loss ADDS beta * kl")
        raise Fail("after one grpo_update the policy differs from the reference: loss = pg + beta * kl, "
                   "with kl = kl_penalty(policy, states[idx], actions[idx], ref_logp[idx])")


def stage_4():
    """No new code: your three pieces train the toy."""
    ref = _reference()
    mine, policy, dead = env.train(sol, seed=0, iterations=60)
    theirs, _, _ = env.train(ref, seed=0, iterations=60)
    need(abs(mine[-1] - theirs[-1]) < 1e-4, f"after 60 iterations J is {mine[-1]:.3f}; the reference reaches "
         f"{theirs[-1]:.3f}")
    print(f"  your GRPO on the multi-step toy (seed 0): J {mine[0]:.3f} after 1 iteration -> "
          f"{mine[9]:.3f} after 10 -> {mine[-1]:.3f} after 60.   best possible {env.BEST_J}")
    print(f"  dead groups (all 8 attempts scored the same): {dead[0]:.2f} of the first batch, "
          f"{sum(dead[-10:]) / 10:.2f} of the last ten")
    print("  learned p(search):")
    for row in env.describe(policy):
        print("  " + row)
    print("  HARD should search twice then stop; EASY should never search.")


STAGES = [
    ("the group advantage (DeepSeekMath §4.1.2)", stage_1),
    ("the KL to pi_ref, k3 (eq. 4)", stage_2),
    ("the loop: your clip + beta * KL (eq. 3)", stage_3),
    ("your GRPO on the multi-step toy", stage_4),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "`python SimpleGRPO/from_scratch/" not in str(exc) and "run_simple_ppo" not in str(exc):
            print("\n  Tip: `python SimpleGRPO/from_scratch/simple_grpo.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all SimpleGRPO stages pass")
