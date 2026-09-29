"""Staged grader for ``simple_ppo.py``. Hand-worked constants, plus the reference on seeded random batches."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simple_ppo as sol  # noqa: E402
import env  # noqa: E402  (simple_ppo.py put SimplePPO/ on the path)

# Three episodes of three turns: rewards and the critic's predictions at collection.
R = torch.tensor([[0.0, -0.1, 1.0], [-0.1, -0.1, 0.5], [0.0, 0.0, 1.2]])
V = torch.tensor([[0.5, 0.6, 0.8], [0.2, 0.4, 0.1], [0.9, 0.3, 0.7]])
ADV = [[0.308, 0.26, 0.2], [0.036, -0.08, 0.4], [0.04, 0.8, 0.5]]              # gamma 1, lam 0.8
RET = [[0.808, 0.86, 1.0], [0.236, 0.32, 0.5], [0.94, 1.1, 1.2]]
ADV_NO_NEXT = [[-0.932, -0.54, 0.2], [-0.444, -0.18, 0.4], [-0.82, 0.1, 0.5]]    # nextvalues never updated
TEN_ITERATIONS_J = 0.78723558                                                   # seed 0


def _reference():
    """SimplePPO/simple_ppo.py, loaded under its own name."""
    spec = importlib.util.spec_from_file_location("simple_ppo_reference", HERE.parent / "simple_ppo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open SimplePPO/from_scratch/simple_ppo.py, fill the lines "
                f"marked 'TODO stage {STAGE}', and try them with `python SimplePPO/from_scratch/simple_ppo.py`")


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
    return torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32), atol=atol)


def random_steps(seed):
    """A policy, a critic and 10 flattened steps, all random but reproducible."""
    torch.manual_seed(seed)
    policy, critic = env.Policy(), env.Critic()
    with torch.no_grad():
        policy.logits.copy_(torch.randn(12, 2))
        critic.v.copy_(torch.randn(12))
    states, actions = torch.randint(0, 12, (10,)), torch.randint(0, 2, (10,))
    old_logp, advantages, returns = torch.rand(10).clamp(0.05, 0.95).log(), torch.randn(10), torch.randn(10)
    return policy, critic, states, actions, old_logp, advantages, returns


def stage_1():
    got = call(sol.compute_gae, R, V, 1.0, 0.8)
    need(isinstance(got, tuple) and len(got) == 2, "return the pair (advantages, returns)")
    advantages, returns = got
    need(tuple(advantages.shape) == (3, 3), f"advantages should be (episodes, turns) = (3, 3), got {tuple(advantages.shape)}")
    if close(advantages, ADV_NO_NEXT):
        raise Fail("nextvalues never changes, so every delta uses V(s_{t+1}) = 0. After each turn, set "
                   "nextvalues to this turn's values -- it is V(s_{t+1}) for the turn to its left")
    if close(advantages, [[0.1, 0.1, 0.2], [0.1, -0.4, 0.4], [-0.6, 0.4, 0.5]]):
        raise Fail("these are the one-step deltas: the advantage must also carry gamma * lam * A_{t+1}")
    need(close(advantages, ADV), f"wrong advantages.\n  expected {ADV}\n  got      {advantages.tolist()}\n"
         "  episode 0 by hand: delta = [0.1, 0.1, 0.2]; A_2 = 0.2, A_1 = 0.1 + 0.8 * 0.2 = 0.26, "
         "A_0 = 0.1 + 0.8 * 0.26 = 0.308")
    need(close(returns, RET), f"advantages are right but returns are not: returns = advantages + values.\n"
         f"  expected {RET}\n  got      {returns.tolist()}")
    ref = _reference()
    for gamma, lam in ((0.9, 0.95), (1.0, 0.0), (1.0, 1.0)):
        mine, theirs = call(sol.compute_gae, R, V, gamma, lam), ref.compute_gae(R, V, gamma, lam)
        need(close(mine[0], theirs[0]) and close(mine[1], theirs[1]),
             f"gamma {gamma}, lam {lam}: differs from the reference -- use gamma and lam, not fixed numbers")
    lam1 = ref.compute_gae(R, V, 1.0, 1.0)[0]
    need(close(lam1[:, 0], R.sum(1) - V[:, 0]),
         "with gamma = lam = 1, A_0 should equal the episode's total reward minus V(s_0)")


def stage_2():
    policy, _, states, actions, old_logp, advantages, _ = random_steps(1)
    ref = _reference()
    expected = ref.policy_loss(policy, states, actions, old_logp, advantages)
    got = call(sol.policy_loss, policy, states, actions, old_logp, advantages)
    need(isinstance(got, torch.Tensor) and got.requires_grad,
         "the loss must carry gradient: take log-probs from policy.dist(...), no .item() or no_grad")
    need(got.numel() == 1, f"return one number (the mean), got shape {tuple(got.shape)}")
    if close(got, -expected):
        raise Fail("the sign is flipped: return MINUS the mean of the min (optimizers minimise)")
    ratio = torch.exp(policy.dist(states).log_prob(actions) - old_logp)
    if close(got, -torch.max(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()):
        raise Fail("that is the max; the clipped objective takes the MIN of the two (then negates)")
    if close(got, -(ratio * advantages).mean()):
        raise Fail("no clip: the unclipped term alone is L^CPI; take the min with the clipped one")
    need(close(got, expected), f"expected {num(expected):.5f}, got {num(got):.5f}\n"
         "  -mean(min(ratio * A, clamp(ratio, 1 - eps, 1 + eps) * A))")
    at_old = call(sol.policy_loss, policy, states, actions,
                  policy.dist(states).log_prob(actions).detach(), advantages)
    need(close(at_old, -advantages.mean()), "at theta_old every ratio is 1, so the loss is -mean(A)")


def stage_3():
    _, critic, states, _, _, _, returns = random_steps(2)
    ref = _reference()
    got = call(sol.value_loss, critic, states, returns)
    expected = ref.value_loss(critic, states, returns)
    need(isinstance(got, torch.Tensor) and got.requires_grad,
         "the value loss must carry gradient into the critic: use critic(states)")
    if close(got, 2 * expected):
        raise Fail("missing the 0.5")
    need(close(got, expected), f"value_loss: expected {num(expected):.5f}, got {num(got):.5f}")

    policy, _, states, _, _, _, _ = random_steps(3)
    got = call(sol.entropy_bonus, policy, states)
    expected = ref.entropy_bonus(policy, states)
    if close(got, -expected):
        raise Fail("entropy is returned POSITIVE; the loss in stage 4 subtracts it")
    need(close(got, expected), f"entropy_bonus: expected {num(expected):.5f}, got {num(got):.5f}")
    need(close(call(sol.entropy_bonus, env.Policy(), torch.tensor([0])), 0.6730117),
         "p(search) = 0.4 should have entropy 0.673")


def stage_4():
    ref = _reference()

    def one_update(update):
        torch.manual_seed(7)
        policy, critic = env.Policy(), env.Critic()
        batch = env.rollout(policy, critic, 16)
        batch["advantages"], batch["returns"] = ref.compute_gae(batch["rewards"], batch["values"], 1.0, 0.8)
        optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=0.3)
        metrics = update(policy, critic, optimizer, batch)
        return policy, critic, metrics

    policy, critic, metrics = one_update(lambda *a: call(sol.ppo_update, *a))
    expected_policy, expected_critic, _ = one_update(ref.ppo_update)
    need(isinstance(metrics, dict) and "pg_loss" in metrics, "ppo_update must return the metrics dict")
    need(not close(critic.v, torch.zeros(12)), "the critic did not move: add vf_coef * vf to the loss")
    if not close(policy.logits, expected_policy.logits):
        raise Fail("after one ppo_update the policy differs from the reference. Check eq. 9's signs, "
                   "loss = pg + vf_coef * vf - entropy_coeff * ent (the entropy bonus is SUBTRACTED), "
                   "and that idx takes minibatch_size indices from `order`, starting at `start`")
    need(close(critic.v, expected_critic.v), "the policy matches but the critic does not: check vf_coef * vf")


def stage_5():
    """No new code: your four pieces train the toy."""
    ref = _reference()
    mine, policy, critic = env.train(sol, seed=0, iterations=60)
    theirs, _, _ = env.train(ref, seed=0, iterations=60)
    need(abs(mine[-1] - theirs[-1]) < 1e-4, f"after 60 iterations J is {mine[-1]:.3f}; the reference reaches "
         f"{theirs[-1]:.3f}")
    print(f"  your PPO on the multi-step toy (seed 0): J {mine[0]:.3f} after 1 iteration -> "
          f"{mine[9]:.3f} after 10 -> {mine[-1]:.3f} after 60.   best possible {env.BEST_J}")
    print("  learned p(search):")
    for row in env.describe(policy):
        print("  " + row)
    print("  HARD should search twice then stop; EASY should never search.")


STAGES = [
    ("the advantage per step, GAE (eq. 11-12)", stage_1),
    ("L^CLIP, one sample per step (eq. 7)", stage_2),
    ("L^VF and the entropy bonus (eq. 9)", stage_3),
    ("the loop: K epochs of minibatches on eq. 9", stage_4),
    ("your PPO on the multi-step toy", stage_5),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "`python SimplePPO/from_scratch/" not in str(exc):
            print("\n  Tip: `python SimplePPO/from_scratch/simple_ppo.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all SimplePPO stages pass")
