"""Staged grader for ``simple_dpo.py``. Hand-worked constants, plus the reference on seeded random pairs."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simple_dpo as sol  # noqa: E402
import pair_env as env  # noqa: E402  (simple_dpo.py put SimpleDPO/ on the path)

# The worked pair: one HARD question, A = (search, search, skip) chosen over B = (skip, skip, skip).
_STATES, _ACTIONS, _ = env.all_attempts(env.HARD)
PAIR = (_STATES[[6, 0]], _ACTIONS[[6, 0]])
REF_AB = [-2.3434, -1.5325]


def _reference():
    """SimpleDPO/simple_dpo.py, loaded under its own name."""
    spec = importlib.util.spec_from_file_location("simple_dpo_reference", HERE.parent / "simple_dpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open SimpleDPO/from_scratch/simple_dpo.py, fill the lines "
                f"marked 'TODO stage {STAGE}', and try them with `python SimpleDPO/from_scratch/simple_dpo.py`")


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
        if "does not require grad" in str(exc):
            raise Fail("the loss has no gradient: the policy's log-probs must come from sequence_logp(policy, ...), "
                       "not from pairs['reference_...'], and must not be detached") from exc
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


def random_policy(seed):
    torch.manual_seed(seed)
    policy = env.Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.randn(12, 2))
    return policy


def random_logps(seed, n=6):
    """Four (n,) log-prob tensors, the policy's two with gradient."""
    g = torch.Generator().manual_seed(seed)
    policy_chosen = (-3 * torch.rand(n, generator=g)).requires_grad_()
    policy_rejected = (-3 * torch.rand(n, generator=g)).requires_grad_()
    return policy_chosen, policy_rejected, -3 * torch.rand(n, generator=g), -3 * torch.rand(n, generator=g)


def stage_1():
    got = call(sol.sequence_logp, env.Policy(), *PAIR)
    need(isinstance(got, torch.Tensor), "return a tensor: one log-probability per attempt")
    if tuple(got.shape) == (2, 3):
        raise Fail("that is still one log-prob per TURN, shape (pairs, 3). Add up dim 1, the turns: "
                   "log pi(attempt) = the sum of its turns' log-probs")
    need(tuple(got.shape) == (2,), f"expected shape (pairs,) = (2,), got {tuple(got.shape)}")
    if close(got, [r / 3 for r in REF_AB]):
        raise Fail("that is the MEAN over the turns. log pi(attempt) is the log of a product, so it is the "
                   "SUM of the turns' log-probs: DPO compares whole attempts")
    if close(got, [-0.5108, -0.5108]):
        raise Fail("that is only the last turn. The attempt's probability is the product of all three turns'")
    need(close(got, REF_AB), f"log pi_ref of A and B: expected {[round(r, 3) for r in REF_AB]}, got "
         f"{[round(x, 3) for x in got.tolist()]}\n  by hand: A = log 0.4 + log 0.4 + log 0.6, B = 3 log 0.6")

    policy = random_policy(3)
    states, actions = torch.randint(0, 12, (5, 3)), torch.randint(0, 2, (5, 3))
    got = call(sol.sequence_logp, policy, states, actions)
    need(got.requires_grad, "the log-probs must carry gradient: take them from policy.dist(states), no .detach()")
    need(close(got, _reference().sequence_logp(policy, states, actions)),
         "right on pi_ref, wrong on a random policy: use policy.dist(states).log_prob(actions)")


def stage_2():
    ref = _reference()
    pc, pr, rc, rr = random_logps(1)
    for beta in (0.1, 0.5):
        got = call(sol.dpo_loss, pc, pr, rc, rr, beta)
        need(isinstance(got, torch.Tensor), "return the loss as a tensor")
        need(got.numel() == 1, f"return one number, the mean over the pairs; got shape {tuple(got.shape)}")
        expected = ref.dpo_loss(pc, pr, rc, rr, beta)
        if close(got, expected):
            continue
        logits = (pc - pr) - (rc - rr)
        wrong = {
            "the logits are upside down: logits = (log pi(chosen) - log pi(rejected)) - (the same under pi_ref), "
            "so the loss falls as the policy prefers CHOSEN": -F.logsigmoid(-beta * logits).mean(),
            "pi_ref is missing: subtract ref_logratios. Without it the policy is pulled toward the pairs "
            "with nothing holding it near pi_ref": -F.logsigmoid(beta * (pc - pr)).mean(),
            "beta is missing: the loss is -log sigmoid(beta * logits)": -F.logsigmoid(logits).mean(),
            "the loss must be the NEGATIVE log-likelihood: -logsigmoid(...)": F.logsigmoid(beta * logits).mean(),
        }
        for message, value in wrong.items():
            if close(got, value):
                raise Fail(message)
        raise Fail(f"beta {beta}: expected {num(expected):.5f}, got {num(got):.5f}\n"
                   "  mean(-logsigmoid(beta * ((pc - pr) - (rc - rr))))")
    need(call(sol.dpo_loss, pc, pr, rc, rr, 0.1).requires_grad, "the loss must carry gradient to the policy")

    far = call(sol.dpo_loss, torch.tensor([-1000.0]), torch.tensor([-1.0]), torch.tensor([-1.0]),
               torch.tensor([-1.0]), 1.0)
    need(torch.isfinite(far).all() and close(far, 999.0, atol=1e-2),
         f"a pair the policy gets badly wrong (logits -999) gave {num(far)}: log(sigmoid(x)) underflows to "
         "-inf there. Use F.logsigmoid, which stays finite (and equals x for large negative x)")

    got = call(sol.implicit_reward, torch.tensor([-1.177, -2.0]), torch.tensor([-2.343, -2.0]), 0.1)
    if close(got, torch.tensor([1.166, 0.0])):
        raise Fail("implicit_reward is missing beta: beta * (log pi - log pi_ref)")
    if close(got, torch.tensor([-0.1166, 0.0])):
        raise Fail("implicit_reward is upside down: beta * (log pi - log pi_ref), positive where pi rose")
    need(close(got, torch.tensor([0.1166, 0.0])), f"implicit_reward: expected [0.117, 0.0], got "
         f"{[round(x, 3) for x in num_list(got)]}")


def num_list(x):
    return [float(v) for v in torch.as_tensor(x).detach().flatten()]


def stage_3():
    ref = _reference()

    def one_update(update, **kwargs):
        torch.manual_seed(7)
        policy, ref_policy = env.Policy(), env.Policy()
        pairs = env.collect_pairs(ref_policy, ref_policy, 8, ref.sequence_logp)
        optimizer = torch.optim.SGD(policy.parameters(), lr=3.0)
        metrics = update(policy, optimizer, pairs, **kwargs)
        return policy, metrics

    def loss_on_pairs(policy):
        """The reference DPO loss of `policy` on stage 3's pairs: lower means it agrees with the rater more."""
        torch.manual_seed(7)
        pairs = env.collect_pairs(env.Policy(), env.Policy(), 8, ref.sequence_logp)
        return float(ref.dpo_loss(ref.sequence_logp(policy, pairs["chosen_states"], pairs["chosen_actions"]),
                                  ref.sequence_logp(policy, pairs["rejected_states"], pairs["rejected_actions"]),
                                  pairs["reference_chosen_logps"], pairs["reference_rejected_logps"]))

    policy, metrics = one_update(lambda *a, **k: call(sol.dpo_update, *a, **k))
    expected, _ = one_update(ref.dpo_update)
    need(isinstance(metrics, dict) and "loss" in metrics, "dpo_update must return the metrics dict")
    if not close(policy.logits, expected.logits):
        if loss_on_pairs(policy) > loss_on_pairs(env.Policy()):
            raise Fail("after your update the policy prefers the REJECTED attempts more than before. dpo_loss takes "
                       "(policy chosen, policy rejected, reference chosen, reference rejected), in that order")
        unchanged = env.Policy()
        if close(policy.logits, unchanged.logits):
            raise Fail("the policy did not move: the policy's log-probs must come from sequence_logp(policy, ...), "
                       "with gradient, not from pairs['reference_...']")
        raise Fail("after one dpo_update the policy differs from the reference: policy_chosen_logps = "
                   "sequence_logp(policy, pairs['chosen_states'][idx], pairs['chosen_actions'][idx]), the same "
                   "for rejected, and loss = dpo_loss(those two, the two reference_..._logps[idx], beta)")
    kw_policy, _ = one_update(lambda *a, **k: call(sol.dpo_update, *a, **k), beta=0.5)
    kw_expected, _ = one_update(ref.dpo_update, beta=0.5)
    need(close(kw_policy.logits, kw_expected.logits), "with beta=0.5 the update differs from the reference: pass "
         "dpo_update's beta on to dpo_loss")


def stage_4():
    """No new code: your three pieces train the toy."""
    ref = _reference()
    mine, policy = env.train(sol, seed=0)
    theirs, _ = env.train(ref, seed=0)
    need(abs(mine[-1] - theirs[-1]) < 1e-4, f"after 60 iterations J is {mine[-1]:.3f}; the reference reaches "
         f"{theirs[-1]:.3f}")
    target = env.true_reward(env.optimal_policy(0.1))
    print(f"  your DPO on the multi-step toy (seed 0): J {mine[0]:.3f} after 1 iteration -> "
          f"{mine[9]:.3f} after 10 -> {mine[-1]:.3f} after 60.   target J(pi*_0.1) {target:.3f}, "
          f"best possible {env.BEST_J}")
    learned = env.implicit_rewards(sol, policy, 0.1)
    true = env.QUALITY - env.QUALITY[:, :1]
    print("  the reward your policy implies, beta * log(pi / pi_ref), relative to 0 searches:")
    for qtype, name in ((env.HARD, "HARD"), (env.EASY, "EASY")):
        print(f"    {name}  learned {[round(float(v), 2) + 0.0 for v in learned[qtype]]}   "
              f"the rater's r {[round(float(v), 2) + 0.0 for v in true[qtype]]}")
    print("  Trained only on which attempt won, it has learned roughly how good each choice is.")


STAGES = [
    ("log pi(attempt), summed over the turns", stage_1),
    ("the DPO loss (eq. 7) and the implicit reward", stage_2),
    ("the loop: minibatches of pairs", stage_3),
    ("your DPO on the multi-step toy", stage_4),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "`python SimpleDPO/from_scratch/" not in str(exc):
            print("\n  Tip: `python SimpleDPO/from_scratch/simple_dpo.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all SimpleDPO stages pass")
