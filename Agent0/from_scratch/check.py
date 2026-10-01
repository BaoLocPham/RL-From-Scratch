"""Staged grader for ``agent0.py``. Hand-worked cases, plus the reference on the same inputs."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent0 as sol  # noqa: E402
import agent0_env as env  # noqa: E402  (agent0.py put Agent0/ on the path)


def _reference():
    """Agent0/agent0.py, loaded under its own name."""
    spec = importlib.util.spec_from_file_location("agent0_reference", HERE.parent / "agent0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()
STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open Agent0/from_scratch/agent0.py, fill the lines marked "
                f"'TODO stage {STAGE}', and try them with `python Agent0/from_scratch/agent0.py`")


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
    return value


def close(actual, expected, atol=1e-4):
    actual = torch.as_tensor(actual.detach() if isinstance(actual, torch.Tensor) else actual, dtype=torch.float32)
    expected = torch.as_tensor(expected, dtype=torch.float32)
    return actual.shape == expected.shape and torch.allclose(actual, expected, atol=atol)


def stage_1():
    answers = torch.tensor([0, 0, 3, 0, 5, 3, 0, 1, 0, 2])
    got = call(sol.self_consistency, answers)
    need(isinstance(got, tuple) and len(got) == 2, "return the pair (y_tilde, p_hat)")
    y_tilde, p_hat = got
    need(int(y_tilde) == 0, f"y~ is the answer given most often: 0 (5 votes); got {y_tilde}")
    if close(p_hat, 5.0):
        raise Fail("p^ is a fraction: divide the majority's votes by k, the number of answers")
    if close(p_hat, 5 / len(set(answers.tolist()))):
        raise Fail("divide by k = 10, all the answers -- not by the number of DIFFERENT answers "
                   f"({len(set(answers.tolist()))} here)")
    need(close(p_hat, 0.5), f"p^ = (1/k) * sum_i 1(o_i = y~) = 5 / 10 = 0.5; got {p_hat}")
    tie = call(sol.self_consistency, torch.tensor([3, 3, 0, 0, 7]))
    if int(tie[0]) == 0:
        raise Fail("on a tie (3 and 0, two votes each), y~ must be the answer seen FIRST, 3. A lowest-index "
                   "argmax picks 0 -- the right answer -- and would quietly hand ties to the truth. "
                   "Counter(...).most_common(1) keeps first-seen order")
    need(int(tie[0]) == 3 and close(tie[1], 0.4), f"[3, 3, 0, 0, 7] should give (3, 0.4); got {tie}")


def stage_2():
    for p_hat, expected in ((0.5, 1.0), (0.3, 0.6), (0.9, 0.2), (0.0, 0.0), (1.0, 0.0)):
        got = call(sol.uncertainty_reward, p_hat)
        if not close(got, expected):
            if close(got, 1 - abs(p_hat - 0.5)):
                raise Fail("R_unc = 1 - 2 * |p^ - 0.5|: the factor 2 makes it reach 0 at p^ = 0 and at p^ = 1")
            if close(got, 1 - 2 * (p_hat - 0.5)):
                raise Fail("take the ABSOLUTE distance from 0.5: R_unc is a tent, symmetric around p^ = 0.5")
            raise Fail(f"R_unc({p_hat}) should be {expected}; got {got}")
    need(close(call(sol.tool_reward, 2), 0.1), "R_tool = gamma * min(N_tool, C) = 0.05 * 2 = 0.1 for 2 calls")
    if close(call(sol.tool_reward, 7), 0.35):
        raise Fail("R_tool is capped: min(N_tool, C) with C = 4, so 7 calls pay like 4 (0.2)")
    need(close(call(sol.tool_reward, 7), 0.2), "R_tool(7 calls) = 0.05 * min(7, 4) = 0.2")

    batch = torch.tensor([3, 3, 3, 0, 13, 7, 4, env.MALFORMED])
    got = call(sol.repetition_penalty, batch)
    expected = REF.repetition_penalty(batch)
    if close(got, expected * len(batch)):
        raise Fail("R_rep = lambda_rep * |C_k| / B: divide each cluster's size by the batch size B")
    need(close(got, expected), f"R_rep for {batch.tolist()}: expected {[round(x, 3) for x in expected.tolist()]}, "
         f"got {got}. |C_k| is how many copies of x_i the batch holds (torch.bincount, then index it)")

    well_formed = torch.tensor([1.0, 1.0, 0.0])
    r_unc, r_tool, r_rep = torch.tensor([0.8, 0.1, 1.0]), torch.tensor([0.1, 0.05, 0.1]), torch.tensor([0.375, 0.5, 0.125])
    got = call(sol.curriculum_reward, well_formed, r_unc, r_tool, r_rep)
    expected = REF.curriculum_reward(well_formed, r_unc, r_tool, r_rep)
    if not close(got, expected):
        inner = r_unc + 0.6 * r_tool - r_rep
        wrong = {
            "max(0, .) is missing: a reward below 0 is cut to 0, so the second question earns 0, not a negative":
                well_formed * inner,
            "R_format must MULTIPLY the rest: a malformed question (the third) earns 0 whatever else it scores":
                torch.clamp(inner, min=0.0),
            "lambda_tool = 0.6 weighs R_tool: 1.0 * R_unc + 0.6 * R_tool - R_rep":
                well_formed * torch.clamp(r_unc + r_tool - r_rep, min=0.0),
        }
        for message, value in wrong.items():
            if close(got, value):
                raise Fail(message)
        raise Fail(f"R_C: expected {[round(x, 3) for x in expected.tolist()]}, got {got}")


def stage_3():
    for p_hat, expected in ((0.3, True), (0.8, True), (0.55, True), (0.29, False), (0.81, False)):
        got = call(sol.keep, p_hat)
        if bool(got) != expected and p_hat in (0.3, 0.8):
            raise Fail(f"the band includes its edges: keep(p^) is 0.3 <= p^ <= 0.8, so p^ = {p_hat} is kept")
        need(bool(got) == expected, f"keep({p_hat}) should be {expected}")
    answers = torch.tensor([0, 2, 2, 5])
    got = call(sol.executor_reward, answers, 2)
    if close(got, torch.tensor([-1.0, 1.0, 1.0, -1.0])):
        raise Fail("R_i = 1(o_i = y~) is 1 or 0, not +1 / -1")
    need(close(got, torch.tensor([0.0, 1.0, 1.0, 0.0])), f"R_i for answers {answers.tolist()} against y~ = 2: "
         f"[0, 1, 1, 0]; got {got}")


def stage_4():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.9, 0.4, 0.4]])
    got = call(sol.group_advantage, rewards)
    expected = REF.group_advantage(rewards)
    if not close(got, expected):
        if close(got, (rewards - rewards.mean()) / (rewards.std() + 1e-6)):
            raise Fail("each row is its own group: take the mean and std along dim=1 with keepdim=True, not over "
                       "the whole batch")
        if close(got, (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, unbiased=False, keepdim=True) + 1e-6)):
            raise Fail("use torch.std's default, the sample (n - 1) std")
        raise Fail(f"A^ = (R - mean(R_group)) / (std(R_group) + eps); got {got}")

    for p_hat, scale, eps_high in ((0.3, 0.5, 0.3), (0.55, 0.75, 0.25), (0.8, 1.0, 0.2), (0.9, 1.0, 0.2)):
        got_scale = call(sol.adpo_scale, p_hat)
        if not close(got_scale, scale):
            if close(got_scale, 1.5 - scale):
                raise Fail("s(x) = f(p^) must INCREASE with p^: trust a high-agreement label more, from 0.5 at "
                           "p^ = 0.3 to 1.0 at p^ = 0.8")
            raise Fail(f"adpo_scale({p_hat}) should be {scale}; got {got_scale}")
        got_eps = call(sol.adpo_eps_high, p_hat)
        if not close(got_eps, eps_high):
            if close(got_eps, 0.5 - eps_high):
                raise Fail("eps_high(x) must DECREASE with p^: the hardest kept question (p^ = 0.3) gets the widest "
                           "upper bound, 0.3")
            raise Fail(f"adpo_eps_high({p_hat}) should be {eps_high}; got {got_eps}")

    g = torch.Generator().manual_seed(0)
    logp, old_logp = -torch.rand(12, generator=g) * 2, -torch.rand(12, generator=g) * 2
    advantages = torch.randn(12, generator=g)
    eps_high = 0.2 + 0.1 * torch.rand(12, generator=g)
    got = call(sol.clipped_loss, logp, old_logp, advantages, eps_high)
    expected = REF.clipped_loss(logp, old_logp, advantages, eps_high)
    if not close(got, expected):
        ratio = torch.exp(logp - old_logp)
        if close(got, REF.clipped_loss(logp, old_logp, advantages, torch.full_like(eps_high, 0.2))):
            raise Fail("the upper bound is per sample: clip r_i to 1 + eps_high[i], not 1 + 0.2")
        if close(got, -(torch.minimum(torch.clamp(ratio, min=0.8), 1.0 + eps_high) * advantages).mean()):
            raise Fail("take the min of the unclipped and clipped terms: min(r_i * A_i, clip(r_i) * A_i)")
        if close(got, -expected):
            raise Fail("the loss is the NEGATIVE objective: -(1/G) sum_i min(...), so descent raises the objective")
        raise Fail(f"clipped loss: expected {float(expected):.5f}, got {float(got):.5f}")


def stage_5():
    """No new code: the loop runs on your pieces."""
    _, executor, history = sol.agent0(seed=0)
    _, _, reference = REF.agent0(seed=0)
    mine = [record["skill"] for record in history]
    theirs = [record["skill"] for record in reference]
    need(all(abs(a - b) < 1e-4 for a, b in zip(mine, theirs)),
         f"after 3 iterations the skill is {[round(s, 3) for s in mine]}; the reference reaches "
         f"{[round(s, 3) for s in theirs]}")
    print("  your Agent0, seed 0, 3 iterations:")
    for i, record in enumerate(history, 1):
        favourite = max(range(5), key=lambda lvl: record["writes"][lvl]) + 1
        print(f"    iteration {i}: Curriculum Agent's favourite level {favourite} ({record['writes'][favourite - 1]:.2f});"
              f" kept {record['kept']}, {record['labels_right']} labels right; skill "
              f"{record['skill_before']:.2f} -> {record['skill']:.2f}")
    print("  The favourite level climbs as the skill grows: the two agents push each other.")


STAGES = [
    ("self-consistency, p^ and y~ (Eq. 6)", stage_1),
    ("the Curriculum Agent's reward R_C (Eq. 2-5)", stage_2),
    ("the curated dataset and the Executor's reward (Eq. 7)", stage_3),
    ("GRPO's advantage, ADPO's scale and clip (Eq. 8)", stage_4),
    ("the loop, on your pieces", stage_5),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "`python Agent0/from_scratch/" not in str(exc):
            print("\n  Tip: `python Agent0/from_scratch/agent0.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all Agent0 stages pass")
