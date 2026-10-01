"""Staged grader for ``gzero.py``. Hand-worked cases, plus the reference on the same inputs."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gzero as sol  # noqa: E402
import gzero_env as env  # noqa: E402  (gzero.py put GZero/ on the path)


def _reference():
    """GZero/gzero.py, loaded under its own name."""
    spec = importlib.util.spec_from_file_location("gzero_reference", HERE.parent / "gzero.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()
STAGE = 0


class Fail(Exception):
    pass


def unfinished():
    return Fail(f"stage {STAGE} is not filled in yet: open GZero/from_scratch/gzero.py, fill the lines marked "
                f"'TODO stage {STAGE}', and try them with `python GZero/from_scratch/gzero.py`")


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
    generator = env.Generator()
    blind = torch.tensor([1, 3, 3])                     # query 1's own answer, wrong at its blind spot (position 1)
    got = call(sol.hint_delta, generator, 1, 1, blind)
    expected = REF.hint_delta(generator, 1, 1, blind)
    if not close(got, expected):
        if close(got, -expected):
            raise Fail("the sign is flipped: delta = log pi_G(a_t | q) - log pi_G(a_t | q, h). A hint that makes the "
                       "Generator's own answer LESS likely must give delta > 0")
        if close(got, expected * env.T):
            raise Fail("average over the T tokens, (1/T) * sum_t, do not just sum: a longer answer must not earn more")
        if close(got, 0.0):
            raise Fail("delta came out 0: score the SAME answer a_hard twice -- once without the hint, once WITH it "
                       "(generator.token_logps(q, a_hard, hint=h))")
        raise Fail(f"delta for query 1, hint on its blind spot, a_hard = [1, 3, 3]: expected {expected:.4f}, got {got}")
    known = call(sol.hint_delta, generator, 1, 0, torch.tensor([1, 1, 3]))
    need(close(known, REF.hint_delta(generator, 1, 0, torch.tensor([1, 1, 3]))),
         f"a hint on a position the Generator already gets right should give a NEGATIVE delta here "
         f"({REF.hint_delta(generator, 1, 0, torch.tensor([1, 1, 3])):.4f}); got {known}")


def stage_2():
    need(close(call(sol.length_penalty, 360), 0.048), "P_length(360 characters) = 0.03 * max(0, (360 - 200) / 100) = 0.048")
    short = call(sol.length_penalty, 120)
    if close(short, -0.024):
        raise Fail("P_length = lambda * max(0, (|h| - 200) / 100): a hint under 200 characters pays nothing, not a "
                   "negative penalty")
    need(close(short, 0.0), f"P_length(120 characters) should be 0; got {short}")

    outputs = torch.tensor([23, 23, 23, 5, 4, 3, 14, 10])
    got = call(sol.repetition_penalty, outputs)
    expected = REF.repetition_penalty(outputs)
    if close(got, expected * len(outputs)):
        raise Fail("P_BLEU = |C_i| / |B|: divide each cluster's size by the batch size")
    need(close(got, expected), f"P_BLEU for {outputs.tolist()}: expected {expected.tolist()}, got {got}")

    delta, p_length, p_bleu = torch.tensor([0.4, -0.2]), torch.tensor([0.048, 0.0]), torch.tensor([0.375, 0.125])
    got = call(sol.proposer_reward, delta, p_length, p_bleu)
    expected = REF.proposer_reward(delta, p_length, p_bleu)
    if not close(got, expected):
        if close(got, torch.clamp(expected, min=0.0)):
            raise Fail("Eq. 5 has no max(0, .): r = delta - P_length - P_BLEU can be negative")
        if close(got, delta + p_length + p_bleu):
            raise Fail("the penalties are SUBTRACTED: r = delta - P_length - P_BLEU")
        raise Fail(f"r = delta - P_length - P_BLEU: expected {expected.tolist()}, got {got}")


def stage_3():
    generator = env.Generator()
    torch.manual_seed(0)
    pairs = [call(sol.make_pair, generator, 5, env.ALL) for _ in range(300)]
    good = env.GOOD[5]
    chosen_good = sum(float((p["chosen"] == good).float().mean()) for p in pairs) / len(pairs)
    rejected_good = sum(float((p["rejected"] == good).float().mean()) for p in pairs) / len(pairs)
    if chosen_good < rejected_good:
        raise Fail("chosen and rejected are swapped: chosen y_w = a_assisted ~ pi_G(. | q, h), the answer WITH the hint")
    need(rejected_good < 0.3, "the rejected answer must be sampled WITHOUT the hint: a_hard ~ pi_G(. | q)")
    need(chosen_good > 0.6, "the chosen answer must be sampled WITH the hint: a_assisted ~ pi_G(. | q, h)")
    torch.manual_seed(1)
    mine = call(sol.make_pair, generator, 3, 0)
    torch.manual_seed(1)
    theirs = REF.make_pair(generator, 3, 0)
    need(torch.equal(mine["chosen"], theirs["chosen"]) and torch.equal(mine["rejected"], theirs["rejected"]),
         "sample a_assisted first, then a_hard (the reference's order), so the same seed gives the same pair")

    deltas = [0.3, -0.1, 0.8, 0.2, 0.5, -0.4]
    kept = call(sol.lower_half, [{"delta": d} for d in deltas])
    kept = sorted(p["delta"] for p in kept)
    if kept == [0.3, 0.5, 0.8]:
        raise Fail("that is the UPPER half: keep the pairs with the LOWEST delta")
    need(kept == [-0.4, -0.1, 0.2], f"lower half of deltas {deltas}: expected [-0.4, -0.1, 0.2], got {kept}")


def stage_4():
    args = (torch.tensor([-1.0, -2.5]), torch.tensor([-2.0, -2.0]), torch.tensor([-3.0, -1.0]),
            torch.tensor([-2.0, -1.5]), torch.tensor([3.0, 3.0]), torch.tensor([3.0, 3.0]))
    got = call(sol.dpo_loss_ln, *args)
    expected = REF.dpo_loss_ln(*args)
    if not close(got, expected):
        logp_w, ref_w, logp_l, ref_l, len_w, len_l = args
        unnormalised = "divide BOTH log-ratios by their answer's length |y|: r_bar = (1/|y|) * log(pi / pi_ref)"
        for value in (-F.logsigmoid(2.0 * ((logp_w - ref_w) - (logp_l - ref_l))).mean(),
                      -F.logsigmoid(2.0 * ((logp_w - ref_w) - (logp_l - ref_l) / len_l)).mean(),
                      -F.logsigmoid(2.0 * ((logp_w - ref_w) / len_w - (logp_l - ref_l))).mean()):
            if close(got, value):
                raise Fail(unnormalised)
        wrong = {
            "beta = 2.0 multiplies the difference: -log sigmoid(beta * (r_bar_w - r_bar_l))":
                -F.logsigmoid((logp_w - ref_w) / len_w - (logp_l - ref_l) / len_l).mean(),
            "chosen minus rejected: r_bar(x, y_w) - r_bar(x, y_l), so the loss falls as y_w gains on y_l":
                -F.logsigmoid(2.0 * ((logp_l - ref_l) / len_l - (logp_w - ref_w) / len_w)).mean(),
        }
        for message, value in wrong.items():
            if close(got, value):
                raise Fail(message)
        raise Fail(f"length-normalised DPO loss: expected {float(expected):.5f}, got {float(got):.5f}")

    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.9, 0.4, 0.4]])
    got = call(sol.group_advantage, rewards)
    expected = REF.group_advantage(rewards)
    if not close(got, expected):
        for value in ((rewards - rewards.mean()) / (rewards.std() + 1e-6),
                      (rewards - rewards.mean()) / (rewards.std(1, keepdim=True) + 1e-6),
                      (rewards - rewards.mean(1, keepdim=True)) / (rewards.std() + 1e-6)):
            if close(got, value):
                raise Fail("each row is its own group: take BOTH the mean and the std along dim=1, with keepdim=True")
        raise Fail(f"A = (r - mean(r_group)) / (std(r_group) + eps); got {got}")

    g = torch.Generator().manual_seed(0)
    logp, old_logp, advantages = -torch.rand(12, generator=g) * 2, -torch.rand(12, generator=g) * 2, torch.randn(12, generator=g)
    got = call(sol.clipped_loss, logp, old_logp, advantages)
    expected = REF.clipped_loss(logp, old_logp, advantages)
    if not close(got, expected):
        ratio = torch.exp(logp - old_logp)
        if close(got, -(torch.clamp(ratio, 0.8, 1.2) * advantages).mean()):
            raise Fail("take the min of the unclipped and clipped terms: min(r_i * A_i, clip(r_i) * A_i)")
        if close(got, -expected):
            raise Fail("the loss is the NEGATIVE objective: -(1/K) sum_i min(...)")
        raise Fail(f"clipped loss: expected {float(expected):.5f}, got {float(got):.5f}")


def stage_5():
    """No new code: the loop runs on your pieces."""
    _, _, history = sol.gzero(seed=0)
    _, _, reference = REF.gzero(seed=0)
    mine = [float(r["p_good"].mean()) for r in history]
    theirs = [float(r["p_good"].mean()) for r in reference]
    need(all(abs(a - b) < 1e-4 for a, b in zip(mine, theirs)),
         f"after 2 rounds the Generator's p(good) is {[round(x, 3) for x in mine]}; the reference reaches "
         f"{[round(x, 3) for x in theirs]}")
    print("  your G-Zero, seed 0, 2 rounds:")
    for i, record in enumerate(history, 1):
        favourite = max(range(env.QUERIES), key=lambda q: record["by_query"][q])
        print(f"    round {i}: Proposer's favourite query {favourite} ({record['by_query'][favourite]:.2f});"
              f" Generator's unassisted p(good) {float(record['p_good_before'].mean()):.3f} -> "
              f"{float(record['p_good'].mean()):.3f}")
    print("  No answer was checked: the Generator improved from its own probabilities, with and without hints.")


STAGES = [
    ("Hint-delta (Eq. 3)", stage_1),
    ("the Proposer's reward r (Eq. 4-5)", stage_2),
    ("the DPO pairs and the lower-50% filter", stage_3),
    ("length-normalised DPO (Eq. 6) and GRPO", stage_4),
    ("the loop, on your pieces", stage_5),
]

for number, (name, stage) in enumerate(STAGES, 1):
    STAGE = number
    try:
        stage()
        print(f"stage {number}: {name} -- pass")
    except Fail as exc:
        print(f"stage {number}: {name} -- FAIL\n  {exc}")
        if "`python GZero/from_scratch/" not in str(exc):
            print("\n  Tip: `python GZero/from_scratch/gzero.py` shows your numbers next to the expected ones.")
        raise SystemExit(1)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
print("all G-Zero stages pass")
