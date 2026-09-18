"""Staged grader for ``agent0.py``. Expected constants are reference outputs."""

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import agent0 as sol
from common import default_equivalent as eq

CANDIDATES = ["42", "\\frac{84}{2}", "17", "42", "0.5", "", "42", "17", "42", ""]


class Fail(Exception):
    pass


def need(condition, message):
    if not condition:
        raise Fail(message)


def call(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except TypeError as exc:
        if "Ellipsis" in str(exc) or "ellipsis" in str(exc):
            raise Fail(f"a TODO is still an ellipsis: {exc}") from exc
        raise Fail(f"raised TypeError: {exc}") from exc
    except Exception as exc:
        raise Fail(f"raised {type(exc).__name__}: {exc}") from exc
    need(value is not None, "returned None; finish this stage's TODO")
    return value


def stage_1():
    got = call(sol.extract_boxed, "the total is \\boxed{42} dollars")
    need(got == "42", f"expected '42', got {got!r}")
    got = call(sol.extract_boxed, "first \\boxed{1}, then \\boxed{2}")
    need(got == "2", f"the LAST boxed answer wins, not the first: got {got!r}")
    got = call(sol.extract_boxed, "so \\boxed{\\frac{1}{2}} exactly")
    if got == "\\frac{1":
        raise Fail("stopped at the first '}'; that brace closes \\frac's numerator, "
                   "not \\boxed -- count depth instead of stopping on sight")
    need(got == "\\frac{1}{2}", f"expected '\\\\frac{{1}}{{2}}', got {got!r}")
    need(call(sol.extract_boxed, "no answer at all") == "",
         "a response with no \\boxed{} must give the empty string")
    need(call(sol.extract_boxed, "truncated \\boxed{42") == "",
         "an unclosed brace must give the empty string, not run off the end")


def stage_2():
    got = call(sol.cluster_answers, ["42", "0.5", "1/2", "42"], eq)
    need(sum(got.values()) == 4, f"every non-empty answer must land in a bucket: {got}")
    if len(got) == 3:
        raise Fail("'0.5' and '1/2' ended up in separate buckets; ask `equivalent`, "
                   "do not compare strings only")
    need(got == {"42": 2, "0.5": 2},
         f"expected {{'42': 2, '0.5': 2}} keyed by first-seen member, got {got}")

    got = call(sol.cluster_answers, ["42", "", "42", ""], eq)
    need(got == {"42": 2},
         f"empty extractions are skipped, never bucketed; got {got}")

    got = call(sol.cluster_answers, ["no solution", "no such x"], eq)
    need(got == {"no solution": 2},
         "two answers that both contain 'no ' share a bucket -- that is the real "
         f"code's cheap shortcut before the expensive grader; got {got}")

    # One-directional equivalence: true as (b, a), false as (a, b).
    def asymmetric(first, second):
        return (first, second) == ("B", "A")

    got = call(sol.cluster_answers, ["A", "B"], asymmetric)
    if got == {"A": 1, "B": 1}:
        raise Fail("equivalence was tried one way only; `grade_answer` normalizes its "
                   "two arguments differently and is not symmetric, so the real loop "
                   "tries both directions before opening a new bucket")
    need(got == {"A": 2}, f"expected {{'A': 2}}, got {got}")


def stage_3():
    answers = ["42", "42", "17", ""]
    majority, score = call(sol.self_consistency_score, answers, eq)
    need(majority == "42", f"expected majority '42', got {majority!r}")
    if abs(score - 2 / 3) < 1e-9:
        raise Fail("that is Step 4's denominator; the default 'candidates' mode is "
                   "Step 3, which divides by every candidate asked for -- including "
                   "the ones that produced no answer")
    need(abs(score - 0.5) < 1e-9, f"expected 0.5 over 4 candidates, got {score}")

    majority, score = call(sol.self_consistency_score, answers, eq, denominator="valid")
    need(abs(score - 2 / 3) < 1e-9,
         f"'valid' divides by the 3 non-empty answers, expected 0.6667, got {score}")

    majority, score = call(sol.self_consistency_score, CANDIDATES, eq)
    need(majority == "42" and abs(score - 0.5) < 1e-9,
         f"'\\\\frac{{84}}{{2}}' is 42: expected ('42', 0.5), got ({majority!r}, {score})")

    need(call(sol.self_consistency_score, ["", ""], eq) == ("", 0.0),
         "when nothing was answered, return ('', 0.0)")


def stage_4():
    need(call(sol.gate_against_claim, "42", "42", 0.5, eq) == 0.5,
         "a majority matching the claim keeps its score")
    need(call(sol.gate_against_claim, "17", "42", 0.7, eq) == 0.0,
         "a confident majority that disagrees with the claim is worth zero, however "
         "consistent it was -- otherwise the proposer farms broken questions")
    got = call(sol.gate_against_claim, "42", "42", 0.1, eq)
    if got == 0.1:
        raise Fail("the guard is strictly greater than the floor; with 10 candidates, "
                   "0.1 is the all-distinct degenerate case, not a 10% consensus")
    need(got == 0.0, f"expected 0.0 at exactly the floor, got {got}")
    need(call(sol.gate_against_claim, "0.5", "1/2", 0.4, eq) == 0.4,
         "the gate compares with `equivalent`, not with ==")
    need(call(sol.gate_against_claim, "", "42", 0.0, eq) == 0.0,
         "an empty majority never passes the gate")


def stage_5():
    got = call(sol.tool_reward, "```output\n5\n``` then ```output\n7\n```")
    need(abs(got - 0.10) < 1e-9, f"two calls at 0.05 each is 0.10, got {got}")
    got = call(sol.tool_reward, "```output\n" * 9)
    if abs(got - 0.45) < 1e-9:
        raise Fail("nine calls paid in full; the count is capped at 4, so an unbounded "
                   "loop of trivial calls cannot farm reward")
    need(abs(got - 0.20) < 1e-9, f"the cap makes this 0.20, got {got}")
    need(call(sol.tool_reward, "plain reasoning, no code") == 0.0,
         "a transcript with no output fences earns nothing")
    got = call(sol.tool_reward, "```output\n5\n```")
    if abs(got - 0.05) > 1e-9:
        raise Fail(f"one call is 0.05, got {got}")


def stage_6():
    got = call(sol.curriculum_reward, 0.5, True, 0.1, "")
    need(abs(got - 0.4) < 1e-9, f"0.5 - 0.1 + 0.0 = 0.4, got {got}")
    got = call(sol.curriculum_reward, 0.9, True, 0.0, "")
    if abs(got - 0.9) < 1e-9:
        raise Fail("a question solved 9 times out of 10 scored 0.9; the base is a tent "
                   "peaking at 0.5, so an easy question must score LOW")
    need(abs(got - 0.1) < 1e-9, f"min(0.9, 0.1) = 0.1, got {got}")
    need(abs(call(sol.curriculum_reward, 0.1, True, 0.0, "") - 0.1) < 1e-9,
         "the tent is symmetric: 0.1 and 0.9 must score the same")

    got = call(sol.curriculum_reward, 0.0, False, 0.1, "")
    if abs(got + 0.1) < 1e-9:
        raise Fail("a malformed generation scored the same as a perfectly solved "
                   "question; its base is -1.0, strictly worse than 0.0")
    need(abs(got + 1.1) < 1e-9, f"-1.0 - 0.1 = -1.1, got {got}")

    got = call(sol.curriculum_reward, 0.5, True, 0.4, "```output\n5\n```")
    need(abs(got - 0.15) < 1e-9,
         f"0.5 - 0.4 + 0.05 = 0.15, got {got}; the cluster share is subtracted raw, "
         "as a fraction, not as a fixed duplicate penalty")


STAGES = [
    ("extract the last boxed answer", stage_1),
    ("cluster equivalent answers", stage_2),
    ("self-consistency score", stage_3),
    ("gate against the proposer's claim", stage_4),
    ("capped tool reward", stage_5),
    ("curriculum tent reward", stage_6),
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
print("all Agent0 stages pass")
