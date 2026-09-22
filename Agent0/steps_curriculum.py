"""The Curriculum Agent's RL, one piece at a time.

    python Agent0/steps_curriculum.py

The proposer half of Agent0: the agent that writes new questions, trained with
plain GRPO on a reward it computes without any answer key.

    WHAT "STEP 3" AND "STEP 4" MEAN

    Agent0's pipeline is numbered 1-5. Steps 1-2 are setup; Step 5 trains the
    Executor. The two in the middle both involve the Curriculum Agent, which is
    why they are easy to confuse -- but only one of them trains it:

    Step 3  TRAINS the Curriculum Agent. It proposes a question, a frozen
            Executor attempts it 10x WITH the Python sandbox, the attempts are
            voted on and GATED against the proposer's own claimed answer, and
            the resulting reward drives a GRPO update. Weights change. No
            dataset comes out of this.

    Step 4  TRAINS NOTHING. The Curriculum Agent, now frozen at the checkpoint
            Step 3 produced, mass-generates ~8,000 questions. A frozen Executor
            answers each 9x by PLAIN generation -- no sandbox, no gate -- and
            its majority answer simply BECOMES the label. How consistent that
            majority was becomes the question's difficulty. Rows scoring inside
            [0.3, 0.8] are written to train.parquet. A dataset comes out; no
            gradient does.

    So Step 3 is the proposer learning to aim; Step 4 is the trained proposer
    building the solver's homework. Both vote on repeated attempts, which is
    why sections 1-2 below can show them disagreeing about the same ten
    answers.

The solver half is ``steps_executor.py``. Read this one first -- it produces
the difficulty label that one consumes.

Set ``RL_IMPL=scratch`` to run against ``from_scratch/agent0.py`` once its
grader passes; every function shown here is one of that exercise's six.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (bleu_cluster_share, default_equivalent,  # noqa: E402
                    difficulty_filter, extract_question)
from overview import print_terms  # noqa: E402

if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from agent0 import (cluster_answers, curriculum_reward, extract_boxed,  # noqa: E402
                        gate_against_claim, self_consistency_score, tool_reward)
else:
    from common import (cluster_answers, curriculum_reward, extract_boxed,  # noqa: E402
                        gate_against_claim, self_consistency_score, tool_reward)

eq = default_equivalent

print("=" * 72)
print("CURRICULUM AGENT -- writes questions, trained with plain GRPO")
print("=" * 72)
print("Two pipeline stages involve this agent, and only one trains it:")
print("  Step 3  trains it. Executor attempts each question 10x WITH tools,")
print("          the vote is gated against the proposer's own claim, and the")
print("          reward drives a GRPO update. Sections 0-5 below.")
print("  Step 4  trains nothing. The frozen, trained proposer mass-generates")
print("          questions; a frozen Executor answers each 9x by PLAIN")
print("          generation, no gate, and its majority BECOMES the label.")
print("          That label is the solver's difficulty. Section 6 below.")
print_terms("step3", "step4")
print("\n  (./scripts/run_agent0.sh overview draws the whole iteration)")

print("0. what the Curriculum Agent hands over")
generation = (
    "<think>I want something with a clean answer.</think>\n"
    "<question>A rectangle is 6 by 7. What is its area?</question>\n"
    "The answer is \\boxed{42}."
)
question = extract_question(generation)
claimed = extract_boxed(generation)
print("question:", question)
print("claimed answer:", repr(claimed))
print("Nobody verifies that claim. It is only used as a gate in section 3.")
print("If either field fails to parse, the reward is -1.0 -- see section 5.")

print("\n1. ten sampled answers collapse into equivalence classes")
sampled = ["42", "\\frac{84}{2}", "17", "42", "0.5", "", "42", "17", "42", ""]
print("raw extractions:", sampled)
print("buckets:        ", cluster_answers(sampled, eq))
print("'\\\\frac{84}{2}' joined '42' because the grader does the algebra.")
print("Without that, self-consistency would be measuring string formatting.")

print("\n2. the same answers, two denominators")
step3 = self_consistency_score(sampled, eq, denominator="candidates")
step4 = self_consistency_score(sampled, eq, denominator="valid")
print(f"Step 3 (consolidate_and_grade, / 10 candidates): {step3[1]:.4f}")
print(f"Step 4 (evaluate.py, / 8 that answered):         {step4[1]:.4f}")
print("Two candidates returned nothing. Step 3 counts that as disagreement;")
print("Step 4 discards them first. Step 4's number is the one that survives")
print("into train.parquet and becomes ADPO's difficulty label.")

print("\n3. the gate: does the consensus back the proposer's claim?")
scenarios = [
    ("too easy", ["42"] * 9 + ["17"]),
    ("well-calibrated", ["42"] * 5 + ["17", "3.5", "8", "99", "0"]),
    ("too hard, coherently wrong", ["17"] * 7 + ["42"] * 3),
    ("genuinely incoherent", [str(value) for value in range(10)]),
]
print(f"{'scenario':<28}{'majority':>10}{'p':>8}{'gated':>8}{'reward':>9}")
rewards = []
for name, attempts in scenarios:
    majority, raw = self_consistency_score(attempts, eq)
    gated = gate_against_claim(majority, "42", raw, eq)
    reward = curriculum_reward(gated, True, 0.0, "")
    rewards.append(reward)
    print(f"{name:<28}{majority:>10}{raw:>8.2f}{gated:>8.2f}{reward:>9.2f}")
print("Rows 3 and 4 both collapse to 0, the same as a perfectly solved question.")
print("A confident WRONG majority earns nothing, however consistent it was.")

print("\n4. the tent: min(score, 1 - score)")
for value in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    height = min(value, 1.0 - value)
    print(f"  score={value:.1f}  base={height:.2f}  {'#' * int(height * 40)}")
print("Peaks at 0.5. Always solvable and never solvable score identically.")

print("\n5. the three terms, separately and then together")
batch = [
    "Find the area of a 6 by 7 rectangle.",
    "Find the area of a 7 by 6 rectangle.",
    "Find the area of a 6 by 8 rectangle.",
    "How many primes are below one hundred?",
]
shares = bleu_cluster_share(batch, distance_threshold=0.5)
for text, share in zip(batch, shares):
    print(f"  cluster share {share:.2f}  {text}")
print("A fraction, not a duplicate flag: the first three are near-paraphrases")
print("that cluster together, and each loses that cluster's whole share of the")
print("batch -- 0.75 here, more than the tent reward's entire 0.5 maximum.")
# The PROPOSER's own generation -- upstream calls calculate_tool_reward on
# predicts[i], the text holding <question>, NOT on the solver's transcript.
proposed = ("<question>What is 6 times 7?</question>\n"
            "Let me check.\n```python\nprint(6*7)\n```\n```output\n42\n```\n"
            "So \\boxed{42}.")
print(f"\ntool reward, proposer used the sandbox once: {tool_reward(proposed):.2f}")
print(f"tool reward capped at four calls:            {tool_reward('```output' * 9):.2f}")
print("Counted in the PROPOSER's own text, not the solver's -- this term")
print("nudges the question-writer to verify its own answer with code.")
print("\nassembled, for the well-calibrated question in a 4-question batch:")
print(f"  base 0.50 - share {shares[0]:.2f} + tool {tool_reward(proposed):.2f}"
      f" = {curriculum_reward(0.5, True, shares[0], proposed):.2f}")
print(f"  malformed generation instead:          "
      f"{curriculum_reward(0.0, False, shares[0], ''):.2f}")
print("A perfectly calibrated question can still end up net negative, purely")
print("for arriving in a crowded batch. Variety is not a tiebreaker here.")

print("\n6. Step 4's filter keeps only the middle of the difficulty range")
rows = [
    {"answer": "42", "score": 0.89},
    {"answer": "42", "score": 0.80},
    {"answer": "42", "score": 0.50},
    {"answer": "42", "score": 0.30},
    {"answer": "42", "score": 0.22},
    {"answer": "", "score": 0.50},
]
kept = difficulty_filter(rows)
for row in rows:
    mark = "keep" if row in kept else "drop"
    print(f"  score={row['score']:.2f} answer={row['answer']!r:<5} -> {mark}")
print("A flat box, not a shaped preference: 0.50 gets no bonus over 0.30.")
print("Bounds are inclusive. Note upload.py's own --max_score default is 0.7,")
print("while the documented invocation passes 0.8.")


print("\nThat is the Curriculum Agent's whole reward, and the label it leaves")
print("behind. Nothing above needed an answer key: the questions are new, so")
print("the only available signal is the Executor agreeing with itself.")
print("The kept rows now carry a difficulty score into the solver's training.")
print("Next: python Agent0/steps_executor.py")
