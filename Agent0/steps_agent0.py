"""Agent0's reward one piece at a time: ``python Agent0/steps_agent0.py``.

Set ``RL_IMPL=scratch`` to run sections 0-5 against ``from_scratch/agent0.py``
once its grader passes. Section 6 is ADPO, which is reference-only and always
comes from ``common.py``.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (adpo_advantage, adpo_clip_high, adpo_trust_scale,  # noqa: E402
                    bleu_cluster_share, correctness_score, default_equivalent,
                    difficulty_filter, extract_question)

if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from agent0 import (cluster_answers, curriculum_reward, extract_boxed,  # noqa: E402
                        gate_against_claim, self_consistency_score, tool_reward)
else:
    from common import (cluster_answers, curriculum_reward, extract_boxed,  # noqa: E402
                        gate_against_claim, self_consistency_score, tool_reward)

eq = default_equivalent

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
transcript = "```python\nprint(6*7)\n```\n```output\n42\n```\nSo \\boxed{42}."
print(f"\ntool reward for one sandbox round-trip: {tool_reward(transcript):.2f}")
print(f"tool reward capped at four calls:       {tool_reward('```output' * 9):.2f}")
print("\nassembled, for the well-calibrated question in a 4-question batch:")
print(f"  base 0.50 - share {shares[0]:.2f} + tool {tool_reward(transcript):.2f}"
      f" = {curriculum_reward(0.5, True, shares[0], transcript):.2f}")
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

print("\n7. Step 5: the Executor's reward is plain right or wrong")
for response, truth in [("So \\boxed{42}.", "42"),
                        ("So \\boxed{17}.", "42"),
                        ("I am not sure.", "42")]:
    print(f"  {response!r:<20} vs {truth!r} -> {correctness_score(response, truth):+.1f}")
print("No partial credit and no format penalty; a missing answer is just wrong.")
print("All of the difficulty-awareness lives in ADPO, below.")

print("\n8. ADPO scales that reward by how hard the question was")
print("Note which agent this is. Everything above trains the Curriculum Agent,")
print("and that half uses plain GRPO, unmodified. ADPO is the OTHER half: it")
print("trains the Executor, the solver, on the questions the Curriculum wrote.")
import torch  # noqa: E402

difficulties = torch.tensor([0.75, 0.35, 0.35, 0.75])
answers = ["\\boxed{42}", "\\boxed{17}", "\\boxed{42}", "\\boxed{8}"]
truths = ["42", "42", "42", "8"]
scalar = torch.tensor([correctness_score(a, g) for a, g in zip(answers, truths)])
step5_rewards = scalar.unsqueeze(-1)                 # (4, 1) token-level rows
mask = torch.ones_like(step5_rewards)
index = ["q"] * 4                                    # all four share one prompt group
# min_advantage_scale=1.0 pins every trust factor at 1.0, which is plain GRPO.
raw = adpo_advantage(step5_rewards.clone(), mask, index, difficulties,
                     min_advantage_scale=1.0)[0][:, 0]
scaled = adpo_advantage(step5_rewards.clone(), mask, index, difficulties)[0][:, 0]
scale = adpo_trust_scale(difficulties)
highs = 1.0 + adpo_clip_high(difficulties)
print(f"{'reward':>7}{'difficulty':>12}{'raw_adv':>9}{'scale':>7}"
      f"{'scaled_adv':>12}{'clip_high':>11}")
for i in range(4):
    print(f"{scalar[i]:>7.1f}{difficulties[i]:>12.2f}{raw[i]:>9.2f}"
          f"{scale[i]:>7.2f}{scaled[i]:>12.2f}{highs[i]:>11.3f}")
print("Rows 1 and 3 earned the same raw advantage -- both correct, same group.")
print("Row 1 (easy) keeps it nearly in full; row 3 (hard) is shrunk to a smaller,")
print("more cautious lesson, and gets the widest clip range to explore with.")
print("\nOn this example the paper notes report raw advantages of 0.58/-1.73,")
print("which is the population std. The code divides by torch.std's default")
print("sample std, giving 0.50/-1.50 -- same ordering, different scale.")
