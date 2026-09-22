"""Follow one question through Step 3 and Step 4: ``python Agent0/trace_one_question.py``.

The other walkthroughs show each mechanism on its own. This one tells the
story end to end -- a single question, the ten attempts made on it, the vote,
the gate, the reward, and then what the SAME agent does in Step 4 once it is
frozen.

Every transcript below is a fixed string standing in for what a model would
have generated. Every number is computed from it by the real functions in
``common.py``.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (bleu_cluster_share, cluster_answers,  # noqa: E402
                    curriculum_reward, default_equivalent, difficulty_filter,
                    extract_boxed, extract_question, gate_against_claim,
                    self_consistency_score, tool_reward)

eq = default_equivalent
RULE = "=" * 74


def rule(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


# ===========================================================================
print("  For the iteration flow and the full glossary:")
print("      ./scripts/run_agent0.sh overview")

rule("ACT 1 -- STEP 3: the proposer is being TRAINED")
# ===========================================================================

print("""
The Curriculum Agent is handed the same placeholder prompt every time -- the
dataset's content is never used -- and asked to invent a question. Here is one
thing it generated, verbatim:
""")

# Note the proposer uses the sandbox ITSELF, to check its own answer before
# committing to it. That is what the tool bonus counts -- see common.py.
generation = (
    "<think>I want something a solver will get right about half the time. "
    "Two steps, and an easy place to slip.</think>\n"
    "<question>A rectangle has area 84 and one side of length 6. "
    "What is its perimeter?</question>\n"
    "Let me check my own answer before committing to it.\n"
    "```python\nprint(2*(84/6 + 6))\n```\n"
    "```output\n40.0\n```\n"
    "Side is 84/6 = 14, so the perimeter is 2*(14+6) = \\boxed{40}."
)
for line in generation.split("\n"):
    print(f"    | {line}")

question = extract_question(generation)
claimed = extract_boxed(generation)
print(f"\n  extract_question(...) -> {question!r}")
print(f"  extract_boxed(...)    -> {claimed!r}")
print("""
Note what just happened: the agent wrote the question AND its own answer key.
Nobody checks that 40 is correct. (It is -- 84/6 = 14, perimeter 2*(14+6) = 40.)
If either field had failed to parse, the reward would be -1.0 and we would stop
here.
""")

print("-" * 74)
print("The question now goes to a FROZEN Executor, which attempts it 10 times,")
print("each attempt allowed up to 4 turns with a Python sandbox. Three of the")
print("ten, in full:")
print("-" * 74)

attempt_ok = (
    "The other side is 84/6.\n"
    "```python\nprint(84/6)\n```\n"
    "```output\n14.0\n```\n"
    "So the sides are 14 and 6, perimeter 2*(14+6) = \\boxed{40}."
)
attempt_area = (
    "Perimeter of a rectangle... I need both sides.\n"
    "```python\nprint(84/6)\n```\n"
    "```output\n14.0\n```\n"
    "The answer is the area, \\boxed{84}."
)
attempt_half = (
    "Other side = 84/6 = 14. Sum of the sides is 14+6.\n"
    "```python\nprint(14+6)\n```\n"
    "```output\n20\n```\n"
    "So \\boxed{20}."
)
for label, text in [("ATTEMPT 1 -- correct", attempt_ok),
                    ("ATTEMPT 2 -- answered the wrong question", attempt_area),
                    ("ATTEMPT 3 -- forgot to double", attempt_half)]:
    print(f"\n  {label}")
    for line in text.split("\n"):
        print(f"    | {line}")

print("""
Attempt 3 is the interesting failure: it used the tool, the tool was right,
and the reasoning still dropped a step. That is what "about half the time"
looks like in practice.
""")

# The full set of ten, as the server would have extracted them.
attempts = [attempt_ok, attempt_area, attempt_half, attempt_ok, attempt_ok,
            attempt_half, attempt_ok, "I am not certain.", attempt_ok,
            attempt_area]
extracted = [extract_boxed(a) for a in attempts]

print("  All ten, after extract_boxed:")
for i, (raw, answer) in enumerate(zip(attempts, extracted), 1):
    shown = repr(answer) if answer else "''  <- no boxed answer at all"
    print(f"    attempt {i:>2}: {shown}")

buckets = cluster_answers(extracted, eq)
print(f"\n  cluster_answers(...) -> {buckets}")
print("  Attempt 8 produced nothing, so it is never bucketed. It still counts")
print("  against the denominator -- that is the Step 3 convention.")

majority, p = self_consistency_score(extracted, eq, denominator="candidates")
print(f"\n  majority answer: {majority!r}, agreement p = {p:.2f}  (5 of 10)")

gated = gate_against_claim(majority, claimed, p, eq)
print(f"  gate_against_claim({majority!r}, claimed={claimed!r}, {p:.2f}) -> {gated:.2f}")
print("  The majority matches what the proposer claimed, so the score stands.")
print("  Had the solver's majority been 84, this would be 0.00 -- no credit for")
print("  a question the solver confidently gets wrong.")

siblings = [
    "A rectangle has area 84 and one side of length 6. What is its perimeter?",
    "A rectangle has area 84 and one side of length 6. What is the perimeter?",
    "A rectangle has area 84 and one side of length 7. What is its perimeter?",
    "Let G be a finite simple group of order 84. Classify G.",
]
shares = bleu_cluster_share(siblings, distance_threshold=0.5)
# NOTE: upstream counts fences in the PROPOSER's own generation, not the
# solver's transcript -- compute_score calls calculate_tool_reward(predicts[i]).
tools = tool_reward(generation)
reward = curriculum_reward(gated, True, shares[0], generation)

print("""
rollout.n = 4, so this prompt slot produced four candidate questions. Three of
them are the same question with a word or a number changed -- an untrained
proposer repeats itself, which is exactly what the diversity penalty is for.
It is computed ACROSS the four:""")
for text, share in zip(siblings, shares):
    print(f"    share {share:.2f}  {text[:60]}")

print(f"""
  Assembling our question's reward:
    base            min({gated:.2f}, 1-{gated:.2f})          = {min(gated, 1 - gated):.2f}
    cluster share   3 of the 4 are paraphrases       = -{shares[0]:.2f}
    tool bonus      fences in the PROPOSER's own text  = +{tools:.2f}
                                                      -------
    curriculum_reward(...)                            = {reward:.2f}""")
print("""
  Net negative -- for a question that hit the target difficulty EXACTLY. The
  proposer was repeating itself, and a 0.75 cluster share outweighs the 0.50
  that perfect calibration can earn at most. The pressure here is on the
  batch's variety, not on any single question's quality.""")

print("""
  GRPO now compares this reward against the other three candidates from the
  same prompt slot, standardizes, and updates the Curriculum Agent's weights.

  >>> THE PROPOSER HAS LEARNED SOMETHING. No dataset exists yet. <<<""")


# ===========================================================================
rule("ACT 2 -- STEP 4: the same agent, frozen, now WRITES THE HOMEWORK")
# ===========================================================================

print("""
Step 3 finished and produced a checkpoint. That checkpoint is now frozen and
asked to mass-generate ~8,000 questions. Nothing is being trained from here on.

Each new question is answered 9 times by a frozen Executor -- and this is a
DIFFERENT code path: plain generation, no sandbox, no tool, and crucially no
gate. Following three of the questions it produced:
""")

pool = [
    ("A rectangle has area 84 and one side of length 6. What is its perimeter?",
     "40",
     ["40", "40", "84", "40", "20", "40", "40", "20", "40"],
     "solver gets it right 6 times in 9"),
    ("Compute 7 times 6.",
     "42",
     ["42", "42", "42", "42", "42", "42", "42", "42", "42"],
     "solver never misses"),
    ("Find the sum of all primes p < 100 such that p+2 is also prime.",
     "1027",
     ["1004", "1004", "1027", "1004", "1004", "836", "1004", "1004", "1027"],
     "solver converges on a WRONG answer"),
]

rows = []
for text, truth, sampled, note in pool:
    majority, score = self_consistency_score(sampled, eq, denominator="valid")
    rows.append({"problem": text, "answer": majority, "score": score})
    print(f"  QUESTION: {text}")
    print(f"    proposer's own claim : {truth!r}")
    print(f"    9 plain attempts     : {sampled}")
    print(f"    majority             : {majority!r}   agreement = {score:.2f}")
    print(f"    LABEL WRITTEN        : {majority!r}    <- no gate; the majority")
    print(f"                             simply becomes the answer key")
    print(f"    ({note})\n")

print("""  Look hard at the third one. The proposer claimed 1027. The solver's
  majority was 1004, six times out of nine. In Step 3 that mismatch would have
  scored the question 0 and it would have taught the proposer nothing. Here
  there is no gate, so 1004 is written down as the answer -- and the solver will
  later be trained to reproduce it.

  That is the price of dropping the gate: Step 4's labels are pseudo-labels, and
  a confidently wrong majority produces a confidently wrong one.
""")

kept = difficulty_filter(rows, min_score=0.3, max_score=0.8)
print("-" * 74)
print("  Then the band filter, [0.3, 0.8]:")
for row in rows:
    verdict = "KEEP" if row in kept else "drop"
    why = ("inside the band" if row in kept else
           "too easy -- the solver already knows it"
           if row["score"] > 0.8 else "too incoherent to trust")
    print(f"    score {row['score']:.2f}  {verdict}  ({why})")
    print(f"              {row['problem'][:58]}")

print(f"""
  {len(kept)} of {len(rows)} rows reach train.parquet:""")
print(f"    {'problem':<52}{'answer':>8}{'score':>7}")
for row in kept:
    print(f"    {row['problem'][:50]:<52}{row['answer']:>8}{row['score']:>7.2f}")

print("""
  >>> A DATASET NOW EXISTS. No weights moved in this entire act. <<<""")


# ===========================================================================
rule("ACT 3 -- the handoff to Step 5")
# ===========================================================================

print("""
Step 5 hands each surviving row to the Executor and trains it with ADPO. The
'score' column stops being an agreement fraction and starts being a DIFFICULTY:

    answer the question, get +1 or -1 against the 'answer' column,
    then multiply that lesson by how much the score says to trust it.

So the one number produced by the vote in Act 2 is what decides how hard the
solver gets pushed on that question in Act 3. Follow it in
python Agent0/steps_executor.py

  Step 3   proposer LEARNS TO AIM    weights move, no data out
  Step 4   proposer SETS HOMEWORK    data out, no weights move
  Step 5   solver STUDIES IT         weights move, data consumed
""")
