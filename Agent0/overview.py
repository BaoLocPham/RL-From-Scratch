"""The iteration flow and the glossary: ``python Agent0/overview.py``.

Orientation for the other Agent0 scripts. Two things:

* :func:`print_flow` draws one full iteration -- what generates what, who is
  frozen, where reward is computed, and where weights actually move.
* :func:`print_terms` prints the vocabulary. The walkthroughs call it for their
  own stage so the words are defined before the numbers start.

Nothing here computes anything; it is all fixed text.
"""

TERMS = {
    "roles": ("who is who", [
        ("Curriculum Agent", "The model that WRITES questions. Also 'the proposer'."),
        ("Executor Agent", "The model that ANSWERS them, using a Python sandbox."),
        ("", "Also 'the solver'."),
        ("frozen", "Loaded for inference only -- its weights do not move."),
        ("iteration", "One pass through Steps 3 -> 4 -> 5. Repeated by hand."),
    ]),
    "step3": ("Step 3 -- scoring one proposed question", [
        ("claimed answer", "The \\boxed{} the proposer wrote FOR ITS OWN question."),
        ("", "Self-declared. Nobody ever checks it is correct."),
        ("attempt", "One of the solver's 10 independent tries."),
        ("extraction", "The \\boxed{} contents of one attempt. '' if it produced none."),
        ("bucket", "A group of extractions the grader treats as the same answer."),
        ("majority", "The key of the largest bucket."),
        ("p", "Majority count / 10. How often the solver agreed with itself."),
        ("", "Also called agreement, or the self-consistency score."),
        ("the gate", "Forces p to 0 unless the majority matches the claimed"),
        ("", "answer AND p > 0.1."),
        ("base", "min(p, 1-p) after the gate. THE PART OF THE REWARD THAT"),
        ("", "COMES FROM DIFFICULTY ALONE, before penalty or bonus."),
        ("", "Peaks at 0.50 on a 50/50 split. Range -1.00 to 0.50:"),
        ("", "it is -1.0 instead when the generation failed to parse."),
        ("cluster share", "This question's BLEU cluster size / batch size."),
        ("", "Subtracted raw. A fraction, not a duplicate flag."),
        ("tool bonus", "min(output fences, 4) * 0.05. Counted and capped."),
        ("", "Counted in the PROPOSER's OWN generation, not the"),
        ("", "solver's transcript. Easy to get backwards."),
        ("reward", "base - cluster share + tool bonus. Can be negative."),
        ("rollout.n", "Candidate questions per prompt slot, so also the GRPO"),
        ("", "group size. Agent0 uses 4."),
        ("advantage", "The reward restated as 'how much better than my siblings'."),
    ]),
    "step4": ("Step 4 -- building the dataset", [
        ("pseudo-label", "The majority answer, WRITTEN DOWN as the answer key."),
        ("", "No gate here, so it can be wrong and nothing catches it."),
        ("score", "Agreement again -- but over 9 plain tries, divided by only"),
        ("", "those that answered. A DIFFERENT number from Step 3's p."),
        ("band", "Keep only 0.3 <= score <= 0.8. Inclusive, and flat inside:"),
        ("", "0.5 gets no preference over 0.3."),
        ("difficulty", "The surviving score, carried into Step 5."),
        ("", "HIGH MEANS EASY. Low means hard. Reads backwards."),
        ("train.parquet", "problem, answer (the pseudo-label), score (the difficulty)."),
    ]),
    "step5": ("Step 5 -- ADPO", [
        ("correctness", "+1 if the solver's boxed answer matches the pseudo-label,"),
        ("", "-1 otherwise. Also -1 when there is no boxed answer."),
        ("raw advantage", "The ordinary GRPO advantage, before ADPO touches it."),
        ("trust", "ADPO dial one. Multiply the advantage by 0.50 on the"),
        ("", "hardest kept question, up to 1.00 on the easiest."),
        ("ratio", "How much likelier the current policy is to emit this"),
        ("", "response than the policy that sampled it. 1.0 = no change."),
        ("clip_high", "ADPO dial two. The upper bound on that ratio, widened"),
        ("", "only on hard questions. The lower bound never moves."),
        ("clipfrac", "Fraction of positions where the clip actually bound."),
        ("GRPO", "Use several samples of the SAME prompt as the baseline,"),
        ("", "instead of training a critic."),
        ("ADPO", "GRPO plus those two dials. That is the whole difference."),
    ]),
}


def print_terms(*stages, width=74):
    """Print the glossary for the named stages, or all of them."""
    for stage in (stages or TERMS):
        title, entries = TERMS[stage]
        print(f"\n  TERMS -- {title}")
        print("  " + "-" * (width - 2))
        for term, gloss in entries:
            print(f"  {term:<15} {gloss}")


FLOW = r"""
======================================================================
  ONE ITERATION
  Two models, two algorithms, and a dataset that does not exist until
  halfway through.
======================================================================

  [TRAINING] weights move        (frozen) inference only

       both agents start this iteration as the checkpoints the
       previous one produced -- at iteration 1, the same base model

                              |
   +---------------------------------------------------------------+
   |  STEP 3   train the proposer                                  |
   |                                                               |
   |    [CURRICULUM]  proposes 4 candidate questions per prompt    |
   |                  slot, each with its own claimed answer       |
   |                             |                                 |
   |                             v                                 |
   |    (EXECUTOR)    10 attempts at each, WITH the sandbox        |
   |                             |                                 |
   |                             v                                 |
   |                  vote: cluster the 10 extractions,            |
   |                        take the majority -> p                 |
   |                             |                                 |
   |                             v                                 |
   |                  gate: majority must match the claimed        |
   |                        answer, and p > 0.1, else 0            |
   |                             |                                 |
   |                             v                                 |
   |                  base = min(p, 1-p)                           |
   |                  reward = base - cluster share + tool bonus   |
   |                             |                                 |
   |                             v                                 |
   |                  GRPO: standardize against the 3 siblings     |
   |                             |                                 |
   |    [CURRICULUM] <===========+  weights move                   |
   |                                                               |
   |  OUT: a Curriculum checkpoint.  NO DATASET.                   |
   +---------------------------------------------------------------+
                              |
                              v
   +---------------------------------------------------------------+
   |  STEP 4   build the dataset       NOTHING IS TRAINED HERE     |
   |                                                               |
   |    (CURRICULUM)  now frozen. Mass-generates ~8,000 questions  |
   |                             |                                 |
   |                             v                                 |
   |    (EXECUTOR)    9 attempts at each, PLAIN -- no sandbox      |
   |                             |                                 |
   |                             v                                 |
   |                  vote again, but NO GATE: the majority        |
   |                  simply BECOMES the answer key                |
   |                             |                                 |
   |                             v                                 |
   |                  score = agreement = this question's          |
   |                          difficulty                           |
   |                             |                                 |
   |                             v                                 |
   |                  band filter: keep 0.3 <= score <= 0.8        |
   |                             |                                 |
   |  OUT: train.parquet.  NO WEIGHTS MOVED.                       |
   +---------------------------------------------------------------+
                              |
                              v
   +---------------------------------------------------------------+
   |  STEP 5   train the solver                                    |
   |                                                               |
   |    [EXECUTOR]    answers each kept question, multi-turn,      |
   |                  with the sandbox                             |
   |                             |                                 |
   |                             v                                 |
   |                  correctness: +1 / -1 against the             |
   |                               pseudo-label                    |
   |                             |                                 |
   |                             v                                 |
   |                  ADPO = GRPO advantage, then                  |
   |                    x trust(difficulty)     <- smaller lesson  |
   |                                               on hard ones    |
   |                    clip_high(difficulty)   <- more room to    |
   |                                               explore there   |
   |                             |                                 |
   |    [EXECUTOR]   <===========+  weights move                   |
   |                                                               |
   |  OUT: an Executor checkpoint.  train.parquet is discarded.    |
   +---------------------------------------------------------------+
                              |
                              v
        both checkpoints become iteration N+1's starting models,
        and the new Executor becomes its frozen grader in Step 3

======================================================================
  What never happens: no human-written question, no answer key from
  outside, and no data carried between iterations. Only the two sets
  of weights survive.
======================================================================
"""


def print_flow():
    print(FLOW)


if __name__ == "__main__":
    print_flow()
    print_terms()
