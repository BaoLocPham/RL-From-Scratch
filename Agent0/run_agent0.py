"""Agent0's loop, run: ``python Agent0/run_agent0.py``.

First one run (seed 0), printing both agents after every step of every
iteration. Then 10 seeds, changing one of the paper's ingredients at a time:

    Agent0             as in the paper: R_C with R_rep, ADPO for the Executor
    GRPO Executor      the Executor trained with plain GRPO: no s(x), eps_high = 0.2
    GRPO, smaller step the same, with the step shrunk to ADPO's average s(x) on the kept questions
    no R_rep           lambda_rep = 0: no penalty for writing the same question again

Takes a few seconds. ``RL_IMPL=scratch`` runs your from_scratch code.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
import agent0 as impl  # noqa: E402
import agent0_env as env  # noqa: E402

SEEDS = 10


def show(iteration, record):
    print(f"\niteration {iteration}")
    print(f"  Step 3  Curriculum Agent pi_theta, p(writing level 1..5): "
          + "  ".join(f"L{lvl} {w:.2f}" for lvl, w in enumerate(record['writes'][:5], 1))
          + f"   MALFORMED {record['writes'][5]:.2f}")
    print(f"  Step 4  kept {record['kept']} of 64 questions (0.3 <= p^ <= 0.8); "
          f"{record['labels_right']} of their majority labels are right")
    print(f"  Step 5  Executor pi_phi, skill logit {record['skill_before']:.2f} -> {record['skill']:.2f};"
          f" p(right) by level: " + "  ".join(f"L{lvl} {p:.2f}" for lvl, p in enumerate(record['p_right'], 1)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    executor = env.Executor()
    print("Agent0 on the toy: 15 questions in 5 levels, a Curriculum Agent of 16 logits, an Executor of one skill logit.")
    print("Before training: the Curriculum Agent writes every output with p 1/16; the Executor is right "
          + ", ".join(f"{executor.p_right(lvl):.2f}" for lvl in range(1, 6)) + " of the time on levels 1..5.")
    print("Each iteration: Step 3 (30 GRPO steps on pi_theta), Step 4 (curate 64), Step 5 (20 ADPO steps on pi_phi).")
    _, _, history = impl.agent0(seed=0, report=show)

    print("""
Watch the favourite level climb: each Step 3 finds where the Executor is right about half the
time, each Step 5 makes the Executor better everywhere, and the next Step 3 has to move up.""")

    print(f"\n{SEEDS} seeds, 3 iterations each:\n")
    print(f"  {'run':<20} | {'skill after 1, 2, 3':<19} | {'favourite level':<15} | {'wrong labels kept':<17} | top question")
    rows = {
        "Agent0": {},
        "GRPO Executor": {"adpo": False},
        "GRPO, smaller step": {"adpo": False, "executor_lr": 0.05 * 0.67},
        "no R_rep": {"lambda_rep": 0.0},
    }
    for name, kwargs in rows.items():
        runs = [impl.agent0(seed=seed, **kwargs)[2] for seed in range(SEEDS)]
        skill = [sum(h[i]["skill"] for h in runs) / SEEDS for i in range(3)]
        favourite = [sum(max(range(5), key=lambda lvl: h[i]["writes"][lvl]) + 1 for h in runs) / SEEDS for i in range(3)]
        wrong = sum(h[i]["kept"] - h[i]["labels_right"] for h in runs for i in range(3)) / SEEDS
        kept = sum(h[i]["kept"] for h in runs for i in range(3)) / SEEDS
        top = sum(h[i]["top_question"] for h in runs for i in range(3)) / (3 * SEEDS)
        print(f"  {name:<20} | {' '.join(f'{s:.2f}' for s in skill):<19} | {' '.join(f'{v:.1f}' for v in favourite):<15} | "
              f"{wrong:>5.1f} of {kept:<8.1f} | {top:.2f}", flush=True)
    print("""
  favourite level: the level pi_theta writes most, averaged over seeds, after each iteration.
  top question: pi_theta's largest probability on any ONE of its 16 outputs (1/16 = 0.06 is uniform).

Reading it:
  Agent0          the favourite level climbs, about one level per iteration, as the skill grows.
  GRPO Executor   learns faster here. ADPO multiplies every advantage by s(x) <= 1, so its steps
                  are smaller: it shrinks the lesson from every low-p^ label, right or wrong.
  smaller step    to separate the two effects: plain GRPO with its step shrunk to ADPO's average
                  s(x) at iteration 1 (0.67). ADPO now comes out slightly ahead: same average
                  step, but spent where the labels are more often right. The gain is small,
                  because on this toy a wrong label (at p^ = 0.3) only pushes the skill down
                  through the few right answers in its group.
  no R_rep        nothing stops pi_theta writing the same question over and over: its top
                  question takes most of the probability. With R_rep it spreads across the three
                  questions of a level. The skill does not suffer: the toy's questions of one
                  level are interchangeable. R_rep is there to keep a real dataset varied.""")
