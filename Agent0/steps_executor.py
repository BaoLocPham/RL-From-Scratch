"""The Executor Agent's RL, one piece at a time.

    python Agent0/steps_executor.py

The solver half of Agent0: the agent that answers the Curriculum Agent's
questions, using a Python sandbox, trained with ADPO -- this paper's own
variant of GRPO.

The proposer half is ``steps_curriculum.py``, and it runs first: the
``difficulty`` values throughout this file are the labels it produced.

Nothing here is part of the ``from_scratch`` exercise; ADPO is reference
material. ``RL_IMPL=scratch`` therefore changes nothing in this file.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (adpo_advantage, adpo_clip_high, adpo_policy_loss,  # noqa: E402
                    adpo_trust_scale, correctness_score)
from overview import equation, print_terms  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False)

print("=" * 72)
print("EXECUTOR AGENT (the solver) -- answers questions, trained with ADPO")
print("=" * 72)
print_terms("step5")

print("\n0. the reward is plain right or wrong")
for response, truth in [("So \\boxed{42}.", "42"),
                        ("So \\boxed{17}.", "42"),
                        ("I am not sure.", "42")]:
    print(f"  {response!r:<20} vs {truth!r} -> {correctness_score(response, truth):+.1f}")
print("No partial credit, and no separate format penalty -- a response with no")
print("\\boxed{} scores -1, the same as being wrong. Compare the Curriculum")
print("Agent, whose reward needed a vote, a gate, a cluster penalty and a tool")
print("bonus. The solver's reward is one comparison, because unlike a brand-new")
print("question, an answer HAS something to be checked against: the pseudo-label")
print("Step 4 attached.")

print("\n1. where 'difficulty' comes from, and what it is not")
print("It is NOT the min(p, 1-p) reward that trained the Curriculum Agent.")
print("It is the Step 4 label: the frozen Executor answering each new question")
print("9 times, plain generation, no tool, no gate -- a different measurement")
print("by a different model for a different purpose. Only rows scoring inside")
print("[0.3, 0.8] survive, so every difficulty ADPO sees is in that band.")

print("\n2. twist one -- trust: how much to believe this question's reward")
print(f"{'difficulty':>11}{'trust weight':>14}{'advantage x':>13}")
for value in (0.30, 0.40, 0.55, 0.70, 0.80):
    tensor = torch.tensor(value)
    trust = ((tensor - 0.3) / (0.8 - 0.3)).clamp(0, 1)
    print(f"{value:>11.2f}{float(trust):>14.2f}{float(adpo_trust_scale(tensor)):>13.2f}")
print()
for d in (0.30, 0.55, 0.80):
    tw = min(max((d - 0.3) / 0.5, 0.0), 1.0)
    equation("trust", "0.5 + clamp((d - 0.3) / (0.8 - 0.3), 0, 1) * 0.5",
             f"0.5 + clamp(({d:.2f} - 0.3) / 0.5, 0, 1) * 0.5",
             f"0.5 + {tw:.2f} * 0.5  =  {0.5 + tw * 0.5:.4f}")
print()
print("Hardest kept question -> x0.50, easiest -> x1.00. A hard question's")
print("+1/-1 is the reward least likely to be right, so it teaches less.")

print("\n3. twist two -- exploration room: how far the policy may move")
print(f"{'difficulty':>11}{'clip range':>22}")
for value in (0.30, 0.40, 0.55, 0.70, 0.80):
    high = float(adpo_clip_high(torch.tensor(value)))
    print(f"{value:>11.2f}{f'[0.80, {1 + high:.3f}]':>22}")
print()
for d in (0.30, 0.80):
    ew = min(max((0.8 - d) / 0.5, 0.0), 1.0)
    equation("eps_high", "0.2 + clamp((0.8 - d) / (0.8 - 0.3), 0, 1) * 0.1",
             f"0.2 + clamp((0.8 - {d:.2f}) / 0.5, 0, 1) * 0.1",
             f"0.2 + {ew:.2f} * 0.1  =  {0.2 + ew * 0.1:.4f}")
print()
print("The LOWER bound never moves; only the upside widens, and only on hard")
print("questions. Being right on a hard question does not yet mean the approach")
print("generalizes, so the policy gets more room to try something else.")
print("DAPO's clip-higher is this same idea with a constant instead of a dial.")

print("\n4. the two dials on one batch")
difficulties = torch.tensor([0.75, 0.35, 0.35, 0.75])
answers = ["\\boxed{42}", "\\boxed{17}", "\\boxed{42}", "\\boxed{8}"]
truths = ["42", "42", "42", "8"]
scalar = torch.tensor([correctness_score(a, g) for a, g in zip(answers, truths)])
token_level_rewards = scalar.unsqueeze(-1)           # (4, 1), verl shapes
response_mask = torch.ones_like(token_level_rewards)
index = ["q"] * 4                                    # one shared prompt group
# min_advantage_scale=1.0 pins every trust factor at 1.0, which is plain GRPO.
raw = adpo_advantage(token_level_rewards.clone(), response_mask, index,
                     difficulties, min_advantage_scale=1.0)[0][:, 0]
scaled = adpo_advantage(token_level_rewards.clone(), response_mask, index,
                        difficulties)[0][:, 0]
scale = adpo_trust_scale(difficulties)
highs = 1.0 + adpo_clip_high(difficulties)
print(f"{'reward':>7}{'difficulty':>12}{'raw_adv':>9}{'scale':>7}"
      f"{'scaled_adv':>12}{'clip_high':>11}")
for i in range(4):
    print(f"{scalar[i]:>7.1f}{difficulties[i]:>12.2f}{raw[i]:>9.2f}"
          f"{scale[i]:>7.2f}{scaled[i]:>12.2f}{highs[i]:>11.3f}")
print()
print("row 3 (hard, correct) in full:")
equation("A_raw", "(reward - group_mean) / (group_std + 1e-6)",
         f"({scalar[2]:+.1f} - {scalar.mean():.4f}) / ({scalar.std():.4f} + 1e-6)",
         f"{raw[2]:+.4f}")
equation("trust", "0.5 + clamp((d - 0.3) / 0.5, 0, 1) * 0.5",
         f"0.5 + clamp(({difficulties[2]:.2f} - 0.3) / 0.5, 0, 1) * 0.5",
         f"{scale[2]:.4f}")
equation("A", "A_raw * trust",
         f"{raw[2]:+.4f} * {scale[2]:.4f}", f"{scaled[2]:+.4f}")
print()
print("Rows 1 and 3 earned the same raw advantage -- both correct, same group.")
print("Row 1 (easy) keeps it nearly in full; row 3 (hard) is shrunk to a")
print("smaller, more cautious lesson, and gets the widest clip range.")
print("\nOn this example the paper notes report raw advantages of 0.58/-1.73,")
print("which is the population std. The code divides by torch.std's default")
print("sample std, giving 0.50/-1.50 -- same ordering, different scale.")

print("\n5. what ADPO is, as a diff against GRPO")
print("The raw_adv column above IS compute_grpo_outcome_advantage: same")
print("grouping by uid, same singleton mean-0/std-1 case, same 1e-6, same")
print("sample std. ADPO adds exactly two things:")
print("  advantage  *= trust_scale(difficulty)      <- section 2")
print("  clip_high   = 0.2 + bonus(difficulty)      <- section 3")
print("That is the whole algorithm. Everything else it runs is verl's.")

print("\n6. the policy loss, where the per-sample bound actually bites")
old_log_prob = torch.zeros(4, 3)
# Rows 0 and 2 get the IDENTICAL ratio, exp(0.25) = 1.284. Row 0 is an easy
# question (bound 1.210) and row 2 is a hard one (bound 1.290), so the same
# policy move is clipped for one and allowed for the other.
log_prob = torch.tensor([[0.25, 0.25, 0.25], [0.05, -0.05, 0.10],
                         [0.25, 0.25, 0.25], [0.15, 0.00, -0.05]])
advantages = scaled.unsqueeze(-1).expand(-1, 3)
mask = torch.ones(4, 3)
ratio = torch.exp(log_prob)
print(f"{'difficulty':>11}{'ratio':>8}{'clip_high':>11}{'clipped?':>10}")
for i in (0, 2):
    bound = float(highs[i])
    print(f"{difficulties[i]:>11.2f}{float(ratio[i, 0]):>8.3f}{bound:>11.3f}"
          f"{('yes' if float(ratio[i, 0]) > bound else 'no'):>10}")
pg_loss, pg_clipfrac, ppo_kl, lower = adpo_policy_loss(
    old_log_prob, log_prob, advantages, mask, difficulties)
print()
equation("ratio", "exp(log_prob - old_log_prob)",
         f"exp({float(log_prob[0, 0]):.4f} - 0.0000)", f"{float(ratio[0, 0]):.4f}")
equation("clipped", "min(max(ratio, 1 - 0.2), 1 + eps_high)",
         f"min(max({float(ratio[0, 0]):.4f}, 0.8000), {float(highs[0]):.4f})",
         f"{min(max(float(ratio[0, 0]), 0.8), float(highs[0])):.4f}   <- row 0 IS clipped")
equation("pg_loss", "agg( max(-A * ratio, -A * clipped) )",
         f"{float(pg_loss):+.6f}")
print(f"\nclipfrac={float(pg_clipfrac):.2f}  ppo_kl={float(ppo_kl):+.4f}"
      f"  clipfrac_lower={float(lower):.2f}")
print("Same move, two verdicts, purely because the questions differ in")
print("difficulty. That is what a per-sample clip bound buys you.")
print("\nOtherwise identical to PPO's compute_policy_loss: the upper bound is a")
print("column vector rather than one shared scalar, the lower bound is fixed,")
print("and ADPO has no dual clip, so clipfrac_lower is a constant zero.")

print("\nThat closes the loop. The Curriculum Agent's checkpoint goes on to")
print("write the next iteration's questions, and this Executor becomes the")
print("frozen grader that scores them. See python Agent0/run_agent0.py")
