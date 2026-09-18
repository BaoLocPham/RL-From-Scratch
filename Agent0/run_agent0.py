"""One Agent0 iteration in miniature: ``python Agent0/run_agent0.py``.

Steps 3, 4 and 5 run end to end against stub agents -- no vLLM, no sandbox, no
network. The Curriculum Agent is a categorical policy over five question
templates of known difficulty; the Executor is a coin weighted by that
difficulty. Everything else -- the vote, the gate, the tent reward, the band
filter, GRPO, ADPO -- is the real code from ``common.py``.

What to watch: the Curriculum Agent starts uniform over all five templates and
ends concentrated on the medium-difficulty ones, without ever being told which
those are.
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import (adpo_advantage, adpo_policy_loss, adpo_trust_scale,  # noqa: E402
                    bleu_cluster_share, correctness_score, curriculum_reward,
                    default_equivalent, difficulty_filter, gate_against_claim,
                    self_consistency_score)


def _load(name, path):
    """Import a module by path. Both directories hold a file named common.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The Curriculum Agent is trained with plain GRPO, so reuse that module as is.
grpo = _load("grpo_common", ROOT.parent / "GRPO" / "common.py")
compute_grpo_outcome_advantage = grpo.compute_grpo_outcome_advantage
compute_policy_loss = grpo.compute_policy_loss

torch.manual_seed(0)

# Five question templates. SOLVE_RATE is the ground truth the Curriculum Agent
# is never shown -- it only ever sees the Executor's answers.
TEMPLATES = ["trivial", "easy", "medium", "hard", "impossible"]
SOLVE_RATE = torch.tensor([0.95, 0.80, 0.50, 0.25, 0.05])
# Each template is phrased with a fresh nonce, and the clustering below runs at
# a tight threshold, so proposals sharing a template are not treated as
# duplicates of each other. A real batch holds thousands of genuinely different
# questions per difficulty band; this miniature has five, so at the paper's own
# 0.5 threshold the penalty would punish whichever template the policy settled
# on -- crowding standing in for repetition, which is not what it is for.
# Section 5 of steps_agent0.py shows the penalty biting as intended.
WORDING = [
    "Compute {} plus zero.",
    "Compute the area of a {} by 7 rectangle.",
    "A rectangle of area {} has one side 6. Find the other side, then its perimeter.",
    "Find every integer n < {} with exactly three divisors, then sum them.",
    "Classify all finite simple groups of order below {}.",
]
DISTRACTORS = ["17", "3.5", "8", "99", "0", "1", "56", "12"]
CLAIMED = "42"
GROUP_SIZE = 4      # rollout.n: candidates per prompt slot
CANDIDATES = 10     # Executor attempts per question, as in Step 3


def executor_attempts(template, generator, n=CANDIDATES):
    """Simulate one Executor rollout: n answers, right with the template's rate."""
    correct = torch.rand(n, generator=generator) < SOLVE_RATE[template]
    picks = torch.randint(len(DISTRACTORS), (n,), generator=generator)
    return [CLAIMED if correct[i] else DISTRACTORS[picks[i]] for i in range(n)]


def wording(template, generator):
    """Phrase one question, with a nonce so proposals stay textually distinct."""
    nonce = int(torch.randint(1000, 9999, (1,), generator=generator))
    return WORDING[template].format(nonce)


def transcript(template, generator):
    """A stand-in answer transcript; harder questions provoke more tool calls."""
    calls = int(torch.randint(int(template) + 1, (1,), generator=generator))
    return "```output\n...\n```" * calls


generator = torch.Generator().manual_seed(7)
logits = torch.zeros(len(TEMPLATES), requires_grad=True)
optimizer = torch.optim.Adam([logits], lr=0.25)

print("STEP 3 -- train the Curriculum Agent with GRPO")
print("The reward needs no answer key: the Executor grades by agreeing with itself.\n")
print(f"{'step':>5}{'reward':>9}   proposal probabilities")
for step in range(60):
    with torch.no_grad():
        old_logits = logits.detach().clone()
        slots = 8
        proposals = torch.multinomial(old_logits.softmax(-1).expand(slots, -1),
                                      GROUP_SIZE, replacement=True)  # (slots, G)
        flat = proposals.reshape(-1)
        rewards = []
        questions = [wording(template, generator) for template in flat.tolist()]
        shares = bleu_cluster_share(questions, distance_threshold=0.05)
        for i, template in enumerate(flat.tolist()):
            answers = executor_attempts(template, generator)
            majority, raw = self_consistency_score(answers, default_equivalent)
            gated = gate_against_claim(majority, CLAIMED, raw, default_equivalent)
            text = transcript(template, generator)
            rewards.append(curriculum_reward(gated, True, shares[i], text))
        rewards = torch.tensor(rewards)
        groups = torch.arange(slots).repeat_interleave(GROUP_SIZE)
        # verl shapes: the scalar reward goes in a (N, 1) token-level row, and
        # the advantage comes back shaped the same way.
        token_level_rewards = rewards.unsqueeze(-1)
        advantage, _ = compute_grpo_outcome_advantage(
            token_level_rewards, torch.ones_like(token_level_rewards),
            groups.numpy())
        old_logp = old_logits.log_softmax(-1)[flat].unsqueeze(-1)

    logp = logits.log_softmax(-1)[flat].unsqueeze(-1)
    loss = compute_policy_loss(old_logp, logp, advantage, torch.ones_like(logp),
                               cliprange=0.2)[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 15 == 0 or step == 59:
        probabilities = logits.softmax(-1).detach()
        bars = "  ".join(f"{name[:4]} {p:.2f}"
                         for name, p in zip(TEMPLATES, probabilities))
        print(f"{step:>5}{rewards.mean():>9.3f}   {bars}")

final = logits.softmax(-1).detach()
print(f"\nmost-proposed template: {TEMPLATES[int(final.argmax())]}"
      f" (true solve rate {SOLVE_RATE[int(final.argmax())]:.2f})")
print("Nothing told it 0.5 was the target. min(p, 1-p) did.")

print("\nSTEP 4 -- the trained Curriculum Agent curates a dataset")
print("No gate here: the Executor's plurality BECOMES the label.\n")
rows = []
for _ in range(24):
    template = int(torch.multinomial(final, 1))
    answers = executor_attempts(template, generator, n=9)
    majority, score = self_consistency_score(answers, default_equivalent,
                                             denominator="valid")
    rows.append({"problem": wording(template, generator), "answer": majority,
                 "score": score, "template": template})

kept = difficulty_filter(rows, min_score=0.3, max_score=0.8)
print(f"generated {len(rows)} questions, kept {len(kept)} inside [0.3, 0.8]")
histogram = {}
for row in rows:
    bucket = "kept" if row in kept else "dropped"
    histogram.setdefault(TEMPLATES[row["template"]], [0, 0])
    histogram[TEMPLATES[row["template"]]][bucket == "kept"] += 1
for name, (dropped, keeps) in histogram.items():
    print(f"  {name:<11} kept {keeps:>2}   dropped {dropped:>2}")
print("Each kept row carries its score forward as ADPO's difficulty label.")

print("\nSTEP 5 -- train the Executor on that dataset with ADPO")
print("Reward is plain +1/-1; the difficulty label only scales the lesson.\n")
if not kept:
    print("nothing survived the filter this seed; rerun with a different seed")
    raise SystemExit

scores = torch.tensor([row["score"] for row in kept], dtype=torch.float32)
groups = torch.arange(len(kept))
executor_logits = torch.zeros(len(kept), 3, requires_grad=True)
# Plain SGD, deliberately: Adam rescales each parameter's step by its own
# gradient history, which would wash out exactly the constant factor ADPO
# applies. The trust dial is only visible through an optimizer that respects
# gradient magnitude.
executor_optimizer = torch.optim.SGD([executor_logits], lr=1.0)
RIGHT_APPROACH = 1  # of three; the Executor has to find it by trial

for step in range(25):
    with torch.no_grad():
        old_logits = executor_logits.detach().clone()
        picks = torch.multinomial(old_logits.softmax(-1), 4, replacement=True)
        responses = [["\\boxed{17}", "\\boxed{42}", "\\boxed{8}"][p]
                     for p in picks.reshape(-1).tolist()]
        rewards = torch.tensor([correctness_score(r, "42") for r in responses])
        token_level_rewards = rewards.unsqueeze(-1)              # (N, 1)
        response_mask = torch.ones_like(token_level_rewards)
        flat_index = [f"q{i}" for i in groups.repeat_interleave(4).tolist()]
        flat_scores = scores.repeat_interleave(4)
        advantage, _ = adpo_advantage(token_level_rewards, response_mask,
                                      flat_index, flat_scores)
        old_logp = old_logits.log_softmax(-1).gather(1, picks).reshape(-1, 1)

    logp = executor_logits.log_softmax(-1).gather(1, picks).reshape(-1, 1)
    loss = adpo_policy_loss(old_logp, logp, advantage, torch.ones_like(logp),
                            flat_scores)[0]
    executor_optimizer.zero_grad()
    loss.backward()
    executor_optimizer.step()

probabilities = executor_logits.softmax(-1).detach()[:, RIGHT_APPROACH]
buckets = {}
for score, probability in zip(scores.tolist(), probabilities.tolist()):
    buckets.setdefault(round(score, 2), []).append(probability)
print(f"{'difficulty':>11}{'rows':>6}{'trust scale':>13}{'P(correct)':>12}")
for score in sorted(buckets):
    rows_here = buckets[score]
    scale = float(adpo_trust_scale(torch.tensor(score)))
    print(f"{score:>11.2f}{len(rows_here):>6}{scale:>13.2f}"
          f"{sum(rows_here) / len(rows_here):>12.3f}")
print("\nEvery row saw the same +1/-1 signal for the same number of steps. The")
print("harder ones moved less, because ADPO shrank their advantage by the trust")
print("scale in column three. That is the whole algorithm.")
