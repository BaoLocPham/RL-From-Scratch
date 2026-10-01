"""Agent0, one step at a time: ``python Agent0/steps_agent0.py``.

Follows the loop through every equation, printing each one with the numbers
substituted into it, and the logits of both agents as they move. Set
``RL_IMPL=scratch`` to run the same walkthrough on your
Agent0/from_scratch/agent0.py, and ``./scripts/run_agent0.sh diff`` to compare
the two outputs line by line.
"""

import math
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

torch.set_num_threads(1)


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x, digits=3):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), digits) + 0.0               # + 0.0: no -0.0


# ---------------------------------------------------------------- step 0
banner("STEP 0  two agents, two tables of logits")
executor, curriculum = env.Executor(), env.Curriculum()
print(f"""  The Executor pi_phi answers questions. It is ONE number, its skill logit, now {f(executor.skill)}.
  On a level-L question it picks one of 10 answers (answer 0 is right, 1..9 are wrong),
  with logits [skill - L, 0, 0, 0, 0, 0, 0, 0, 0, 0]:

      p(right | L) = e^(skill - L) / (e^(skill - L) + 9)
""")
for lvl in range(1, env.LEVELS + 1):
    e = math.exp(f(executor.skill) - lvl)
    print(f"    level {lvl}:  e^({f(executor.skill)} - {lvl}) / (e^({f(executor.skill)} - {lvl}) + 9) = "
          f"{e:.3f} / {e + 9:.3f} = {executor.p_right(lvl):.3f}")
print(f"""
  The Curriculum Agent pi_theta writes questions. It is 16 logits: 15 questions (3 per
  level: ids 0-2 are level 1, ..., ids 12-14 level 5) and MALFORMED (id 15), a
  generation with no usable question. All 16 start at 0: each output has p = 1/16 = {1 / 16:.4f}.""")

# ---------------------------------------------------------------- step 1
banner("STEP 1  self-consistency: the Executor answers each question k = 10 times (Eq. 6)")
print("""      p^(x) = (1/k) * sum_i 1(o_i = y~),     y~ = argmax_y sum_i 1(o_i = y)

  Nobody knows the right answers. The only thing to go on is how often the Executor
  agrees with itself. Four questions, 10 answers each:
""")
torch.manual_seed(0)
for question in (3, 0, 13, 7):
    answers = executor.answer(question, 10)
    y_tilde, p_hat = impl.self_consistency(answers)
    votes = int(round(p_hat * 10))
    note = "right" if y_tilde == env.RIGHT else "WRONG: answer 0 is right"
    print(f"    question {question:>2} (level {env.level(question)}):  answers {answers.tolist()}")
    print(f"        y~ = {y_tilde} ({note}),  p^ = {votes} / 10 = {p_hat:.1f}")
print("""
  p^ measures AGREEMENT, not correctness. On the level-5 question the Executor is
  mostly guessing, and by chance a wrong answer got 4 votes: the majority is wrong.""")

# ---------------------------------------------------------------- step 2
banner("STEP 2  the Curriculum Agent's reward, R_C (Eq. 2-5)")
print("""      R_unc(x)  = 1 - 2 * |p^ - 0.5|                    (Eq. 2)  1 at p^ = 0.5, 0 at 0 or 1
      R_tool(x) = gamma * min(N_tool, C),  0.05, C = 4    (Eq. 3)  N_tool = the level: harder needs more tool calls
      R_rep(x)  = lambda_rep * |C_k| / B                  (Eq. 4)  |C_k| = copies of x in the batch
      R_C(x)    = R_format * max(0, 1.0 * R_unc + 0.6 * R_tool - R_rep)       (Eq. 5)

  One batch of B = 8 written questions. Question 3 is written three times, and the
  last one is MALFORMED:
""")
torch.manual_seed(1)
batch = torch.tensor([3, 3, 3, 0, 13, 7, 4, env.MALFORMED])
terms = impl.score_questions(batch, executor)
print(f"  {'x':>3} {'level':>5} | {'p^':>4} | {'R_unc':>5} | {'R_tool':>6} | {'R_rep':>5} | R_C = R_format * max(0, R_unc + 0.6 R_tool - R_rep)")
for i, question in enumerate(batch.tolist()):
    if question == env.MALFORMED:
        print(f"  {question:>3} {'-':>5} | {'-':>4} | {'-':>5} | {'-':>6} | {'-':>5} | R_format = 0  ->  R_C = {f(terms['r_c'][i])}")
        continue
    p, ru, rt, rr, rc = (f(terms[key][i]) for key in ("p_hat", "r_unc", "r_tool", "r_rep", "r_c"))
    print(f"  {question:>3} {env.level(question):>5} | {p:>4} | {ru:>5} | {rt:>6} | {rr:>5} | "
          f"1 * max(0, {ru} + 0.6 * {rt} - {rr}) = {rc}")
print("""
  - The level-2 question is near p^ = 0.5, so it is worth most -- but it was written three
    times, and each copy pays R_rep = 3/8 for it.
  - The level-1 question is too easy (p^ near 1): R_unc is small.
  - The level-5 question scored well by luck. With 10 answers and 9 wrong ones, pure
    guesses still agree 2 or 3 times in 10, so p^ = 0.2-0.3 and R_unc = 0.4-0.6:
    R_unc cannot tell a guess from a fair hard question.
  - MALFORMED earns 0 whatever else: R_format multiplies everything.
  - max(0, .) means no reward is ever negative.""")

# ---------------------------------------------------------------- step 3
banner("STEP 3  GRPO moves the Curriculum Agent's logits")
print("""  The 8 questions form 2 groups of 4 (4 questions per prompt). Each reward is compared
  with its own group:

      A^_i = (R_i - mean(R_group)) / std(R_group)
""")
r_c = terms["r_c"]
advantages = impl.group_advantage(r_c.view(2, 4)).view(-1)
for g in range(2):
    group = r_c[4 * g:4 * g + 4]
    print(f"    group {g + 1}: R_C {[f(r) for r in group]}  mean {f(group.mean())}, std {f(group.std())}"
          f"  ->  A^ {[f(a) for a in advantages[4 * g:4 * g + 4]]}")
optimizer = torch.optim.SGD(curriculum.parameters(), lr=1.0)
before = curriculum.logits.detach().clone()
old_logp = curriculum.log_prob(batch).detach()
loss = impl.clipped_loss(curriculum.log_prob(batch), old_logp, advantages, torch.full_like(advantages, 0.2))
optimizer.zero_grad()
loss.backward()
optimizer.step()
moved = curriculum.logits.detach() - before
print("\n  One gradient step on the clipped loss (Eq. 8, eps 0.2), SGD lr 1.0. The logits that moved:")
for question in sorted(set(batch.tolist())):
    name = "MALFORMED" if question == env.MALFORMED else f"question {question} (level {env.level(question)})"
    print(f"    {name:<24} logit 0.0 -> {f(curriculum.logits[question])}   ({f(moved[question]):+})")
print("""  Questions that beat their group (A^ > 0) go up; the rest go down. The logits of
  questions nobody wrote also shift a little, because softmax shares the probability.""")

print("\n  30 steps of this, 32 questions each (curriculum.train_curriculum). p(writing each level):")
torch.manual_seed(2)
curriculum = env.Curriculum()
impl.train_curriculum(curriculum, executor, torch.optim.SGD(curriculum.parameters(), lr=1.0))
writes = curriculum.by_level()
for lvl in range(1, env.LEVELS + 1):
    print(f"    level {lvl}: {writes[lvl - 1]:.2f}   (the Executor is right {executor.p_right(lvl):.2f} of the time)")
print(f"    MALFORMED: {writes[-1]:.2f}")
print("""  Nobody told it which level to write. It found the level where the Executor is right
  about half the time, because that is where R_unc pays.""")

# ---------------------------------------------------------------- step 4
banner("STEP 4  curate: keep 0.3 <= p^ <= 0.8, and the majority becomes the label (Eq. 7)")
torch.manual_seed(3)
written = curriculum.write(8).tolist()
print(f"  The trained Curriculum Agent writes 8 questions; the Executor answers each 10 times:\n")
for question in written:
    if question == env.MALFORMED:
        print(f"    MALFORMED       -> skipped")
        continue
    y_tilde, p_hat = impl.self_consistency(executor.answer(question, 10))
    verdict = "KEEP" if impl.keep(p_hat) else "drop"
    label = "right" if y_tilde == env.RIGHT else "WRONG"
    print(f"    question {question:>2} (level {env.level(question)}):  y~ = {y_tilde} ({label}),  p^ = {p_hat:.1f}  ->  {verdict}")
torch.manual_seed(4)
dataset = impl.curate(curriculum, executor)
wrong = [row for row in dataset if row["y_tilde"] != env.RIGHT]
print(f"""
  On a full batch of 64: {len(dataset)} kept, and {len(wrong)} of their labels are WRONG, all at p^ =
  {sorted({row['p_hat'] for row in wrong})}. Low agreement means an unreliable label. ADPO is built on that.""")

# ---------------------------------------------------------------- step 5
banner("STEP 5  ADPO moves the Executor's skill logit")
print("""      R_i   = 1(o_i = y~)                          the reward: agree with the LABEL, not the truth
      A^_i  = (R_i - mean) / std                   GRPO's advantage, within the question's 8 answers
      A~_i  = A^_i * s(x),  s(x) = f(p^)           ADPO: trust a low-agreement label less
      clip(r_i, 1 - 0.2, 1 + eps_high(x))          ADPO: a wider upper bound on hard questions""")
right_row = next(r for r in dataset if r["y_tilde"] == env.RIGHT and r["p_hat"] >= 0.5)
wrong_row = next(r for r in dataset if r["y_tilde"] != env.RIGHT and env.level(r["question"]) == 2)
for row, seed in ((right_row, 5), (wrong_row, 8)):
    torch.manual_seed(seed)
    levels = torch.tensor([env.level(row["question"])] * 8)
    answers = executor.dist(levels).sample()                               # 8 answers o_i
    rewards = impl.executor_reward(answers, torch.tensor(row["y_tilde"]))  # R_i = 1(o_i = y~)
    raw = impl.group_advantage(rewards.view(1, 8)).view(-1)                # A^_i
    scale, eps_high = impl.adpo_scale(row["p_hat"]), impl.adpo_eps_high(row["p_hat"])
    label = "right" if row["y_tilde"] == env.RIGHT else "WRONG: the right answer is 0"
    print(f"""
  Question {row['question']} (level {env.level(row['question'])}), label y~ = {row['y_tilde']} ({label}), p^ = {row['p_hat']:.1f}:
    s(x)        = 0.5 + 0.5 * clamp(({row['p_hat']:.1f} - 0.3) / 0.5, 0, 1) = {scale:.2f}
    eps_high(x) = 0.2 + 0.1 * clamp((0.8 - {row['p_hat']:.1f}) / 0.5, 0, 1) = {eps_high:.2f}
    8 answers  {answers.tolist()}
    R_i  = {[int(r) for r in rewards]}
    A^_i = {[f(a, 2) for a in raw]}
    A~_i = A^_i * {scale:.2f} = {[f(a * scale, 2) for a in raw]}""")
    for name, adpo in (("ADPO", True), ("GRPO", False)):
        learner = env.Executor()
        opt = torch.optim.SGD(learner.parameters(), lr=0.05)
        adv = raw * scale if adpo else raw
        eps = torch.full_like(adv, eps_high if adpo else 0.2)
        old = learner.dist(levels).log_prob(answers).detach()
        loss = impl.clipped_loss(learner.dist(levels).log_prob(answers), old, adv, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"    one step, {name}: skill {f(env.START_SKILL)} -> {f(learner.skill, 4)}")
print("""
  How the skill moves. All 9 wrong answers share one logit (0), so only the RIGHT answers in
  the group move the skill: it changes in proportion to the sum of their advantages.
  - Right label: the right answers are the rewarded ones (A > 0), so the skill goes UP.
  - Wrong label: the right answers disagree with it (A < 0), so the skill goes DOWN.
  - No right answer in the group, or a group that all agrees (A^ = 0): nothing moves.
  ADPO's step is s(x) times GRPO's. s(x) is smallest at low p^, where the wrong labels
  are, so it halves the damage there -- and also halves the lesson from right labels
  with the same low p^.""")

# ---------------------------------------------------------------- step 6
banner("STEP 6  the loop: Steps 3 -> 4 -> 5, three times (seed 0)")
print(f"  {'iter':>4} | p(writing level 1..5)           | kept | labels right | skill       | p(right), level 1..5")


def report(iteration, record):
    print(f"  {iteration:>4} | {' '.join(f'{w:.2f}' for w in record['writes'][:5])} | {record['kept']:>4} | "
          f"{record['labels_right']:>5} of {record['kept']:<4} | {record['skill_before']:.2f} -> {record['skill']:.2f} | "
          f"{' '.join(f'{p:.2f}' for p in record['p_right'])}")


impl.agent0(seed=0, report=report)
print("""  Each iteration, the Curriculum Agent writes where the Executor is right about half the
  time; the Executor trains on those questions and gets better at every level; so next
  time, that level is too easy and the Curriculum Agent moves up. Neither had any data.""")
