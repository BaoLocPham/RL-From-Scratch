"""Simple GRPO, one step at a time: ``python SimpleGRPO/steps_simple_grpo.py``.

Follows one group of attempts at one question through every piece of GRPO,
printing each equation with the numbers substituted into it. Set
``RL_IMPL=scratch`` to run the same walkthrough on your
SimpleGRPO/from_scratch/simple_grpo.py, and ``./scripts/run_simple_grpo.sh diff``
to compare the two outputs line by line.
"""

import itertools
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
from simple_grpo import group_advantage, grpo_update, kl_penalty, policy_loss  # noqa: E402
import group_env as env  # noqa: E402


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 3) + 0.0                    # + 0.0: no -0.0


def row(t):
    return [f(v) for v in t]


def value(qtype, searches):
    """E[R | question type, searches]: 2 * p(correct) - 1 for the answer, minus 0.1 per search."""
    return 2 * float(env.P_CORRECT[qtype, searches]) - 1 - env.SEARCH_COST * searches


def sequences(policy, qtype):
    """Every action sequence for one question type: (actions, probability, searches, E[R])."""
    probs = policy.logits.softmax(-1).detach()
    out = []
    for actions in itertools.product((env.SKIP, env.SEARCH), repeat=env.TURNS):
        chance, searches = 1.0, 0
        for turn, action in enumerate(actions):
            chance *= float(probs[env.state_id(qtype, turn, searches), action])   # pi(a_t | s_t)
            searches += action
        out.append((actions, chance, searches, value(qtype, searches)))
    return out


def j_of_type(policy, qtype):
    """E[R | question type]: the sum over its 8 action sequences, each weighted by its probability."""
    return sum(chance * v for _, chance, _, v in sequences(policy, qtype))


# ---------------------------------------------------------------- step 1
banner("STEP 1  the goal: J(theta), the average total reward of one attempt")
print("""  One attempt: a question (HARD or EASY, 50/50), three choices (SEARCH or SKIP),
  then the answer. With k = the number of searches, its total reward is

      R = (+1 if the answer is right, -1 if it is wrong) - 0.1 * k

  The chance p of being right depends on the question and on k, so the AVERAGE reward
  of an attempt that searched k times is

      E[R | q, k] = (+1) * p + (-1) * (1 - p) - 0.1 * k = 2p - 1 - 0.1 * k
""")
print(f"  {'':<5} {'k searches':>10} | {'p(right)':>8} | {'2p - 1':>6} | {'- 0.1 k':>7} | {'E[R | q, k]':>11}")
for qtype, name in ((env.HARD, "HARD"), (env.EASY, "EASY")):
    for k in range(env.TURNS + 1):
        p = float(env.P_CORRECT[qtype, k])
        best = "   <- best" if (qtype, k) in ((env.HARD, 2), (env.EASY, 0)) else ""
        print(f"  {name if k == 0 else '':<5} {k:>10} | {p:>8.2f} | {2 * p - 1:>6.2f} | {-0.1 * k + 0.0:>7.1f} | "
              f"{value(qtype, k):>11.2f}{best}")
print("""  (HARD with one search is the coin flip: +1 - 0.1 = +0.9 or -1 - 0.1 = -1.1, on average -0.1.)

  J averages that over everything random: the question type, and the policy's three choices.
  Each choice is made in the state s_t = (question type, turn t, searches so far):

      J(theta) = sum over q of 0.5 * sum over (a0, a1, a2) of
                 pi(a0 | s0) * pi(a1 | s1) * pi(a2 | s2) * E[R | q, k = a0 + a1 + a2]

  2 question types x 2^3 action sequences = 16 terms. At the start, p(search) = 0.4 in every
  state. The 8 terms for HARD:""")
start = env.Policy()
names = {env.SKIP: "SKIP", env.SEARCH: "SEARCH"}
print(f"    {'actions':<22} | {'probability':<24} | k | {'E[R | HARD, k]':>14} | {'product':>7}")
for actions, chance, k, v in sequences(start, env.HARD):
    factors = " x ".join("0.4" if a == env.SEARCH else "0.6" for a in actions)
    print(f"    {' '.join(names[a] for a in actions):<22} | {factors} = {chance:.3f} | {k} | {v:>14.2f} | "
          f"{chance * v:>+7.4f}")
hard, easy = j_of_type(start, env.HARD), j_of_type(start, env.EASY)
print(f"    {'':<22}   {'':<24}   {'':<1}   {'E[R | HARD] =':>14} {hard:>+7.4f}")
print(f"""  The same 8 terms for EASY give E[R | EASY] = {easy:.4f}.
      J = 0.5 * {hard:.4f} + 0.5 * {easy:.4f} = {0.5 * hard + 0.5 * easy:.3f}      env.true_reward: {env.true_reward(start):.3f}
  The best policy searches twice on HARD (0.8) and never on EASY (1.0):
      J* = 0.5 * 0.8 + 0.5 * 1.0 = {env.BEST_J}
  Training never sees J: it sees single rewards, each a coin flip. J is computed exactly
  here only because the toy is tiny (env.true_reward is this sum); on an LLM it can only
  be estimated, by sampling. Every "J" printed below and in run_simple_grpo.py is this sum.""")


# ---------------------------------------------------------------- step 2
banner("STEP 2  one question, answered four times: a group")
print("""  A HARD question (it needs two searches). The policy attempts it 4 times. Each search
  costs 0.1; the answer is graded right (+1) or wrong (-1), like a math checker.

  attempt | turn 0  turn 1  turn 2 | searches | answer | rewards per turn   | total R
     1    | SEARCH  SEARCH  SKIP   |    2     | right  | -0.1  -0.1   1.0   |   0.8
     2    | SKIP    SKIP    SKIP   |    0     | wrong  |  0.0   0.0  -1.0   |  -1.0
     3    | SEARCH  SKIP    SKIP   |    1     | wrong  | -0.1   0.0  -1.0   |  -1.1   (one search: a coin flip, lost)
     4    | SKIP    SEARCH  SEARCH |    2     | right  |  0.0  -0.1   0.9   |   0.8

  SimplePPO asked a critic "how good is this state?" at every turn. GRPO has no critic:
  it asks how each attempt did compared with the OTHER attempts at the same question.""")
rewards = torch.tensor([[-0.1, -0.1, 1.0], [0.0, 0.0, -1.0], [-0.1, 0.0, -1.0], [0.0, -0.1, 0.9]])

# ---------------------------------------------------------------- step 3
banner("STEP 3  the group advantage: A_i = (R_i - mean(R)) / std(R), one line of code at a time")
print("  group_advantage(rewards, group_size=4) does five things. The numbers for step 2's group:")

print("\n  (a) scores = rewards.sum(1): add each attempt's rewards into ONE number, its total R")
totals = rewards.sum(1)
for i in range(4):
    terms = " + ".join(f"({float(r):+.1f})" for r in rewards[i])
    print(f"      attempt {i + 1}: {terms} = {float(totals[i]):+.1f}")
print("      GRPO never looks at the turns again: outcome supervision, one grade per attempt.")

print("\n  (b) groups = scores.view(-1, group_size): one row per question, one column per attempt")
print(f"      {tuple(totals.shape)} -> {tuple(totals.view(-1, 4).shape)}: [{', '.join(f'{float(r):+.1f}' for r in totals)}]")
print("      One question here; a batch of 2 questions x 8 attempts would be (16,) -> (2, 8), and each")
print("      row would get its own mean and std in (c) and (d).")

print("\n  (c) centred = groups - groups.mean(1, keepdim=True): better or worse than THIS question's average")
mean = float(totals.mean())
print(f"      mean = ({' + '.join(f'({float(r):+.1f})' for r in totals)}) / 4 = {float(totals.sum()):+.1f} / 4 = {mean:+.3f}")
deviations = totals - mean
for i in range(4):
    print(f"      attempt {i + 1}: {float(totals[i]):+.1f} - ({mean:+.3f}) = {float(deviations[i]):+.3f}")
print("      The mean plays the part of SimplePPO's critic V(s_0): how an average attempt at this")
print("      question goes, estimated from the attempts themselves.")
print("      Dr.GRPO stops here: its advantages are these centred rewards.")

print("\n  (d) centred / (groups.std(1, keepdim=True) + eps): put every question on the same scale")
squares = deviations ** 2
print("      std = sqrt( sum of (R - mean)^2 / (n - 1) ), with n = 4 attempts:")
print("      squares: " + ",  ".join(f"({float(d):+.3f})^2 = {float(q):.4f}" for d, q in zip(deviations, squares)))
variance = float(squares.sum()) / 3
std = variance ** 0.5
print(f"      sum = {float(squares.sum()):.4f};   / (4 - 1) = {variance:.4f};   sqrt = {std:.4f}")
print(f"      n - 1, not n: torch.std's default, the sample std (dividing by 4 would give "
      f"{float(totals.std(unbiased=False)):.4f}).")
print("      eps = 1e-6 is added to the std only so a group with std 0 (step 6) does not divide by 0.")
for i in range(4):
    print(f"      A_{i + 1} = {float(deviations[i]):+.3f} / {std:.4f} = {float(deviations[i]) / std:+.3f}")
scaled = deviations / std
print(f"      check: the A's sum to {f(scaled.sum()):.3f} and their std is {f(scaled.std()):.3f}. Every group")
print("      comes out with mean 0 and std 1, whatever its rewards were.")

print("\n  (e) centred.reshape(-1, 1).expand_as(rewards): the same A_i on every turn of attempt i")
print(f"      {tuple(scaled.shape)} -> reshape(-1, 1) -> (4, 1), one column -> expand_as(rewards) -> "
      f"{tuple(rewards.shape)}:")
print("      the column is copied across the 3 turns.")
advantages = group_advantage(rewards, 4)
print(f"  group_advantage -> (attempts, turns) =")
for i in range(4):
    print(f"    attempt {i + 1}: {row(advantages[i])}")
print("""  Every turn of an attempt carries the same number. Attempt 1's final SKIP gets +0.865
  just like its two searches: the grade says how the attempt went, not which turn did it.
  SimplePPO's GAE gave the turns of one episode different advantages ([0.208, 0.26, 0.2]);
  here there is nothing to tell them apart. Over many attempts the credit still sorts
  itself out: turns that help show up more often in the attempts that did well.""")

# ---------------------------------------------------------------- step 4
banner("STEP 4  why a group, and not the whole batch")
easy = torch.tensor([[0.0, 0.0, 1.0], [-0.1, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, -0.1, 1.0]])
both = torch.cat([rewards, easy])
print(f"  add an EASY question's group: totals {row(easy.sum(1))}   (two wasted a search, 0.9)")
batch = both.sum(1) - both.sum(1).mean()
grouped = group_advantage(both, 4, scale_by_std=False)[:, 0]
print(f"  R - the batch's mean ({f(both.sum(1).mean())}):  HARD {row(batch[:4])}   EASY {row(batch[4:])}")
print(f"  R - its group's mean:              HARD {row(grouped[:4])}   EASY {row(grouped[4:])}")
print("""  Against the batch, every EASY attempt looks good -- even the ones that wasted a search
  -- only because EASY questions are easy. Against its own group, a wasted search is
  below average. The group mean is a baseline for THIS question: what a critic's V(s_0)
  would estimate, taken from the attempts themselves.""")

# ---------------------------------------------------------------- step 5
banner("STEP 5  the std divide, and Dr.GRPO (scale_by_std=False)")
pairs = torch.tensor([[-0.1, 0.0, 1.0], [-0.1, -0.1, 1.0], [-0.1, 0.0, -1.0], [-0.1, -0.1, 1.0]])
print("  groups of two, on a HARD question: one search vs two searches")
print(f"    lucky coin: 1 search right (0.9) vs 2 searches (0.8)  GRPO {row(group_advantage(pairs[:2], 2)[:, 0])}"
      f"   Dr.GRPO {row(group_advantage(pairs[:2], 2, scale_by_std=False)[:, 0])}")
print(f"    lost coin:  1 search wrong (-1.1) vs 2 searches (0.8) GRPO {row(group_advantage(pairs[2:], 2)[:, 0])}"
      f"   Dr.GRPO {row(group_advantage(pairs[2:], 2, scale_by_std=False)[:, 0])}")
print("""  With two attempts, (R - mean) / std is always +-0.707, however far apart they are. The
  coin lands either way half the time, so GRPO's pushes between one and two searches
  cancel -- though two searches are worth 0.9 more. Dr.GRPO keeps the gap's size.""")
nearly = torch.tensor([[0.0, 0.0, 1.0]] * 3 + [[-0.1, 0.0, 1.0]])
print(f"  a group that nearly agrees, totals {row(nearly.sum(1))}:")
print(f"    GRPO {row(group_advantage(nearly, 4)[:, 0])}   Dr.GRPO "
      f"{row(group_advantage(nearly, 4, scale_by_std=False)[:, 0])}")
print("  The divide blows a 0.1 difference up to full size, and keeps pushing a nearly settled")
print("  question as hard as an open one.")

# ---------------------------------------------------------------- step 6
banner("STEP 6  a dead group: every attempt scored the same")
solved = torch.tensor([[0.0, 0.0, 1.0]] * 4)
print(f"  totals {row(solved.sum(1))}: mean 1.0, std 0 -> A = 0 / (0 + eps) = {row(group_advantage(solved, 4)[:, 0])}")
print("""  No attempt was better than another, so the group teaches nothing: its whole gradient is
  zero. Once a question is solved every group of it is dead; DAPO's dynamic sampling
  throws such groups away and samples new questions. run_simple_grpo.py counts them.""")

# ---------------------------------------------------------------- step 7
banner("STEP 7  the loss: SimplePPO's clip, plus beta * KL(pi_theta || pi_ref)")
print("  k3, DeepSeekMath's eq. 4, on the action taken: x - log x - 1, with x = pi_ref / pi_theta")
probe = env.Policy()
with torch.no_grad():
    probe.logits.copy_(torch.tensor([[0.4, 0.6]] * 12).log())
ref_probs = torch.tensor([0.6, 0.4])                  # pi_ref: SKIP 0.6, SEARCH 0.4 (the starting policy)
k_search = kl_penalty(probe, torch.tensor([0]), torch.tensor([env.SEARCH]), ref_probs[env.SEARCH:].log())
k_skip = kl_penalty(probe, torch.tensor([0]), torch.tensor([env.SKIP]), ref_probs[:env.SEARCH].log())
print(f"  pi_theta(search) 0.6, pi_ref(search) 0.4:")
k_search, k_skip = float(k_search.detach()), float(k_skip.detach())
print(f"    took SEARCH: x = 0.4/0.6 -> {k_search:.4f}      took SKIP: x = 0.6/0.4 -> {k_skip:.4f}")
exact = 0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.6)
print(f"    averaged by pi_theta: 0.6 * {k_search:.4f} + 0.4 * {k_skip:.4f} = "
      f"{0.6 * k_search + 0.4 * k_skip:.4f};  exact KL = {exact:.4f}")
print("  One action per step is all an LLM can afford to look at (TRPO's sum is over ~150k tokens).")

torch.manual_seed(0)
policy, ref_policy = env.Policy(), env.Policy()
batch = env.rollout(policy, ref_policy, 2, 8)
batch["advantages"] = group_advantage(batch["rewards"], 8)
steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "ref_logp", "advantages")}
pg = policy_loss(policy, steps["states"], steps["actions"], steps["old_logp"], steps["advantages"])
kl = kl_penalty(policy, steps["states"], steps["actions"], steps["ref_logp"])
print(f"\n  a batch of 2 questions x 8 attempts = {steps['states'].numel()} steps, at theta_old = pi_ref:")
print(f"  pg = -L^CLIP = {f(pg):+.3f}   (ratio 1: -mean(A), and each group's A sums to 0)")
print(f"  kl           = {f(kl):.3f}    (the policy has not moved from pi_ref yet)")
optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)
metrics = grpo_update(policy, optimizer, batch)
print(f"  grpo_update: 10 epochs x 3 minibatches of 16 = 30 steps, mean kl {metrics['kl']:.4f}. "
      f"J {env.true_reward(env.Policy()):.3f} -> {env.true_reward(policy):.3f}")
print("  No value loss, no entropy bonus, no critic to train: one network, one loss.")

# ---------------------------------------------------------------- step 8
banner("STEP 8  the whole loop: 60 iterations of 2 questions x 8 attempts (seed 0)")
import simple_grpo as impl  # noqa: E402  (whichever implementation was imported above)
curve, policy, dead = env.train(impl, seed=0)
print(f"  true J: {curve[0]:.3f} after 1, {curve[9]:.3f} after 10, {curve[29]:.3f} after 30, "
      f"{curve[-1]:.3f} after 60.   best possible {env.BEST_J}")
hard, easy = j_of_type(policy, env.HARD), j_of_type(policy, env.EASY)
print(f"  step 1's sum on the learned policy: J = 0.5 * {hard:.3f} (HARD, best 0.8) + 0.5 * {easy:.3f} "
      f"(EASY, best 1.0) = {0.5 * hard + 0.5 * easy:.3f}")
print("  learned p(search):")
for line in env.describe(policy):
    print("  " + line)
print(f"  dead groups: {sum(dead[:10]) / 10:.2f} of the first ten batches, {sum(dead[-10:]) / 10:.2f} of the last ten")
print("  ./scripts/run_simple_grpo.sh run compares GRPO with Dr.GRPO, smaller groups, no KL, and PPO.")
