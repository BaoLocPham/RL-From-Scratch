"""Simple DPO, one step at a time: ``python SimpleDPO/steps_simple_dpo.py``.

Follows one pair of attempts at one question through every piece of DPO,
printing each equation with the numbers substituted into it. Set
``RL_IMPL=scratch`` to run the same walkthrough on your
SimpleDPO/from_scratch/simple_dpo.py, and ``./scripts/run_simple_dpo.sh diff``
to compare the two outputs line by line.
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
import simple_dpo as impl  # noqa: E402
from simple_dpo import dpo_loss, implicit_reward, sequence_logp  # noqa: E402
import pair_env as env  # noqa: E402


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 3) + 0.0                    # + 0.0: no -0.0


def row(t):
    return [f(v) for v in t]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def policy_at(p_search):
    """A policy with p(search) = p_search in every state."""
    policy = env.Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.tensor([[1 - p_search, p_search]] * 12).log())
    return policy


Q = env.QUALITY

# ---------------------------------------------------------------- step 1
banner("STEP 1  no reward: every pair is labelled, the better attempt is chosen")
print("""  SimpleGRPO graded every attempt right (+1) or wrong (-1). DPO never sees a number.
  It trains on LABELLED PAIRS: two attempts at one question, and a label saying which
  one is better. On an LLM, people (or a judge model) write those labels before
  training. The toy has no people, so pair_env.py writes them with the simplest rule:

      the attempt with the higher average total is CHOSEN, the other is REJECTED;
      two equally good attempts say nothing, so that pair is skipped.

  An attempt's average total depends only on its question type and its searches k:
""")
print(f"  {'':<5} {'k searches':>10} | {'average total':>13}")
for qtype, name in ((env.HARD, "HARD"), (env.EASY, "EASY")):
    for k in range(env.TURNS + 1):
        best = "   <- best" if (qtype, k) in ((env.HARD, 2), (env.EASY, 0)) else ""
        print(f"  {name if k == 0 else '':<5} {k:>10} | {float(Q[qtype, k]):>13.2f}{best}")
print("\n  Some labels:")
for label, a, b, verdict in (("HARD: two searches vs none ", Q[0, 2], Q[0, 0], "chosen = two searches"),
                             ("HARD: two searches vs three", Q[0, 2], Q[0, 3], "chosen = two searches"),
                             ("HARD: two searches vs two  ", Q[0, 2], Q[0, 2], "equally good: skipped"),
                             ("EASY: none vs one          ", Q[1, 0], Q[1, 1], "chosen = none")):
    print(f"    {label}  {f(a)} vs {f(b)}   ->  {verdict}")
print("""  The labels are all DPO gets. The averages themselves never reach training.""")

# ---------------------------------------------------------------- step 2
banner("STEP 2  one pair: two attempts at one HARD question")
states, actions, searches = env.all_attempts(env.HARD)
A, B = 6, 0                                            # (search, search, skip) and (skip, skip, skip)
print(f"""  attempt A: actions {actions[A].tolist()} (1 = SEARCH), states {states[A].tolist()}, {int(searches[A])} searches, average total {f(Q[0, 2])}
  attempt B: actions {actions[B].tolist()},                  states {states[B].tolist()}, {int(searches[B])} searches, average total {f(Q[0, 0])}

  {f(Q[0, 2])} > {f(Q[0, 0])}, so the label is:
      chosen = A, rejected = B
  A pair is (chosen, rejected). Which is which is the whole label.
  (The states are env.state_id(question type, turn, searches so far), as in SimplePPO.)""")
chosen_states, chosen_actions = states[A:A + 1], actions[A:A + 1]          # (1, 3): one pair
rejected_states, rejected_actions = states[B:B + 1], actions[B:B + 1]

# ---------------------------------------------------------------- step 3
banner("STEP 3  sequence_logp: the log-probability of a whole attempt")
reference = env.Policy()
print("""  DPO compares whole attempts, so it needs log pi(attempt). Each turn is a choice, and
  the attempt's probability is their product; in logs, a sum:

      log pi(a_0, a_1, a_2) = log pi(a_0|s_0) + log pi(a_1|s_1) + log pi(a_2|s_2)
""")
per_turn_a = reference.dist(chosen_states).log_prob(chosen_actions)[0]
per_turn_b = reference.dist(rejected_states).log_prob(rejected_actions)[0]
ref_chosen = sequence_logp(reference, chosen_states, chosen_actions)
ref_rejected = sequence_logp(reference, rejected_states, rejected_actions)
print(f"  pi_ref searches with p = 0.4 in every state:  log 0.4 = {math.log(0.4):.3f}, log 0.6 = {math.log(0.6):.3f}")
print(f"  A: {row(per_turn_a)} -> sum {f(ref_chosen[0])}      pi_ref(A) = 0.4 * 0.4 * 0.6 = {0.4 * 0.4 * 0.6:.3f}")
print(f"  B: {row(per_turn_b)} -> sum {f(ref_rejected[0])}      pi_ref(B) = 0.6 * 0.6 * 0.6 = {0.6 ** 3:.3f}")
print("""
  Summed, not averaged. A mean would divide by the number of turns: harmless here, where
  every attempt has 3, but on an LLM it would change what is being compared. verl's
  get_batch_logps sums the response's tokens the same way.""")

# ---------------------------------------------------------------- step 4
banner("STEP 4  where the loss comes from: the best policy has a closed form")
print("""  RLHF (and SimpleGRPO with its KL) maximises reward while staying near pi_ref:

      max_pi  E[r(attempt)] - beta * KL(pi || pi_ref)

  Its solution is known exactly (DPO eq. 4):

      pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z             Z: whatever makes it sum to 1

  Solve that for r:   r(y) = beta * log(pi*(y) / pi_ref(y)) + beta * log Z

  DPO's paper assumes the labels come from a judge who prefers the better attempt
  more often the bigger the gap (the Bradley-Terry model):

      p(A labelled better than B) = sigmoid(r(A) - r(B))

  Put r into that. Z is the same for both attempts at one question, so it cancels:

      p(A over B) = sigmoid(beta * log(pi*(A) / pi_ref(A)) - beta * log(pi*(B) / pi_ref(B)))

  No reward and no Z left: only the policy and pi_ref. So fit the POLICY to the
  labels directly, by maximum likelihood. That is DPO. (Our labeller always picks the
  better attempt: the most confident judge there is.)
""")
beta = 0.1
star = env.optimal_policy(beta)
star_a = sequence_logp(star, chosen_states, chosen_actions)
star_b = sequence_logp(star, rejected_states, rejected_actions)
reward_a = implicit_reward(star_a, ref_chosen, beta)
reward_b = implicit_reward(star_b, ref_rejected, beta)
print(f"  Check it on the toy. pi*_0.1, enumerated over all 8 attempts (env.optimal_policy):")
print(f"    log pi*(A) = {f(star_a[0])}   beta * log(pi*(A) / pi_ref(A)) = 0.1 * ({f(star_a[0])} - ({f(ref_chosen[0])})) = {f(reward_a[0])}")
print(f"    log pi*(B) = {f(star_b[0])}  beta * log(pi*(B) / pi_ref(B)) = 0.1 * ({f(star_b[0])} - ({f(ref_rejected[0])})) = {f(reward_b[0])}")
print(f"    difference {f(reward_a[0] - reward_b[0])}  =  r(A) - r(B) = 0.8 - (-1.0) = 1.8")
print("""  Each one is r plus the same unknown constant (beta * log Z), so their difference is
  exactly the reward difference. That is the paper's title: the policy's log-probs,
  measured against pi_ref, can stand in for a reward model.""")

# ---------------------------------------------------------------- step 5
banner("STEP 5  dpo_loss: logistic regression on the pair")
print("""  With the policy in the place of pi*, the chance it gives the label is
  sigmoid(beta * logits), and the loss is its negative log (DPO eq. 7):

      pi_logratios  = log pi(chosen)     - log pi(rejected)
      ref_logratios = log pi_ref(chosen) - log pi_ref(rejected)
      logits        = pi_logratios - ref_logratios
      loss          = -log sigmoid(beta * logits)
""")
for label, policy in (("at the start, pi = pi_ref       ", env.Policy()),
                      ("after moving to p(search) = 0.6", policy_at(0.6))):
    pc = sequence_logp(policy, chosen_states, chosen_actions)
    pr = sequence_logp(policy, rejected_states, rejected_actions)
    logits = (pc - pr) - (ref_chosen - ref_rejected)
    loss = dpo_loss(pc, pr, ref_chosen, ref_rejected, beta)
    print(f"  {label}")
    print(f"    pi_logratios  = {f(pc[0])} - ({f(pr[0])}) = {f(pc[0] - pr[0])}")
    print(f"    ref_logratios = {f(ref_chosen[0])} - ({f(ref_rejected[0])}) = {f(ref_chosen[0] - ref_rejected[0])}")
    print(f"    logits        = {f(pc[0] - pr[0])} - ({f(ref_chosen[0] - ref_rejected[0])}) = {f(logits[0])}")
    print(f"    loss          = -log sigmoid(0.1 * {f(logits[0])}) = -log {sigmoid(0.1 * f(logits[0])):.3f} = {f(loss)}")
print("""
  At pi = pi_ref the logits are 0 and the loss is log 2 = 0.693: a coin flip. Raising
  p(search) on HARD makes A likelier and B rarer, the logits positive, the loss lower.
  Note what is NOT here: no reward, no advantage, no ratio to theta_old, no clip. pi_ref
  is inside the loss, not a penalty added to it as SimpleGRPO's KL was.""")

# ---------------------------------------------------------------- step 6
banner("STEP 6  the gradient: push chosen up, rejected down, and weigh by how wrong")
print("""  Differentiate the loss (DPO §4, "What does the DPO update do?"):

      grad loss = -beta * sigmoid(-beta * logits) * [grad log pi(chosen) - grad log pi(rejected)]

  a policy gradient on two attempts, with +w on the chosen and -w on the rejected, where

      w = beta * sigmoid(-beta * logits)      large while the policy still gets the pair wrong,
                                              shrinking to 0 as it agrees with the label
""")
policy = env.Policy()
pc = sequence_logp(policy, chosen_states, chosen_actions)
pr = sequence_logp(policy, rejected_states, rejected_actions)
dpo_loss(pc, pr, ref_chosen, ref_rejected, beta).backward()
grad = policy.logits.grad
print(f"  At the start: w = 0.1 * sigmoid(0) = {0.1 * sigmoid(0):.3f}. By autograd, d loss / d logits in the states they visit:")
for state, note in ((0, "turn 0: A searched, B skipped"), (2, "turn 1, 1 search so far: A searched"),
                    (5, "turn 2, 2 so far: A skipped"), (1, "turn 1, 0 so far: B skipped"),
                    (3, "turn 2, 0 so far: B skipped")):
    print(f"    state {state}  [skip, search] {row(grad[state])}   {note}")
print("""  Descent subtracts these: SEARCH goes up where A searched and B skipped, SKIP goes up
  where A skipped. By hand for state 0: A adds -0.05 * ([0, 1] - [0.6, 0.4]) and B adds
  +0.05 * ([1, 0] - [0.6, 0.4]); together [0.05, -0.05].

  Compare SimpleGRPO with groups of 2. Graded right or wrong, A scores 0.8 and B -1.0, and
  Dr.GRPO gives them +0.9 and -0.9: HOW MUCH better A was. DPO's w = 0.05 is the same
  for any pair A won; it only knows WHICH one won, and it carries a factor beta. That
  factor is why the demo's SGD uses lr 0.3 / beta = 3.0 where SimpleGRPO used 0.3.""")

# ---------------------------------------------------------------- step 7
banner("STEP 7  beta: how hard to push, and when to stop")
print("""  beta multiplies the logits inside the loss, so it sets how much the policy must move
  before a pair counts as settled. The push on a pair is w = beta * sigmoid(-beta * logits):
""")
print(f"  {'logits':>8} | {'w at beta 0.1':>13} | {'w at beta 0.5':>13}")
for logits in (0, 5, 10, 20, 40):
    print(f"  {logits:>8} | {0.1 * sigmoid(-0.1 * logits):>13.4f} | {0.5 * sigmoid(-0.5 * logits):>13.4f}")
print("""  The push fades as the policy moves toward the chosen attempt: faster with a bigger
  beta, which keeps the policy closer to pi_ref. On an LLM, pi_ref is the model before
  training, and staying near it keeps what it already knew.

  The push never reaches exactly 0. Our labels never disagree (the better attempt always
  wins), so nothing ever pushes back, and the policy keeps moving, slowly, toward always
  choosing the better attempt. With real labellers, who sometimes disagree, the pairs
  that go the other way hold it back. (IPO, in DPO/, is a loss that stops on its own.)""")

# ---------------------------------------------------------------- step 8
banner("STEP 8  the loop: 60 iterations, pairs sampled from pi_ref and labelled before training")
print("""  for iteration:
      pairs = the next batch of labelled pairs               the published DPO: all from pi_ref
      for epoch in range(10):
          for minibatch of 4 pairs:
              loss = dpo_loss(...)                           steps 3 and 5
              zero_grad, backward, step
""")
curve, policy = env.train(impl, seed=0)
print(f"  seed 0: J {curve[0]:.3f} after 1 iteration -> {curve[9]:.3f} after 10 -> {curve[-1]:.3f} after 60."
      f"   best possible {env.BEST_J}")
print("\n  learned p(search):")
for row_text in env.describe(policy):
    print("  " + row_text)
print("""  HARD should search twice then stop; EASY should never search. Some states are rarely
  reached (EASY after a search), so their numbers barely matter.""")
learned = env.implicit_rewards(impl, policy, beta)
print("\n  how the policy now scores each number of searches, beta * log(pi / pi_ref), relative to 0:")
for qtype, name in ((env.HARD, "HARD"), (env.EASY, "EASY")):
    order = sorted(range(env.TURNS + 1), key=lambda k: -float(learned[qtype, k]))
    best_first = sorted(range(env.TURNS + 1), key=lambda k: -float(Q[qtype, k]))
    print(f"    {name}  learned {row(learned[qtype])}   best to worst: {order}"
          f"   (by average total: {best_first})")
print("""  Trained only on which attempt was better, the policy now ranks the attempts in the same
  order as their average totals: it has become a scorer of attempts. The numbers are larger
  than the averages and keep growing with training, because the labels never disagree.""")
