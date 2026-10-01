"""G-Zero, one step at a time: ``python GZero/steps_gzero.py``.

Follows one round through every equation, printing each one with the numbers
substituted into it, and the logits of both agents as they move. Set
``RL_IMPL=scratch`` to run the same walkthrough on your
GZero/from_scratch/gzero.py, and ``./scripts/run_gzero.sh diff`` to compare the
two outputs line by line.
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
import gzero as impl  # noqa: E402
import gzero_env as env  # noqa: E402

torch.set_num_threads(1)


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x, digits=3):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), digits) + 0.0               # + 0.0: no -0.0


def probs(logits):
    return [f(p, 2) for p in torch.softmax(logits.detach(), -1)]


HINT_NAME = {0: "position 0", 1: "position 1", 2: "position 2", 3: "all positions"}

# ---------------------------------------------------------------- step 0
banner("STEP 0  two agents, two tables of logits")
generator, proposer = env.Generator(), env.Proposer()
print(f"""  The Generator pi_G answers a query with 3 tokens (vocab 4), one per position. It is a
  table of logits [query, position, token]. Query 1's good answer is {env.GOOD[1].tolist()} (hidden: used
  only to measure), and position 1 is a blind spot -- pi_G prefers a wrong token there:
""")
for t in range(env.T):
    tag = "blind spot" if t in env.BLIND[1] else "known"
    print(f"    position {t} ({tag:<10}) logits {[f(x, 1) for x in generator.logits(1)[t]]}"
          f"  ->  p {probs(generator.logits(1)[t])}   p(good token {int(env.GOOD[1, t])}) = "
          f"{f(torch.softmax(generator.logits(1)[t], -1)[env.GOOD[1, t]], 2)}")
print(f"""
  A HINT names the good token at one position (or all three). Reading it adds kappa = {env.KAPPA}
  to that token's logit: logits(q, h) = logits(q) + kappa * onehot(position, good token).
  Query 1 with the hint on position 1:
""")
t = 1
print(f"    position 1 logits {[f(x, 1) for x in generator.logits(1, hint=1)[t]]}  ->  p {probs(generator.logits(1, hint=1)[t])}"
      f"   p(good) = {f(torch.softmax(generator.logits(1, hint=1)[t], -1)[env.GOOD[1, t]], 2)}")
print(f"""
  The Proposer pi_P picks a (query, hint) pair: 6 queries x 4 hints = 24 logits, all 0, so
  each has p = 1/24 = {1 / 24:.4f}. Blind spots per query: {env.BLIND}.""")

# ---------------------------------------------------------------- step 1
banner("STEP 1  Hint-delta: how much does the hint move the Generator's OWN answer? (Eq. 3)")
print("""      delta(q, h, a_hard) = (1/T) * sum_t [ log pi_G(a_t | q) - log pi_G(a_t | q, h) ]

  a_hard is the Generator's unassisted answer. Score it twice: without the hint, then with
  it. Query 1, two different hints:
""")
torch.manual_seed(0)
for h in (1, 0):
    a_hard = generator.answer(1)
    alone, hinted = generator.token_logps(1, a_hard), generator.token_logps(1, a_hard, hint=h)
    delta = impl.hint_delta(generator, 1, h, a_hard)
    print(f"  hint on {HINT_NAME[h]} ({'the blind spot' if h in env.BLIND[1] else 'a known position'}), a_hard = {a_hard.tolist()}:")
    print(f"    log pi_G(a_t | q)    {[f(x) for x in alone]}")
    print(f"    log pi_G(a_t | q, h) {[f(x) for x in hinted]}")
    terms = alone - hinted
    print(f"    delta = (1/3) * ({' + '.join(f'({f(x)})' for x in terms)}) = {f(delta)}\n")
print("""  The blind-spot hint makes the Generator's own wrong token less likely: delta > 0. The
  hint on a position it already gets right makes its (right) token MORE likely: delta < 0.
  Positions the hint does not name cancel exactly.

  On ONE answer delta can be negative; averaged over the Generator's answers it is the KL
  divergence from pi_G(. | q) to pi_G(. | q, h), never negative -- large where the hint
  changes the answer a lot:
""")
for q, h in ((1, 1), (1, 0), (5, 3), (0, 3)):
    p_alone, p_hinted = torch.softmax(generator.logits(q), -1), torch.softmax(generator.logits(q, hint=h), -1)
    kl = (p_alone * (p_alone.log() - p_hinted.log())).sum(-1).mean()
    n = len(env.BLIND[q])
    print(f"    query {q}, hint on {HINT_NAME[h]:<13}  average delta = {f(kl)}   (query {q} has {n} blind spot{'' if n == 1 else 's'})")

# ---------------------------------------------------------------- step 2
banner("STEP 2  the Proposer's reward (Eq. 4-5)")
print("""      P_length = 0.03 * max(0, (|h| - 200) / 100)      (Eq. 4)  one-position hints are 120 characters, the
                                                                  all-positions hint 360: it pays 0.048
      P_BLEU   = |C_i| / |B|                           copies of this output in the batch, over B
      r(q, h)  = delta - P_length - P_BLEU             (Eq. 5)

  One batch of B = 8 outputs. (query 5, all positions) is picked three times:
""")
torch.manual_seed(1)
batch = torch.tensor([5 * 4 + 3, 5 * 4 + 3, 5 * 4 + 3, 1 * 4 + 1, 1 * 4 + 0, 0 * 4 + 3, 3 * 4 + 2, 2 * 4 + 2])
terms = impl.score_outputs(batch, generator)
print(f"  {'q':>2} {'hint':<14} | {'delta':>6} | {'P_length':>8} | {'P_BLEU':>6} | r = delta - P_length - P_BLEU")
for i, output in enumerate(batch.tolist()):
    q, h = env.query_of(output), env.hint_of(output)
    d, pl, pb, r = (f(terms[key][i]) for key in ("delta", "p_length", "p_bleu", "r"))
    print(f"  {q:>2} {HINT_NAME[h]:<14} | {d:>6} | {pl:>8} | {pb:>6} | {d} - {pl} - {pb} = {r}")
print("""
  - Query 5 has three blind spots, so its all-positions hint moves the Generator most --
    but each of its three copies pays P_BLEU = 3/8, and the long hint pays P_length.
  - A hint on something the Generator knows (query 0) earns little, and can go negative.""")

# ---------------------------------------------------------------- step 3
banner("STEP 3  GRPO moves the Proposer's logits")
r = terms["r"]
advantages = impl.group_advantage(r.view(2, 4)).view(-1)
print("  2 groups of 4 (the paper uses K = 16). Each reward against its own group:\n")
for g in range(2):
    group = r[4 * g:4 * g + 4]
    print(f"    group {g + 1}: r {[f(x) for x in group]}  ->  A {[f(a) for a in advantages[4 * g:4 * g + 4]]}")
optimizer = torch.optim.SGD(proposer.parameters(), lr=1.0)
before = proposer.logits.detach().clone()
old_logp = proposer.log_prob(batch).detach()
loss = impl.clipped_loss(proposer.log_prob(batch), old_logp, advantages)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("\n  One step on the clipped loss (eps 0.2), SGD lr 1.0. The logits that moved:")
for output in sorted(set(batch.tolist())):
    q, h = env.query_of(output), env.hint_of(output)
    print(f"    query {q}, {HINT_NAME[h]:<14} logit 0.0 -> {f(proposer.logits[output])}   "
          f"({f(proposer.logits[output] - before[output]):+})")

print("\n  Phase 1 in full: 30 steps of 4 groups x 16. p(picking each query) and each hint kind:")
torch.manual_seed(2)
proposer = env.Proposer()
impl.train_proposer(proposer, generator, torch.optim.SGD(proposer.parameters(), lr=1.0))
for q, p in enumerate(proposer.by_query()):
    n = len(env.BLIND[q])
    print(f"    query {q}: {p:.2f}   ({n} blind spot{'' if n == 1 else 's'})")
print("    hint kind: " + "  ".join(f"{HINT_NAME[h]} {p:.2f}" for h, p in enumerate(proposer.by_hint())))
print("""  Nobody told it where the blind spots are. delta found them: the more blind spots a query
  has, the more its hints move the Generator. The long hint wins despite P_length, which
  at lambda = 0.03 charges it only 0.048.""")

# ---------------------------------------------------------------- step 4
banner("STEP 4  Phase 2's data: a pair per (q, h), then the lower 50% of delta")
print("""      chosen   y_w = a_assisted ~ pi_G(. | q, h)       the answer WITH the hint
      rejected y_l = a_hard     ~ pi_G(. | q)          the answer without it
""")
torch.manual_seed(3)
pairs = [impl.make_pair(generator, env.query_of(o), env.hint_of(o)) for o in proposer.write(8).tolist()]
kept = impl.lower_half(pairs)
for pair in sorted(pairs, key=lambda p: p["delta"]):
    q, h = pair["q"], pair["h"]
    good = env.GOOD[q].tolist()
    print(f"    query {q} ({HINT_NAME[h]:<13}) chosen {pair['chosen'].tolist()}  rejected {pair['rejected'].tolist()}"
          f"  good {good}  delta {f(pair['delta']):>6}  -> {'KEEP' if any(pair is k for k in kept) else 'drop'}")
print("""
  The lower half by delta is kept. The paper's reason: on an LLM, a very high-delta pair is
  far off the Generator's distribution and breaks DPO's implicit KL budget. In this toy the
  high-delta pairs are the ones where a_hard was furthest from what the hint pointed to --
  here the three query-5 pairs and the query-0 pair -- so the filter drops the pairs with
  the most to teach. The demo measures what that costs.""")

# ---------------------------------------------------------------- step 5
banner("STEP 5  length-normalised DPO moves the Generator's logits (Eq. 6)")
print("""      r_bar(x, y) = (1/|y|) * log( pi_theta(y | x) / pi_ref(y | x) )
      L           = -log sigmoid( beta * (r_bar(x, y_w) - r_bar(x, y_l)) ),     beta = 2.0

  x is the query ALONE: no hint. pi_ref is the Generator frozen at the round's start.
""")
pair = next(p for p in pairs if p["q"] == 5 and (p["chosen"] == env.GOOD[5]).sum() > (p["rejected"] == env.GOOD[5]).sum())
learner, reference = env.Generator(), env.Generator()
logp_w = learner.token_logps(pair["q"], pair["chosen"]).sum()
logp_l = learner.token_logps(pair["q"], pair["rejected"]).sum()
ref_w = reference.token_logps(pair["q"], pair["chosen"]).sum().detach()
ref_l = reference.token_logps(pair["q"], pair["rejected"]).sum().detach()
three = torch.tensor([3.0])
loss = impl.dpo_loss_ln(logp_w.view(1), ref_w.view(1), logp_l.view(1), ref_l.view(1), three, three)
print(f"  query 5, chosen {pair['chosen'].tolist()}, rejected {pair['rejected'].tolist()}, good {env.GOOD[5].tolist()}:")
print(f"    at the start pi_theta = pi_ref: r_bar = 0 for both, loss = -log sigmoid(0) = {f(loss)} (log 2)")
before = learner.p_good()[5]
optimizer = torch.optim.SGD(learner.parameters(), lr=2.0)
optimizer.zero_grad()
loss.backward()
optimizer.step()
after = learner.p_good()[5]
print(f"    one step, SGD lr 2.0. Query 5's unassisted p(good) by position: {[f(x, 3) for x in before]} -> {[f(x, 3) for x in after]}")
print("""  The chosen answer's tokens go up and the rejected answer's go down -- in the table the
  Generator uses WITHOUT a hint. Position 2 did not move: both answers had the same token
  there, so its push up and its push down cancel. That is "internalising" the hint: next time it answers
  query 5 alone, it leans the way the hint pointed.""")

# ---------------------------------------------------------------- step 6
banner("STEP 6  the loop: Phase 1 -> Phase 2, two rounds (seed 0)")


def report(rnd, record):
    print(f"  round {rnd}: Proposer p(query 0..5) {' '.join(f'{p:.2f}' for p in record['by_query'])}"
          f"   all-positions hint {record['by_hint'][3]:.2f}")
    print(f"           Generator p(good), mean over each query: "
          + " ".join(f"{float(x):.2f}" for x in record['p_good_before'].mean(1)) + "  ->  "
          + " ".join(f"{float(x):.2f}" for x in record['p_good'].mean(1))
          + f"   (overall {float(record['p_good'].mean()):.3f})")


print(f"  start: Generator p(good), mean over each query: "
      + " ".join(f"{float(x):.2f}" for x in env.Generator().p_good().mean(1)))
impl.gzero(seed=0, report=report)
print("""  Round 1 aims at query 5, the most blind spots; DPO fixes much of it; so in round 2 the
  Proposer shifts weight from query 5 to query 4, whose blind spots are now the bigger ones.
  Some queries slip back a little (query 3, 0.62 -> 0.56): DPO also pushes DOWN every
  rejected token, and the lower-50% filter keeps pairs whose rejected answer was partly
  right. No answer was ever checked: the only signal was the Generator's own probabilities.""")
