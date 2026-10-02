"""G-Zero from scratch: the paper's equations on the toy. Fill the TODOs, then run ``check.py``.

How to work through it:
  1. Fill the lines marked TODO stage 1 (each TODO is one term of one equation).
  2. Try it:    python GZero/from_scratch/gzero.py    prints your results next to the expected ones
  3. Check it:  ./scripts/run_gzero.sh check           stops at the first stage that is not right yet
  4. Move on to the next stage.

    stage 1  hint_delta                                       Hint-delta                    (Eq. 3)
    stage 2  length_penalty, repetition_penalty,              the Proposer's reward r       (Eq. 4-5)
             proposer_reward
    stage 3  make_pair, lower_half                            the Generator's DPO data
    stage 4  dpo_loss_ln, group_advantage, clipped_loss       the two updates               (Eq. 6, GRPO)
    stage 5  no code                                          the loop runs on your pieces

The loops (score_outputs, train_proposer, collect_pairs, train_generator,
gzero) are given: read them, they are the algorithm. Try not to open
../gzero.py (the reference).

G-Zero (https://arxiv.org/abs/2605.09959) improves a model with no data and no
verifier. Two agents, two phases per round, 2 rounds in the paper:

    Phase 1  train the Proposer pi_P (GRPO); the Generator pi_G is frozen
             pi_P picks a query q and a hint h; pi_G answers q WITHOUT the hint (a_hard);
             the reward is how much the hint would have changed that answer: Hint-delta (Eq. 3-5)
    Phase 2  train the Generator pi_G (DPO); the Proposer is frozen
             pi_P picks N (q, h); pi_G answers with the hint (a_assisted) and without (a_hard);
             keep the lower 50% of delta; DPO with a_assisted chosen, a_hard rejected,
             and the prompt q ALONE -- so pi_G learns to answer as if it had the hint (Eq. 6)

No answer is ever checked. The only signal is the Generator's own probabilities,
with and without a hint. The toy -- the queries and both agents -- is in gzero_env.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gzero_env import HINT_CHARS, T, Generator, Proposer, hint_of, query_of  # noqa: E402

# ================================================================ the Proposer's reward


@torch.no_grad()
def hint_delta(generator, q, h, a_hard):
    """Hint-delta: how much the hint moves the Generator away from its own answer (Eq. 3).

        delta(q, h, a_hard) = (1/T) * sum_t [ log pi_G(a_t | q, a_<t) - log pi_G(a_t | q, h, a_<t) ]

    q:       a query id (int)
    h:       a hint id (int): 0, 1, 2 for one position, 3 for all of them
    a_hard:  (T,) token ids, the Generator's unassisted answer to q
    Returns delta, one float.

    a_hard is scored twice: without the hint, then with it. delta > 0 when the
    hint makes that answer LESS likely -- the hint points somewhere the
    Generator was not going. Averaged per token (1/T), so a longer answer earns
    no more. It is a reward, so no gradient flows through it.
    """
    logp_alone = ...                                # TODO stage 1: log pi_G(a_t | q, a_<t), (T,)
    logp_hinted = ...                               # TODO stage 1: log pi_G(a_t | q, h, a_<t), (T,)
    return ...                                      # TODO stage 1: delta = (1/T) * sum_t [ ... ]: (T,) -> one float


def length_penalty(hint_chars, lam=0.03):
    """Penalty for a long hint (Eq. 4).

        P_length = lambda * max(0, (|h| - 200) / 100),     lambda = 0.03

    Free up to 200 characters, then 0.03 per extra 100. A hint that spells out
    everything would move the Generator most; this stops it being free.

    hint_chars: |h|, the hint's length in characters (int). Returns P_length, one float.
    """
    return ...                                      # TODO stage 2: P_length


def repetition_penalty(outputs):
    """Penalty for picking what the rest of the batch also picked.

        P_BLEU(x_i) = |C_i| / |B|

    The paper clusters the batch's questions by BLEU similarity; C_i is the
    cluster x_i falls in. Here an output is only a (query, hint) id, so a
    cluster is every copy of the same output.

    outputs: (B,) the batch's Proposer output ids. Returns (B,): each output's P_BLEU.
    """
    cluster_size = ...                              # TODO stage 2: |C_i|: copies of this output in the batch, (B,)
    return ...                                      # TODO stage 2: |C_i| / |B|, (B,)


def proposer_reward(delta, p_length, p_bleu):
    """The Proposer's reward (Eq. 5).

        r(q, h) = delta(q, h, a_hard) - P_length - P_BLEU

    delta, p_length, p_bleu: each (B,). Returns r, (B,).

    No max(0, .): a hint about something the Generator already knows can earn
    less than nothing.
    """
    return ...                                      # TODO stage 2: r = delta - P_length - P_BLEU, (B,)


# ================================================================ the Generator's data and loss


def make_pair(generator, q, h):
    """One DPO pair for query q and hint h.

        chosen   y_w = a_assisted ~ pi_G(. | q, h)      the answer WITH the hint
        rejected y_l = a_hard     ~ pi_G(. | q)         the answer without it

    q: a query id (int). h: a hint id (int). Returns a dict:
        q, h       the inputs
        chosen     (T,) token ids, a_assisted
        rejected   (T,) token ids, a_hard
        delta      one float: Eq. 3 on this a_hard, for the filter
    """
    a_assisted = ...                                # TODO stage 3: y_w ~ pi_G(. | q, h), (T,)
    a_hard = ...                                    # TODO stage 3: y_l ~ pi_G(. | q), (T,)
    return {"q": q, "h": h, "chosen": a_assisted, "rejected": a_hard,
            "delta": hint_delta(generator, q, h, a_hard)}


def lower_half(pairs):
    """Keep the pairs in the lower 50% of delta.

    The paper's reason: on an LLM, a very high-delta pair is far off the
    Generator's distribution and breaks DPO's implicit KL budget. Here it is
    applied as the paper does it; the demo shows what it costs on this toy.

    pairs: a list of make_pair dicts. Returns a list: the half with the lowest delta.
    """
    ranked = ...                                    # TODO stage 3: lowest delta first
    return ...                                      # TODO stage 3: the lower 50%


def dpo_loss_ln(logp_w, ref_logp_w, logp_l, ref_logp_l, len_w, len_l, beta=2.0):
    """Length-normalised DPO (Eq. 6), averaged over the pairs.

        L = -log sigmoid( beta * (r_bar(x, y_w) - r_bar(x, y_l)) ),
        r_bar(x, y) = (1/|y|) * log( pi_theta(y | x) / pi_ref(y | x) ),     beta = 2.0

    logp_w, logp_l:          (n,) each answer's summed log-probability under the
                             Generator being trained, log pi_theta(y | x)
    ref_logp_w, ref_logp_l:  (n,) the same under pi_ref, the Generator frozen at
                             the start of the round, log pi_ref(y | x)
    len_w, len_l:            (n,) each answer's length |y|
    Returns the loss, a scalar (): the mean over the n pairs.

    The prompt x is the query ALONE: no hint.
    """
    r_bar_w = ...                                   # TODO stage 4: r_bar(x, y_w): per-token log-ratio of the chosen, (n,)
    r_bar_l = ...                                   # TODO stage 4: r_bar(x, y_l): the same for the rejected, (n,)
    return ...                                      # TODO stage 4: -log sigmoid(beta * (r_bar_w - r_bar_l)), mean over the n pairs: ()


# ================================================================ GRPO, for the Proposer


def group_advantage(rewards, eps=1e-6):
    """GRPO's advantage: each output against the others in its group of K.

        A_i = (r_i - mean(r_group)) / std(r_group)

    rewards: (groups, K). Returns (groups, K).
    """
    mean = ...                                      # TODO stage 4: mean(r_group): (groups, 1)
    std = ...                                       # TODO stage 4: std(r_group), sample std: (groups, 1)
    return ...                                      # TODO stage 4: A_i, (groups, K)


def clipped_loss(logp, old_logp, advantages, eps=0.2):
    """PPO's clipped loss, as GRPO uses it (negated, to minimise).

        L = -(1/K) * sum_i min( r_i * A_i, clip(r_i, 1 - eps, 1 + eps) * A_i ),    r_i = pi_P / pi_P_old

    logp:        (n,) log pi_P of each output, under the Proposer being trained
    old_logp:    (n,) log pi_P_old, the same before this update (no gradient)
    advantages:  (n,) A_i
    Returns the loss, a scalar ().
    """
    ratio = ...                                     # TODO stage 4: r_i = pi_P / pi_P_old, (n,)
    clipped = ...                                   # TODO stage 4: clip(r_i, 1 - eps, 1 + eps), (n,)
    return ...                                      # TODO stage 4: -(1/K) sum_i min(...), mean: ()


# ================================================================ the loop


@torch.no_grad()
def score_outputs(outputs, generator, use_bleu=True):
    """Phase 1's reward for a batch of Proposer outputs: every term of Eq. 5.

    outputs: (B,) Proposer output ids. Returns a dict of four (B,) tensors:
    delta, p_length, p_bleu and r. For each (q, h), the frozen Generator answers
    q once without the hint, and hint_delta scores that answer with and without h.
    """
    delta = torch.tensor([hint_delta(generator, query_of(o), hint_of(o), generator.answer(query_of(o)))
                          for o in outputs.tolist()])                              # delta(q, h, a_hard), (B,)   (Eq. 3)
    p_length = torch.tensor([length_penalty(HINT_CHARS[hint_of(o)]) for o in outputs.tolist()])   # P_length, (B,)   (Eq. 4)
    p_bleu = repetition_penalty(outputs) if use_bleu else torch.zeros(len(outputs))     # P_BLEU, (B,)
    return {"delta": delta, "p_length": p_length, "p_bleu": p_bleu,
            "r": proposer_reward(delta, p_length, p_bleu)}                             # r, (B,)   (Eq. 5)


def train_proposer(proposer, generator, optimizer, steps=30, groups=4, k=16, epochs=2, use_bleu=True):
    """Phase 1: GRPO on the Proposer's 24 logits. The Generator is frozen.

    Each step writes 4 groups of K = 16 outputs (the paper's K), so a batch of
    B = 64, and GRPO compares each output with the others in its group.
    """
    for _ in range(steps):
        with torch.no_grad():
            outputs = proposer.write(groups * k)                              # (q, h) ~ pi_P, (B,)
            r = score_outputs(outputs, generator, use_bleu)["r"]              # r(q, h), (B,)
            advantages = group_advantage(r.view(groups, k)).view(-1)          # A_i within each group of K: (groups, K) -> (B,)
            old_logp = proposer.log_prob(outputs)                             # log pi_P_old, frozen, (B,)
        for _ in range(epochs):
            loss = clipped_loss(proposer.log_prob(outputs), old_logp, advantages)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def collect_pairs(proposer, generator, n=200, use_filter=True):
    """Phase 2's data: N (q, h) from the frozen Proposer, a DPO pair for each, then the lower-50% filter.

    Returns a list of make_pair dicts: N // 2 of them with the filter, N without.
    """
    pairs = [make_pair(generator, query_of(o), hint_of(o)) for o in proposer.write(n).tolist()]
    return lower_half(pairs) if use_filter else pairs


def train_generator(generator, pairs, steps=50, batch=8, beta=2.0, lr=2.0):
    """Phase 2: length-normalised DPO on the Generator's unassisted table. The Proposer is frozen.

    pairs: a list of make_pair dicts; each step draws `batch` of them.
    pi_ref is a frozen copy of the Generator from the start of the round. The
    paper runs 50 steps with batch size 8; so does this.
    """
    if not pairs:
        return
    reference = Generator()
    reference.load_state_dict(generator.state_dict())                        # pi_ref: frozen at the round's start
    optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    for _ in range(steps):
        chosen = [pairs[i] for i in torch.randint(len(pairs), (batch,)).tolist()]
        logp_w = torch.stack([generator.token_logps(p["q"], p["chosen"]).sum() for p in chosen])      # log pi_theta(y_w | q), (batch,)
        logp_l = torch.stack([generator.token_logps(p["q"], p["rejected"]).sum() for p in chosen])    # log pi_theta(y_l | q), (batch,)
        with torch.no_grad():
            ref_w = torch.stack([reference.token_logps(p["q"], p["chosen"]).sum() for p in chosen])   # log pi_ref(y_w | q), (batch,)
            ref_l = torch.stack([reference.token_logps(p["q"], p["rejected"]).sum() for p in chosen])  # log pi_ref(y_l | q), (batch,)
        length = torch.full((batch,), float(T))                                                         # |y| = 3 tokens, (batch,)
        loss = dpo_loss_ln(logp_w, ref_w, logp_l, ref_l, length, length, beta)                          # Eq. 6: the loss, ()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def gzero(seed=0, rounds=2, use_filter=True, use_bleu=True, report=None):
    """The whole loop: Phase 1 then Phase 2, `rounds` times (the paper's 2).

    Returns (proposer, generator, history): the trained agents, and one record
    per round in history, a dict of
        by_query       6 floats: pi_P's p(query 0..5), summed over hints
        by_hint        4 floats: pi_P's p(hint 0..3), summed over queries
        top_output     one float: pi_P's largest p on any one (query, hint)
        pairs          an int: how many DPO pairs Phase 2 used
        p_good_before  (6, 3): pi_G's unassisted p(good token), per query and position, before DPO
        p_good         (6, 3): the same, after DPO
    `report(round, record)`, if given, is called after each round.
    """
    torch.manual_seed(seed)
    proposer, generator = Proposer(), Generator()
    proposer_optimizer = torch.optim.SGD(proposer.parameters(), lr=1.0)
    history = []
    for rnd in range(1, rounds + 1):
        train_proposer(proposer, generator, proposer_optimizer, use_bleu=use_bleu)    # Phase 1
        by_query, by_hint = proposer.by_query(), proposer.by_hint()
        top = float(proposer.logits.detach().softmax(-1).max())
        pairs = collect_pairs(proposer, generator, use_filter=use_filter)              # Phase 2: the data
        before = generator.p_good()
        train_generator(generator, pairs)                                              # Phase 2: DPO
        record = {"by_query": by_query, "by_hint": by_hint, "top_output": top, "pairs": len(pairs),
                  "p_good_before": before, "p_good": generator.p_good()}
        history.append(record)
        if report:
            report(rnd, record)
    return proposer, generator, history


# ================================================================ playground
def _show(label, fn, expected):
    """Run one of your functions and print it next to the expected value."""
    try:
        got = fn()
    except Exception as exc:                           # unfinished TODOs land here
        got = f"not done yet ({type(exc).__name__})"
    if got is Ellipsis:
        got = "not done yet"
    elif isinstance(got, torch.Tensor):
        got = [round(x, 3) + 0.0 for x in got.detach().flatten().tolist()]
        got = got[0] if len(got) == 1 else got
    elif isinstance(got, float):
        got = round(got, 3) + 0.0
    print(f"  {label:<40} yours: {str(got):<30} expected: {expected}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    generator = Generator()
    print("Your functions on worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  delta, blind-spot hint", lambda: hint_delta(generator, 1, 1, torch.tensor([1, 3, 3])), 0.422)
    _show("stage 1  delta, known-position hint", lambda: hint_delta(generator, 1, 0, torch.tensor([1, 1, 3])), -0.044)
    _show("stage 2  length_penalty(360), (120)", lambda: (length_penalty(360), length_penalty(120)), "(0.048, 0.0)")
    _show("stage 2  repetition_penalty", lambda: repetition_penalty(torch.tensor([23, 23, 23, 5, 4, 3, 14, 10])),
          [0.375, 0.375, 0.375, 0.125, 0.125, 0.125, 0.125, 0.125])
    _show("stage 2  proposer_reward", lambda: proposer_reward(torch.tensor([0.4]), torch.tensor([0.048]),
                                                              torch.tensor([0.375])), -0.023)
    _show("stage 3  lower_half, the deltas kept", lambda: [p["delta"] for p in lower_half(
        [{"delta": 0.3}, {"delta": -0.1}, {"delta": 0.8}, {"delta": 0.2}])], [-0.1, 0.2])
    _show("stage 4  dpo_loss_ln", lambda: dpo_loss_ln(torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-3.0]),
                                                      torch.tensor([-2.0]), torch.tensor([3.0]), torch.tensor([3.0])), 0.234)
    _show("stage 4  group_advantage", lambda: group_advantage(torch.tensor([[1.0, 0.0, 0.0, 1.0]])),
          [0.866, -0.866, -0.866, 0.866])
    _show("stage 4  clipped_loss", lambda: clipped_loss(torch.tensor([-0.5, -2.0]), torch.tensor([-1.0, -1.0]),
                                                        torch.tensor([1.0, -1.0])), -0.2)

    def one_round():
        _, _, history = gzero(seed=0, rounds=1)
        return f"p(good) 0.502 -> {float(history[0]['p_good'].mean()):.3f}"
    _show("stage 5  one round of the loop", one_round, "p(good) 0.502 -> 0.657")
    print("\nWhen these match, run:  ./scripts/run_gzero.sh check")
