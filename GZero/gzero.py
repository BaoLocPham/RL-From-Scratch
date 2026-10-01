"""G-Zero, the paper's equations on the toy. Demo: ``python GZero/run_gzero.py``.

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

import torch
import torch.nn.functional as F

from gzero_env import HINT_CHARS, T, Generator, Proposer, hint_of, query_of

# ================================================================ the Proposer's reward


@torch.no_grad()
def hint_delta(generator, q, h, a_hard):
    """Hint-delta: how much the hint moves the Generator away from its own answer (Eq. 3).

        delta(q, h, a_hard) = (1/T) * sum_t [ log pi_G(a_t | q, a_<t) - log pi_G(a_t | q, h, a_<t) ]

    a_hard: (T,) the Generator's unassisted answer to q. It is scored twice:
    without the hint, then with it. delta > 0 when the hint makes that answer
    LESS likely -- the hint points somewhere the Generator was not going.
    Averaged per token (1/T), so a longer answer earns no more. It is a reward,
    so no gradient flows through it.
    """
    logp_alone = generator.token_logps(q, a_hard)                 # log pi_G(a_t | q, a_<t), (T,)
    logp_hinted = generator.token_logps(q, a_hard, hint=h)        # log pi_G(a_t | q, h, a_<t), (T,)
    return float((logp_alone - logp_hinted).mean())               # delta = (1/T) * sum_t [ ... ]


def length_penalty(hint_chars, lam=0.03):
    """Penalty for a long hint (Eq. 4).

        P_length = lambda * max(0, (|h| - 200) / 100),     lambda = 0.03

    Free up to 200 characters, then 0.03 per extra 100. A hint that spells out
    everything would move the Generator most; this stops it being free.
    """
    return lam * max(0.0, (hint_chars - 200) / 100)               # P_length


def repetition_penalty(outputs):
    """Penalty for picking what the rest of the batch also picked.

        P_BLEU(x_i) = |C_i| / |B|

    The paper clusters the batch's questions by BLEU similarity; C_i is the
    cluster x_i falls in. Here an output is only a (query, hint) id, so a
    cluster is every copy of the same output. outputs: (B,). Returns (B,).
    """
    cluster_size = torch.bincount(outputs)[outputs]               # |C_i|: copies of this output in the batch
    return cluster_size.float() / len(outputs)                    # |C_i| / |B|


def proposer_reward(delta, p_length, p_bleu):
    """The Proposer's reward (Eq. 5).

        r(q, h) = delta(q, h, a_hard) - P_length - P_BLEU

    All three are (B,). No max(0, .): a hint about something the Generator
    already knows can earn less than nothing.
    """
    return delta - p_length - p_bleu                              # r = delta - P_length - P_BLEU


# ================================================================ the Generator's data and loss


def make_pair(generator, q, h):
    """One DPO pair for query q and hint h.

        chosen   y_w = a_assisted ~ pi_G(. | q, h)      the answer WITH the hint
        rejected y_l = a_hard     ~ pi_G(. | q)         the answer without it

    Also returns delta for this pair (Eq. 3, on the fresh a_hard), for the filter.
    """
    a_assisted = generator.answer(q, hint=h)                      # y_w ~ pi_G(. | q, h)
    a_hard = generator.answer(q)                                  # y_l ~ pi_G(. | q)
    return {"q": q, "h": h, "chosen": a_assisted, "rejected": a_hard,
            "delta": hint_delta(generator, q, h, a_hard)}


def lower_half(pairs):
    """Keep the pairs in the lower 50% of delta.

    The paper's reason: on an LLM, a very high-delta pair is far off the
    Generator's distribution and breaks DPO's implicit KL budget. Here it is
    applied as the paper does it; the demo shows what it costs on this toy.
    """
    ranked = sorted(pairs, key=lambda pair: pair["delta"])        # lowest delta first
    return ranked[:len(pairs) // 2]                                # the lower 50%


def dpo_loss_ln(logp_w, ref_logp_w, logp_l, ref_logp_l, len_w, len_l, beta=2.0):
    """Length-normalised DPO (Eq. 6), averaged over the pairs.

        L = -log sigmoid( beta * (r_bar(x, y_w) - r_bar(x, y_l)) ),
        r_bar(x, y) = (1/|y|) * log( pi_theta(y | x) / pi_ref(y | x) ),     beta = 2.0

    Inputs are (n,): each answer's summed log-probability under the Generator
    being trained (logp_*) and under pi_ref, the Generator frozen at the start
    of the round (ref_logp_*), and each answer's length |y|. The prompt x is the
    query ALONE: no hint.
    """
    r_bar_w = (logp_w - ref_logp_w) / len_w                       # r_bar(x, y_w): per-token log-ratio of the chosen
    r_bar_l = (logp_l - ref_logp_l) / len_l                       # r_bar(x, y_l): the same for the rejected
    return -F.logsigmoid(beta * (r_bar_w - r_bar_l)).mean()       # -log sigmoid(beta * (r_bar_w - r_bar_l))


# ================================================================ GRPO, for the Proposer


def group_advantage(rewards, eps=1e-6):
    """GRPO's advantage: each output against the others in its group of K.

        A_i = (r_i - mean(r_group)) / std(r_group)

    rewards: (groups, K). Returns (groups, K).
    """
    mean = rewards.mean(dim=1, keepdim=True)                      # mean(r_group): (groups, 1)
    std = rewards.std(dim=1, keepdim=True)                        # std(r_group), sample std: (groups, 1)
    return (rewards - mean) / (std + eps)                         # A_i


def clipped_loss(logp, old_logp, advantages, eps=0.2):
    """PPO's clipped loss, as GRPO uses it (negated, to minimise).

        L = -(1/K) * sum_i min( r_i * A_i, clip(r_i, 1 - eps, 1 + eps) * A_i ),    r_i = pi_P / pi_P_old

    All inputs are (n,).
    """
    ratio = torch.exp(logp - old_logp)                            # r_i = pi_P / pi_P_old
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)            # clip(r_i, 1 - eps, 1 + eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()   # -(1/K) sum_i min(...)


# ================================================================ the loop


@torch.no_grad()
def score_outputs(outputs, generator, use_bleu=True):
    """Phase 1's reward for a batch of Proposer outputs: every term of Eq. 5, each (B,).

    For each (q, h), the frozen Generator answers q once without the hint, and
    hint_delta scores that answer with and without h.
    """
    delta = torch.tensor([hint_delta(generator, query_of(o), hint_of(o), generator.answer(query_of(o)))
                          for o in outputs.tolist()])                              # delta(q, h, a_hard)   (Eq. 3)
    p_length = torch.tensor([length_penalty(HINT_CHARS[hint_of(o)]) for o in outputs.tolist()])   # (Eq. 4)
    p_bleu = repetition_penalty(outputs) if use_bleu else torch.zeros(len(outputs))     # P_BLEU
    return {"delta": delta, "p_length": p_length, "p_bleu": p_bleu,
            "r": proposer_reward(delta, p_length, p_bleu)}                             # r (Eq. 5)


def train_proposer(proposer, generator, optimizer, steps=30, groups=4, k=16, epochs=2, use_bleu=True):
    """Phase 1: GRPO on the Proposer's 24 logits. The Generator is frozen.

    Each step writes 4 groups of K = 16 outputs (the paper's K), and GRPO
    compares each output with the others in its group.
    """
    for _ in range(steps):
        with torch.no_grad():
            outputs = proposer.write(groups * k)                              # (q, h) ~ pi_P, (B,)
            r = score_outputs(outputs, generator, use_bleu)["r"]              # r(q, h), (B,)
            advantages = group_advantage(r.view(groups, k)).view(-1)          # A_i within each group of K
            old_logp = proposer.log_prob(outputs)                             # log pi_P_old, frozen
        for _ in range(epochs):
            loss = clipped_loss(proposer.log_prob(outputs), old_logp, advantages)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def collect_pairs(proposer, generator, n=200, use_filter=True):
    """Phase 2's data: N (q, h) from the frozen Proposer, a DPO pair for each, then the lower-50% filter."""
    pairs = [make_pair(generator, query_of(o), hint_of(o)) for o in proposer.write(n).tolist()]
    return lower_half(pairs) if use_filter else pairs


def train_generator(generator, pairs, steps=50, batch=8, beta=2.0, lr=2.0):
    """Phase 2: length-normalised DPO on the Generator's unassisted table. The Proposer is frozen.

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
        logp_w = torch.stack([generator.token_logps(p["q"], p["chosen"]).sum() for p in chosen])      # log pi_theta(y_w | q)
        logp_l = torch.stack([generator.token_logps(p["q"], p["rejected"]).sum() for p in chosen])    # log pi_theta(y_l | q)
        with torch.no_grad():
            ref_w = torch.stack([reference.token_logps(p["q"], p["chosen"]).sum() for p in chosen])   # log pi_ref(y_w | q)
            ref_l = torch.stack([reference.token_logps(p["q"], p["rejected"]).sum() for p in chosen])  # log pi_ref(y_l | q)
        length = torch.full((batch,), float(T))                                                         # |y| = 3 tokens
        loss = dpo_loss_ln(logp_w, ref_w, logp_l, ref_l, length, length, beta)                          # Eq. 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def gzero(seed=0, rounds=2, use_filter=True, use_bleu=True, report=None):
    """The whole loop: Phase 1 then Phase 2, `rounds` times (the paper's 2).

    Returns the trained (proposer, generator) and one record per round.
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
